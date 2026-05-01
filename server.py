"""
FastAPI server for deepfake detection using AFSL and Baseline models.
Run with: uvicorn server:app --reload --port 8000
"""

import sys
import base64
import io
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.detector import Detector
from config import DEVICE, RUNS_DIR, IMAGE_SIZE, NORM_MEAN, NORM_STD
from torchvision import transforms

# ============== App Setup ==============

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfakes using AFSL and Baseline models",
    version="1.0.0"
)

# CORS middleware to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============== Models ==============

# Model paths
BASELINE_MODEL_PATH = RUNS_DIR / "tinker_baseline_epoch_5.pth"
AFSL_MODEL_PATH = RUNS_DIR / "afsl_epoch_6.pth"

# Global model instances
baseline_model = None
afsl_model = None


def load_models():
    """Load both models into memory."""
    global baseline_model, afsl_model
    
    print(f"Loading models on device: {DEVICE}")
    
    # Load Baseline model
    print(f"Loading Baseline model from: {BASELINE_MODEL_PATH}")
    baseline_model = Detector()
    baseline_model.load_state_dict(torch.load(BASELINE_MODEL_PATH, map_location=DEVICE))
    baseline_model.to(DEVICE)
    baseline_model.eval()
    print("✓ Baseline model loaded")
    
    # Load AFSL model
    print(f"Loading AFSL model from: {AFSL_MODEL_PATH}")
    afsl_model = Detector()
    afsl_model.load_state_dict(torch.load(AFSL_MODEL_PATH, map_location=DEVICE))
    afsl_model.to(DEVICE)
    afsl_model.eval()
    print("✓ AFSL model loaded")


# ============== Image Processing ==============

def get_inference_transforms():
    """Get transforms for inference (same as training)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64 image string to PIL Image."""
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def predict(model: torch.nn.Module, image: Image.Image) -> dict:
    """Run inference on a single image."""
    transform = get_inference_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, logits = model(img_tensor)
        probability = torch.sigmoid(logits).item()
    
    # Label: 1 = Real, 0 = Fake (threshold 0.6 as per evaluate.py)
    is_real = probability > 0.6
    
    # Confidence is how sure the model is of its prediction
    confidence = probability if is_real else (1 - probability)
    
    return {
        "label": "Real" if is_real else "Fake",
        "confidence": round(confidence, 4)
    }


def encode_image_base64(image: Image.Image) -> str:
    """Encode a PIL Image to base64 string with data URI prefix."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    b64_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_string}"


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a tensor (C, H, W) with values in [0, 1] to PIL Image."""
    tensor = tensor.clamp(0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor (1, C, H, W) with values in [0, 1]."""
    array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


# ============== Adversarial Attacks ==============

def fgsm_attack(model: torch.nn.Module, image_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Fast Gradient Sign Method (FGSM) attack.
    Tries to flip the prediction by adding perturbation in the gradient direction.
    """
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    
    _, logits = model(image_tensor)
    # Use current prediction as target (attack wants to flip it)
    prob = torch.sigmoid(logits)
    target = (prob > 0.5).float()
    
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), target.view(-1))
    loss.backward()
    
    grad_sign = image_tensor.grad.sign()
    # Add perturbation in gradient direction to maximize loss
    perturbed = image_tensor + epsilon * grad_sign
    perturbed = torch.clamp(perturbed, 0, 1)
    
    return perturbed.detach()


def pgd_attack_adversarial(model: torch.nn.Module, image_tensor: torch.Tensor, 
                           epsilon: float, alpha: float = None, steps: int = 10) -> torch.Tensor:
    """
    Projected Gradient Descent (PGD) attack - iterative FGSM.
    """
    if alpha is None:
        alpha = epsilon / 4
    
    model.eval()
    original = image_tensor.clone().detach()
    perturbed = image_tensor.clone().detach()
    
    for _ in range(steps):
        perturbed.requires_grad_(True)
        _, logits = model(perturbed)
        prob = torch.sigmoid(logits)
        target = (prob > 0.5).float()
        
        loss = F.binary_cross_entropy_with_logits(logits.view(-1), target.view(-1))
        grad = torch.autograd.grad(loss, perturbed)[0]
        
        perturbed = perturbed + alpha * grad.sign()
        # Project back to epsilon ball
        delta = torch.clamp(perturbed - original, -epsilon, epsilon)
        perturbed = torch.clamp(original + delta, 0, 1).detach()
    
    return perturbed


def gaussian_noise_attack(image_tensor: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Add random Gaussian noise to the image.
    """
    noise = torch.randn_like(image_tensor) * epsilon
    perturbed = image_tensor + noise
    return torch.clamp(perturbed, 0, 1)


def blur_attack(image: Image.Image, epsilon: float) -> Image.Image:
    """
    Apply Gaussian blur to the image.
    Epsilon controls blur radius (scaled to reasonable range).
    """
    # Map epsilon (0.001-0.1) to blur radius (0.1-5), capped to preserve image structure
    blur_radius = max(0.1, min(epsilon * 50, 5))
    return image.filter(ImageFilter.GaussianBlur(radius=blur_radius))


# ============== API Endpoints ==============

class PredictRequest(BaseModel):
    image: str  # Base64 encoded image


class PredictionResult(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    baseline: PredictionResult
    afsl: PredictionResult


@app.on_event("startup")
async def startup_event():
    """Load models when server starts."""
    load_models()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Deepfake Detection API is running",
        "models": {
            "baseline": str(BASELINE_MODEL_PATH.name),
            "afsl": str(AFSL_MODEL_PATH.name)
        }
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Predict if an image is real or fake using both models.
    
    Args:
        request: JSON body with base64-encoded image
        
    Returns:
        Predictions from both baseline and AFSL models
    """
    try:
        # Decode image
        image = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Run predictions
    baseline_result = predict(baseline_model, image)
    afsl_result = predict(afsl_model, image)
    
    return {
        "baseline": baseline_result,
        "afsl": afsl_result
    }


# Adversarial endpoint models
class AdversarialRequest(BaseModel):
    image: str  # Base64 encoded image with data URI prefix
    attack_type: Literal["fgsm", "pgd", "gaussian", "blur"] = "fgsm"
    epsilon: float = Field(default=0.03, ge=0.001, le=0.1)


class ModelPredictions(BaseModel):
    original: PredictionResult
    adversarial: PredictionResult


class AdversarialResponse(BaseModel):
    adversarial_image: str  # Base64 encoded perturbed image
    baseline: ModelPredictions
    afsl: ModelPredictions


@app.post("/api/adversarial", response_model=AdversarialResponse)
async def adversarial_endpoint(request: AdversarialRequest):
    """
    Generate an adversarial image and compare model predictions.
    
    Args:
        request: JSON body with base64-encoded image, attack_type, and epsilon
        
    Returns:
        Adversarial image and predictions from both models on original + adversarial
    """
    try:
        # Decode image
        original_image = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    
    # Get predictions on original image
    baseline_original = predict(baseline_model, original_image)
    afsl_original = predict(afsl_model, original_image)
    
    # Generate adversarial image based on attack type
    attack_type = request.attack_type
    epsilon = request.epsilon
    
    if attack_type == "blur":
        # Blur attack works directly on PIL image
        adversarial_image = blur_attack(original_image, epsilon)
    else:
        # Convert to tensor for gradient-based attacks
        image_tensor = pil_to_tensor(original_image)
        
        if attack_type == "fgsm":
            perturbed_tensor = fgsm_attack(baseline_model, image_tensor, epsilon)
        elif attack_type == "pgd":
            perturbed_tensor = pgd_attack_adversarial(baseline_model, image_tensor, epsilon)
        elif attack_type == "gaussian":
            perturbed_tensor = gaussian_noise_attack(image_tensor, epsilon)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown attack type: {attack_type}")
        
        # Convert tensor back to PIL image
        adversarial_image = tensor_to_pil(perturbed_tensor.squeeze(0))
    
    # Get predictions on adversarial image
    baseline_adversarial = predict(baseline_model, adversarial_image)
    afsl_adversarial = predict(afsl_model, adversarial_image)
    
    # Encode adversarial image to base64
    adversarial_image_b64 = encode_image_base64(adversarial_image)
    
    return {
        "adversarial_image": adversarial_image_b64,
        "baseline": {
            "original": baseline_original,
            "adversarial": baseline_adversarial
        },
        "afsl": {
            "original": afsl_original,
            "adversarial": afsl_adversarial
        }
    }


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
