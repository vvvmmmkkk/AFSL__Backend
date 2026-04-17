"""
FastAPI server for deepfake detection using AFSL and Baseline models.
Also provides a privacy filter using the Carlini-Wagner L2 attack.
Run with: uvicorn server:app --reload --port 8000
"""

import sys
import base64
import io
from pathlib import Path
from typing import Literal, Optional
from contextlib import asynccontextmanager

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mtcnn import MTCNN

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from models.detector import Detector
from config import DEVICE, RUNS_DIR, IMAGE_SIZE, NORM_MEAN, NORM_STD, DECISION_THRESHOLD
from torchvision import transforms

# ============== App Setup ==============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and face detector when server starts."""
    load_models()
    yield

app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfakes using AFSL and Baseline models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://afsl-frontend-git-main-vishnus-projects-6def863b.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== Models ==============

# Model paths
BASELINE_MODEL_PATH = RUNS_DIR / "tinker_baseline_epoch_5.pth"
AFSL_MODEL_PATH = RUNS_DIR / "afsl_epoch_6.pth"

# Global model instances
face_detector = None
baseline_model = None
afsl_model = None
ensemble_model = None  # For privacy filter


def load_models():
    """Load both models, face detector, and ensemble for privacy filter."""
    global baseline_model, afsl_model, face_detector, ensemble_model
    
    print(f"Loading models on device: {DEVICE}")
    
    # Load MTCNN face detector
    print("Loading MTCNN face detector...")
    face_detector = MTCNN()
    print("✓ MTCNN face detector loaded")
    
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
    
    # Load ensemble feature extractors for privacy filter
    print("Loading ensemble feature extractors (ResNet-50, VGG-16, DenseNet-121)...")
    from attacks.cw import EnsembleFeatureExtractor
    ensemble_model = EnsembleFeatureExtractor(device=DEVICE)
    print("✓ Ensemble feature extractors loaded")


# ============== Image Processing ==============

def get_inference_transforms():
    """Get transforms for inference (same as training)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


MAX_IMAGE_SIZE_MB = 10

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode a base64 image string to PIL Image."""
    # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
    if "," in base64_string:
        base64_string = base64_string.split(",", 1)[1]
    
    image_bytes = base64.b64decode(base64_string)
    
    # Reject images larger than MAX_IMAGE_SIZE_MB
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise ValueError(f"Image too large ({size_mb:.1f} MB). Maximum is {MAX_IMAGE_SIZE_MB} MB.")
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def extract_face(image: Image.Image) -> Optional[Image.Image]:
    """
    Detect and crop the largest face from a PIL Image using MTCNN.
    Returns the cropped face as a PIL Image, or None if no face is found.
    This matches the training preprocessing (tools/face_preprocess.py).
    """
    rgb_array = np.array(image)
    
    try:
        detections = face_detector.detect_faces(rgb_array)
    except Exception:
        return None
    
    if not detections:
        return None
    
    # Pick the largest face (same logic as face_preprocess.py)
    largest = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = largest["box"]
    
    # Clamp to image boundaries
    x, y = max(0, x), max(0, y)
    x2 = min(x + w, rgb_array.shape[1])
    y2 = min(y + h, rgb_array.shape[0])
    
    face_crop = rgb_array[y:y2, x:x2]
    
    if face_crop.size == 0:
        return None
    
    return Image.fromarray(face_crop)


def predict(model: torch.nn.Module, image: Image.Image) -> dict:
    """Run inference on a single image."""
    transform = get_inference_transforms()
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, logits = model(img_tensor)
        probability = torch.sigmoid(logits).item()
    
    # Label: 1 = Real, 0 = Fake
    is_real = probability > DECISION_THRESHOLD
    
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
    Flips the prediction by maximizing loss against the opposite of current prediction.
    """
    model.eval()
    image_tensor = image_tensor.clone().detach().requires_grad_(True)
    
    _, logits = model(image_tensor)
    prob = torch.sigmoid(logits)
    # Flip the prediction to create a target that maximizes loss
    target = 1.0 - (prob > 0.5).float()
    
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
    Determines flipped target once, then iterates to maximize loss.
    """
    if alpha is None:
        alpha = epsilon / 4
    
    model.eval()
    original = image_tensor.clone().detach()
    perturbed = image_tensor.clone().detach()
    
    # Determine target once (flip original prediction)
    with torch.no_grad():
        _, logits_orig = model(original)
        target = 1.0 - (torch.sigmoid(logits_orig) > 0.5).float()
    
    for _ in range(steps):
        perturbed.requires_grad_(True)
        _, logits = model(perturbed)
        
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
    face_detected: bool
    cropped_face: Optional[str] = None  # Base64 encoded cropped face



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
    Extracts the face from the image first (matching training preprocessing).
    
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
    
    # Extract face from the image (matches training pipeline)
    face_image = extract_face(image)
    
    if face_image is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image. Please upload an image containing a clearly visible face."
        )
    
    # Run predictions on the cropped face
    baseline_result = predict(baseline_model, face_image)
    afsl_result = predict(afsl_model, face_image)
    
    return {
        "baseline": baseline_result,
        "afsl": afsl_result,
        "face_detected": True,
        "cropped_face": encode_image_base64(face_image),
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
    Extracts the face first, then applies attacks to the cropped face.
    
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
    
    # Extract face from the image (matches training pipeline)
    face_image = extract_face(original_image)
    
    if face_image is None:
        raise HTTPException(
            status_code=422,
            detail="No face detected in the image. Please upload an image containing a clearly visible face."
        )
    
    # Get predictions on original face crop
    baseline_original = predict(baseline_model, face_image)
    afsl_original = predict(afsl_model, face_image)
    
    # Generate adversarial image based on attack type (on the face crop)
    attack_type = request.attack_type
    epsilon = request.epsilon
    
    if attack_type == "blur":
        # Blur attack works directly on PIL image
        adversarial_image = blur_attack(face_image, epsilon)
    else:
        # Convert face crop to tensor for gradient-based attacks
        image_tensor = pil_to_tensor(face_image)
        
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

# ============== Privacy Filter (Ensemble Transferable Attack) ==============

import json
import asyncio
import threading
from fastapi.responses import StreamingResponse
from attacks.cw import privacy_attack

# Strength presets: tuned for smooth, imperceptible perturbation.
# Gradient smoothing keeps noise natural; lower TV weight preserves attack power.
STRENGTH_PRESETS = {
    "low":    {"epsilon": 4/255,  "num_iterations": 100, "alpha": 0.3/255,
               "momentum": 0.9, "tv_weight": 0.15, "smooth_sigma": 1.5,
               "smooth_kernel": 7, "use_input_diversity": True},
    "medium": {"epsilon": 8/255,  "num_iterations": 150, "alpha": 0.4/255,
               "momentum": 0.9, "tv_weight": 0.1, "smooth_sigma": 1.5,
               "smooth_kernel": 7, "use_input_diversity": True},
    "high":   {"epsilon": 12/255, "num_iterations": 200, "alpha": 0.5/255,
               "momentum": 0.9, "tv_weight": 0.08, "smooth_sigma": 1.0,
               "smooth_kernel": 5, "use_input_diversity": True},
}


class PrivacyFilterRequest(BaseModel):
    image: str  # Base64 encoded image
    strength: Literal["low", "medium", "high"] = "medium"


@app.post("/api/privacy-filter")
async def privacy_filter_endpoint(request: PrivacyFilterRequest):
    """
    Apply a transferable ensemble adversarial attack to protect an image
    from reverse image search and face-recognition systems.

    Uses ResNet-50 + VGG-16 + DenseNet-121 ensemble with input diversity
    and momentum for maximum transferability to unknown models.

    Returns an SSE stream with progress events and a final result event.
    """
    try:
        image = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    original_size = image.size  # (width, height) — preserve for later

    # Resize to 224×224 for the ensemble models
    image_tensor = pil_to_tensor(image)  # (1, 3, 224, 224) in [0, 1]

    params = STRENGTH_PRESETS[request.strength]

    # --- SSE streaming with real progress ---
    progress_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    result_holder = {}

    def progress_callback(current, total, info):
        """Called by the attack from the worker thread."""
        pct = min(round(current / total * 100), 100)
        asyncio.run_coroutine_threadsafe(
            progress_queue.put({
                "type": "progress",
                "percent": pct,
                "cos_sim": info.get("cos_sim"),
                "l2": info.get("l2"),
                "step": f"Iteration {info.get('iteration')}",
            }),
            loop,
        )

    def run_attack():
        """Run the ensemble attack (blocking) in a worker thread."""
        try:
            protected_tensor, stats = privacy_attack(
                ensemble=ensemble_model,
                image_tensor=image_tensor,
                progress_callback=progress_callback,
                **params,
            )

            # --- Preserve original aspect ratio ---
            with torch.no_grad():
                delta_224 = protected_tensor - image_tensor

                orig_h, orig_w = original_size[1], original_size[0]
                delta_full = torch.nn.functional.interpolate(
                    delta_224,
                    size=(orig_h, orig_w),
                    mode="bilinear",
                    align_corners=False,
                )

                original_full = np.array(image).astype(np.float32) / 255.0
                original_full_t = torch.from_numpy(original_full).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                protected_full_t = torch.clamp(original_full_t + delta_full, 0, 1)

                protected_array = (protected_full_t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                protected_pil = Image.fromarray(protected_array)

            result_holder["result"] = {
                "protected_image": encode_image_base64(protected_pil),
                "l2_distance": round(stats["l2_distance"], 4),
                "original_similarity": stats["cos_sim_orig"],
                "protected_similarity": round(stats["cos_sim_adv"], 4),
            }
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result_holder["error"] = str(exc)

        asyncio.run_coroutine_threadsafe(progress_queue.put(None), loop)

    thread = threading.Thread(target=run_attack, daemon=True)
    thread.start()

    async def event_stream():
        while True:
            msg = await progress_queue.get()
            if msg is None:
                if "error" in result_holder:
                    yield f"data: {json.dumps({'type': 'error', 'detail': result_holder['error']})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'result', **result_holder['result']})}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
