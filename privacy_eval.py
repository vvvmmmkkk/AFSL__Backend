"""
Privacy Shield Evaluation — Face Recognition Evasion Test

Tests whether adversarial perturbations from the privacy filter can
prevent face recognition models from matching protected images to originals.

Usage:
    python privacy_eval.py --image_dir <path_to_face_images> [--strength medium] [--max_images 50]

Requires: facenet-pytorch, scikit-image
    pip install facenet-pytorch scikit-image
"""

import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import DEVICE, IMAGE_SIZE
from attacks.cw import EnsembleFeatureExtractor, privacy_attack

# -------------------------------------------------------------------
# Strength presets (same as server.py)
# -------------------------------------------------------------------
STRENGTH_PRESETS = {
    "low":    {"epsilon": 4/255,  "num_iterations": 200, "alpha": 0.3/255,
               "momentum": 0.9, "tv_weight": 0.15, "smooth_sigma": 1.5,
               "smooth_kernel": 7, "use_input_diversity": True},
    "medium": {"epsilon": 8/255,  "num_iterations": 300, "alpha": 0.4/255,
               "momentum": 0.9, "tv_weight": 0.1, "smooth_sigma": 1.5,
               "smooth_kernel": 7, "use_input_diversity": True},
    "high":   {"epsilon": 12/255, "num_iterations": 400, "alpha": 0.5/255,
               "momentum": 0.9, "tv_weight": 0.08, "smooth_sigma": 1.0,
               "smooth_kernel": 5, "use_input_diversity": True},
}


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor (1, C, H, W) in [0, 1]."""
    array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor (1, C, H, W) in [0, 1] to numpy HWC uint8."""
    return (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def load_face_recognition_model():
    """Load FaceNet (InceptionResnetV1) for face recognition evaluation."""
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError:
        print("ERROR: facenet-pytorch not installed.")
        print("Install with: pip install facenet-pytorch")
        sys.exit(1)

    model = InceptionResnetV1(pretrained='vggface2').to(DEVICE).eval()
    print(f"✓ FaceNet (VGGFace2) loaded on {DEVICE}")
    return model


def get_facenet_embedding(model, image_np: np.ndarray) -> torch.Tensor:
    """
    Get FaceNet embedding for a face image.
    image_np: HWC uint8 numpy array
    Returns: (1, 512) embedding tensor
    """
    # FaceNet expects 160x160 input in [-1, 1]
    img = Image.fromarray(image_np).resize((160, 160))
    tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 160, 160)
    tensor = (tensor - 0.5) / 0.5  # normalize to [-1, 1]
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        embedding = model(tensor)
    return embedding


def compute_image_quality(original_np: np.ndarray, protected_np: np.ndarray) -> dict:
    """Compute SSIM, PSNR, L2, and L-inf between original and protected images."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    # Both should be uint8 HWC
    ssim = structural_similarity(
        original_np, protected_np, channel_axis=2, data_range=255
    )
    psnr = peak_signal_noise_ratio(original_np, protected_np, data_range=255)

    # L2 and L-inf on normalized [0, 1]
    orig_f = original_np.astype(np.float32) / 255.0
    prot_f = protected_np.astype(np.float32) / 255.0
    diff = orig_f - prot_f
    l2 = np.sqrt(np.sum(diff ** 2))
    linf = np.max(np.abs(diff))

    return {
        "ssim": round(ssim, 4),
        "psnr": round(psnr, 2),
        "l2": round(l2, 4),
        "linf": round(linf, 4),
    }


def collect_images(image_dir: str, max_images: int) -> list:
    """Collect image paths from a directory."""
    image_dir = Path(image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [
        p for p in sorted(image_dir.rglob("*"))
        if p.suffix.lower() in extensions
    ]
    if len(images) > max_images:
        # Evenly sample
        step = len(images) // max_images
        images = images[::step][:max_images]
    return images


def run_evaluation(args):
    """Main evaluation loop."""
    print("=" * 60)
    print("PRIVACY SHIELD EVALUATION")
    print("=" * 60)
    print(f"Image directory : {args.image_dir}")
    print(f"Strength preset : {args.strength}")
    print(f"Max images      : {args.max_images}")
    print(f"Device          : {DEVICE}")
    print()

    # Load models
    print("Loading models...")
    facenet = load_face_recognition_model()
    ensemble = EnsembleFeatureExtractor(device=DEVICE)
    print(f"✓ Ensemble (ResNet50 + VGG16 + DenseNet121) loaded on {DEVICE}")
    print()

    # Collect images
    image_paths = collect_images(args.image_dir, args.max_images)
    print(f"Found {len(image_paths)} images to evaluate")
    print()

    params = STRENGTH_PRESETS[args.strength]

    # Results accumulators
    results = []
    cos_sims_before = []
    cos_sims_after = []
    ssims = []
    psnrs = []
    l2s = []
    linfs = []
    attack_successes = 0
    total_processed = 0
    match_threshold = 0.6  # Standard FaceNet matching threshold

    output_dir = PROJECT_ROOT / "eval_results" / "privacy_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}] Processing: {img_path.name}")

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ✗ Failed to load: {e}")
            continue

        # Convert to tensor
        image_tensor = pil_to_tensor(image)
        original_np = tensor_to_numpy(image_tensor)

        # Get original FaceNet embedding
        emb_original = get_facenet_embedding(facenet, original_np)

        # Apply privacy filter
        t_start = time.time()
        protected_tensor, attack_stats = privacy_attack(
            ensemble=ensemble,
            image_tensor=image_tensor,
            **params,
        )
        elapsed = time.time() - t_start

        protected_np = tensor_to_numpy(protected_tensor)

        # Get protected FaceNet embedding
        emb_protected = get_facenet_embedding(facenet, protected_np)

        # Compute face recognition similarity
        cos_sim = F.cosine_similarity(emb_original, emb_protected, dim=1).item()

        # A match means the face is still recognized (attack failed)
        matched = cos_sim > match_threshold
        if not matched:
            attack_successes += 1

        # Compute image quality metrics
        quality = compute_image_quality(original_np, protected_np)

        total_processed += 1
        cos_sims_before.append(1.0)  # self-similarity
        cos_sims_after.append(cos_sim)
        ssims.append(quality["ssim"])
        psnrs.append(quality["psnr"])
        l2s.append(quality["l2"])
        linfs.append(quality["linf"])

        result = {
            "image": img_path.name,
            "facenet_cos_sim": round(cos_sim, 4),
            "matched": matched,
            "ensemble_cos_sim": attack_stats["cos_sim_adv"],
            "time_seconds": round(elapsed, 1),
            **quality,
        }
        results.append(result)

        status = "✗ MATCHED (attack failed)" if matched else "✓ EVADED (attack succeeded)"
        print(f"  FaceNet cos_sim: {cos_sim:.4f}  |  {status}")
        print(f"  SSIM: {quality['ssim']:.4f}  |  PSNR: {quality['psnr']:.1f} dB  |  Time: {elapsed:.1f}s")

        # Save sample images (first 5)
        if i < 5:
            Image.fromarray(original_np).save(output_dir / f"sample_{i}_original.jpg")
            Image.fromarray(protected_np).save(output_dir / f"sample_{i}_protected.jpg")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    if total_processed == 0:
        print("\nNo images were processed!")
        return

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Images processed    : {total_processed}")
    print(f"Strength preset     : {args.strength}")
    print(f"Epsilon             : {params['epsilon']*255:.0f}/255")
    print()

    # Face Recognition Evasion
    evasion_rate = attack_successes / total_processed * 100
    avg_cos_sim = np.mean(cos_sims_after)
    print(f"--- Face Recognition Evasion (FaceNet, threshold={match_threshold}) ---")
    print(f"  Evasion rate      : {evasion_rate:.1f}% ({attack_successes}/{total_processed})")
    print(f"  Avg cos similarity: {avg_cos_sim:.4f} (lower = better protection)")
    print(f"  Min cos similarity: {min(cos_sims_after):.4f}")
    print(f"  Max cos similarity: {max(cos_sims_after):.4f}")
    print()

    # Image Quality
    print(f"--- Image Quality ---")
    print(f"  Avg SSIM          : {np.mean(ssims):.4f}")
    print(f"  Avg PSNR          : {np.mean(psnrs):.2f} dB")
    print(f"  Avg L2            : {np.mean(l2s):.4f}")
    print(f"  Avg L∞            : {np.mean(linfs):.4f} ({np.mean(linfs)*255:.1f}/255)")
    print()

    # Save detailed results as JSON
    summary = {
        "config": {
            "strength": args.strength,
            "epsilon": f"{params['epsilon']*255:.0f}/255",
            "num_iterations": params["num_iterations"],
            "match_threshold": match_threshold,
            "device": DEVICE,
        },
        "face_recognition_evasion": {
            "evasion_rate_pct": round(evasion_rate, 2),
            "avg_cosine_similarity": round(avg_cos_sim, 4),
            "min_cosine_similarity": round(min(cos_sims_after), 4),
            "max_cosine_similarity": round(max(cos_sims_after), 4),
        },
        "image_quality": {
            "avg_ssim": round(np.mean(ssims), 4),
            "avg_psnr_db": round(np.mean(psnrs), 2),
            "avg_l2": round(np.mean(l2s), 4),
            "avg_linf": round(np.mean(linfs), 4),
        },
        "per_image_results": results,
    }

    output_file = output_dir / f"privacy_eval_{args.strength}.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to: {output_file}")

    # Also print a LaTeX-ready table row
    print("\n--- LaTeX Table Row ---")
    print(f"  {args.strength.capitalize()} ({params['epsilon']*255:.0f}/255) "
          f"& {evasion_rate:.1f}\\% "
          f"& {avg_cos_sim:.4f} "
          f"& {np.mean(ssims):.4f} "
          f"& {np.mean(psnrs):.2f} \\\\")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Privacy Shield Evaluation")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing face images to test")
    parser.add_argument("--strength", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Privacy filter strength preset")
    parser.add_argument("--max_images", type=int, default=50,
                        help="Maximum number of images to evaluate")
    args = parser.parse_args()
    run_evaluation(args)
