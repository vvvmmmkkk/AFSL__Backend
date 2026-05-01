"""
Ablation Study — Contribution of Each Privacy Filter Technique

Runs the privacy attack with each technique disabled to measure
its contribution to both attack effectiveness and image quality.

Usage:
    python ablation_eval.py --image_dir <path_to_face_images> [--max_images 20]

Requires: facenet-pytorch, scikit-image
"""

import sys
import argparse
import json
from typing import cast
from pathlib import Path

import torch
import torch.nn.functional as torchF
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import DEVICE, IMAGE_SIZE
from attacks.cw import (
    EnsembleFeatureExtractor,
    privacy_attack,
)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def compute_metrics(facenet, original_np, protected_np):
    """Compute face recognition evasion + image quality metrics."""
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

    # FaceNet embedding
    def get_embedding(img_np):
        img = Image.fromarray(img_np).resize((160, 160))
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = (t - 0.5) / 0.5
        t = t.to(DEVICE)
        with torch.no_grad():
            return facenet(t)

    emb_orig = get_embedding(original_np)
    emb_prot = get_embedding(protected_np)
    cos_sim = float(torchF.cosine_similarity(emb_orig, emb_prot, dim=1).item())

    ssim = cast(float, structural_similarity(original_np, protected_np, channel_axis=2, data_range=255))
    psnr = cast(float, peak_signal_noise_ratio(original_np, protected_np, data_range=255))

    return {
        "facenet_cos_sim": round(float(cos_sim), 4),
        "evasion": cos_sim < 0.6,
        "ssim": round(float(ssim), 4),
        "psnr": round(float(psnr), 2),
    }


# -------------------------------------------------------------------
# Ablation variants
# -------------------------------------------------------------------
# Base config (medium strength)
BASE_CONFIG = {
    "epsilon": 8/255,
    "num_iterations": 300,
    "alpha": 0.4/255,
    "momentum": 0.9,
    "tv_weight": 0.1,
    "smooth_sigma": 1.5,
    "smooth_kernel": 7,
    "use_input_diversity": True,
}

ABLATION_VARIANTS = {
    "full_method": {
        "description": "Full method (all techniques)",
        "config": {**BASE_CONFIG},
        "single_model": False,
    },
    "no_gradient_smoothing": {
        "description": "No gradient smoothing",
        "config": {**BASE_CONFIG, "smooth_sigma": 0.01, "smooth_kernel": 1},
        "single_model": False,
    },
    "no_tv_regularization": {
        "description": "No TV regularization",
        "config": {**BASE_CONFIG, "tv_weight": 0.0},
        "single_model": False,
    },
    "no_color_regularization": {
        "description": "No color regularization",
        "config": {**BASE_CONFIG, "use_color_reg": False},
        "single_model": False,
    },
    "no_input_diversity": {
        "description": "No input diversity",
        "config": {**BASE_CONFIG, "use_input_diversity": False},
        "single_model": False,
    },
    "no_momentum": {
        "description": "No momentum",
        "config": {**BASE_CONFIG, "momentum": 0.0},
        "single_model": False,
    },
    "single_model_resnet": {
        "description": "Single model (ResNet50 only)",
        "config": {**BASE_CONFIG},
        "single_model": True,
    },
}


def run_ablation(args):
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    print(f"Image directory : {args.image_dir}")
    print(f"Max images      : {args.max_images}")
    print(f"Device          : {DEVICE}")
    print(f"Variants        : {len(ABLATION_VARIANTS)}")
    print()

    # Load models
    from facenet_pytorch import InceptionResnetV1
    facenet = InceptionResnetV1(pretrained='vggface2').to(DEVICE).eval()
    print("✓ FaceNet loaded")

    ensemble = EnsembleFeatureExtractor(device=DEVICE)
    print("✓ Ensemble loaded")

    # Collect images
    image_dir = Path(args.image_dir)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [p for p in sorted(image_dir.rglob("*")) if p.suffix.lower() in extensions]
    if len(image_paths) > args.max_images:
        step = len(image_paths) // args.max_images
        image_paths = image_paths[::step][:args.max_images]
    print(f"Using {len(image_paths)} images\n")

    # Pre-load and convert all images
    images_data = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = pil_to_tensor(img)
            original_np = tensor_to_numpy(tensor)
            images_data.append((p.name, tensor, original_np))
        except Exception:
            continue

    # Run each ablation variant
    all_results = {}
    output_dir = PROJECT_ROOT / "eval_results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    for variant_name, variant_info in ABLATION_VARIANTS.items():
        print(f"\n{'='*50}")
        print(f"Variant: {variant_info['description']}")
        print(f"{'='*50}")

        config = variant_info["config"]

        # Use single-model ensemble if specified
        if variant_info.get("single_model"):
            print("  Loading single-model ensemble (ResNet50 only)...")
            variant_ensemble = EnsembleFeatureExtractor(
                device=DEVICE, models_to_use=["resnet"]
            )
        else:
            variant_ensemble = ensemble

        variant_metrics = []

        for img_name, image_tensor, original_np in tqdm(images_data, desc=variant_name):
            try:
                protected_tensor, stats = privacy_attack(
                    ensemble=variant_ensemble,
                    image_tensor=image_tensor,
                    **config,
                )
                protected_np = tensor_to_numpy(protected_tensor)
                metrics = compute_metrics(facenet, original_np, protected_np)
                metrics["ensemble_cos_sim"] = stats["cos_sim_adv"]
                variant_metrics.append(metrics)
            except Exception as e:
                print(f"  Error on {img_name}: {e}")
                continue

        if variant_metrics:
            avg_cos_sim = float(np.mean([m["facenet_cos_sim"] for m in variant_metrics]))
            evasion_rate = sum(1 for m in variant_metrics if m["evasion"]) / len(variant_metrics) * 100
            avg_ssim = float(np.mean([m["ssim"] for m in variant_metrics]))
            avg_psnr = float(np.mean([m["psnr"] for m in variant_metrics]))

            summary = {
                "variant": variant_name,
                "description": variant_info["description"],
                "num_images": len(variant_metrics),
                "evasion_rate_pct": round(evasion_rate, 1),
                "avg_facenet_cos_sim": round(avg_cos_sim, 4),
                "avg_ssim": round(avg_ssim, 4),
                "avg_psnr_db": round(avg_psnr, 2),
            }
            all_results[variant_name] = summary

            print(f"  Evasion rate   : {evasion_rate:.1f}%")
            print(f"  Avg cos sim    : {avg_cos_sim:.4f}")
            print(f"  Avg SSIM       : {avg_ssim:.4f}")
            print(f"  Avg PSNR       : {avg_psnr:.2f} dB")

    # -------------------------------------------------------------------
    # Final summary table
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Variant':<30} {'Evasion%':>10} {'CosSim':>10} {'SSIM':>10} {'PSNR':>10}")
    print("-" * 70)
    for name, s in all_results.items():
        print(f"{s['description']:<30} {s['evasion_rate_pct']:>9.1f}% "
              f"{s['avg_facenet_cos_sim']:>10.4f} {s['avg_ssim']:>10.4f} "
              f"{s['avg_psnr_db']:>9.2f}")

    # LaTeX table
    print("\n--- LaTeX Table ---")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{Variant} & \\textbf{Evasion Rate} & \\textbf{Cos Sim} & \\textbf{SSIM} & \\textbf{PSNR (dB)} \\\\")
    print("\\midrule")
    for name, s in all_results.items():
        desc = s['description'].replace("&", "\\&")
        print(f"{desc} & {s['evasion_rate_pct']:.1f}\\% & {s['avg_facenet_cos_sim']:.4f} "
              f"& {s['avg_ssim']:.4f} & {s['avg_psnr_db']:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Save JSON
    output_file = output_dir / "ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing face images")
    parser.add_argument("--max_images", type=int, default=20,
                        help="Max images per variant (20 recommended — 5 variants × 20 = 100 runs)")
    args = parser.parse_args()
    run_ablation(args)
