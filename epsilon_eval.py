"""
Multi-Epsilon Evaluation — Protection vs. Quality Trade-off

Tests the privacy filter at multiple epsilon values to show the trade-off
between protection strength and image quality.

Usage:
    python epsilon_eval.py --image_dir <path_to_face_images> [--max_images 20]

Requires: facenet-pytorch, scikit-image
"""

import sys
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as torchF
import numpy as np
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from config import DEVICE, IMAGE_SIZE
from attacks.cw import EnsembleFeatureExtractor, privacy_attack


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


# Epsilon values to test
EPSILON_CONFIGS = [
    {"label": "2/255",  "epsilon": 2/255,  "num_iterations": 200, "alpha": 0.2/255},
    {"label": "4/255",  "epsilon": 4/255,  "num_iterations": 200, "alpha": 0.3/255},
    {"label": "6/255",  "epsilon": 6/255,  "num_iterations": 250, "alpha": 0.35/255},
    {"label": "8/255",  "epsilon": 8/255,  "num_iterations": 300, "alpha": 0.4/255},
    {"label": "10/255", "epsilon": 10/255, "num_iterations": 300, "alpha": 0.45/255},
    {"label": "12/255", "epsilon": 12/255, "num_iterations": 350, "alpha": 0.5/255},
    {"label": "16/255", "epsilon": 16/255, "num_iterations": 400, "alpha": 0.6/255},
]

# Shared params
SHARED_PARAMS = {
    "momentum": 0.9,
    "tv_weight": 0.1,
    "smooth_sigma": 1.5,
    "smooth_kernel": 7,
    "use_input_diversity": True,
}


def run_epsilon_eval(args):
    print("=" * 60)
    print("MULTI-EPSILON TRADE-OFF ANALYSIS")
    print("=" * 60)
    print(f"Image directory : {args.image_dir}")
    print(f"Max images      : {args.max_images}")
    print(f"Epsilon values  : {len(EPSILON_CONFIGS)}")
    print(f"Device          : {DEVICE}")
    print()

    # Load models
    from facenet_pytorch import InceptionResnetV1
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio

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

    # Pre-load images
    images_data = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            tensor = pil_to_tensor(img)
            original_np = tensor_to_numpy(tensor)
            images_data.append((p.name, tensor, original_np))
        except Exception:
            continue

    def get_embedding(img_np):
        img = Image.fromarray(img_np).resize((160, 160))
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
        t = t.permute(2, 0, 1).unsqueeze(0)
        t = (t - 0.5) / 0.5
        t = t.to(DEVICE)
        with torch.no_grad():
            return facenet(t)

    # Run each epsilon
    all_results = {}
    output_dir = PROJECT_ROOT / "eval_results" / "epsilon_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    for eps_config in EPSILON_CONFIGS:
        label = eps_config["label"]
        config = {
            "epsilon": eps_config["epsilon"],
            "num_iterations": eps_config["num_iterations"],
            "alpha": eps_config["alpha"],
            **SHARED_PARAMS,
        }

        print(f"\n{'='*50}")
        print(f"Epsilon: {label}")
        print(f"{'='*50}")

        cos_sims = []
        ssims = []
        psnrs = []
        evasions = 0

        for img_name, image_tensor, original_np in tqdm(images_data, desc=f"ε={label}"):
            try:
                protected_tensor, stats = privacy_attack(
                    ensemble=ensemble,
                    image_tensor=image_tensor,
                    **config,
                )
                protected_np = tensor_to_numpy(protected_tensor)

                emb_orig = get_embedding(original_np)
                emb_prot = get_embedding(protected_np)
                cos_sim = torchF.cosine_similarity(emb_orig, emb_prot, dim=1).item()

                ssim = structural_similarity(original_np, protected_np, channel_axis=2, data_range=255)
                psnr = peak_signal_noise_ratio(original_np, protected_np, data_range=255)

                cos_sims.append(cos_sim)
                ssims.append(ssim)
                psnrs.append(psnr)
                if cos_sim < 0.6:
                    evasions += 1
            except Exception as e:
                print(f"  Error: {e}")
                continue

        if cos_sims:
            n = len(cos_sims)
            summary = {
                "epsilon": label,
                "num_images": n,
                "evasion_rate_pct": round(evasions / n * 100, 1),
                "avg_cos_sim": round(np.mean(cos_sims), 4),
                "avg_ssim": round(np.mean(ssims), 4),
                "avg_psnr_db": round(np.mean(psnrs), 2),
            }
            all_results[label] = summary

            print(f"  Evasion rate : {summary['evasion_rate_pct']}%")
            print(f"  Avg cos sim  : {summary['avg_cos_sim']}")
            print(f"  Avg SSIM     : {summary['avg_ssim']}")
            print(f"  Avg PSNR     : {summary['avg_psnr_db']} dB")

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EPSILON SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'Epsilon':<10} {'Evasion%':>10} {'CosSim':>10} {'SSIM':>10} {'PSNR':>10}")
    print("-" * 50)
    for label, s in all_results.items():
        print(f"{label:<10} {s['evasion_rate_pct']:>9.1f}% "
              f"{s['avg_cos_sim']:>10.4f} {s['avg_ssim']:>10.4f} "
              f"{s['avg_psnr_db']:>9.2f}")

    # LaTeX table
    print("\n--- LaTeX Table ---")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("\\textbf{$\\epsilon$} & \\textbf{Evasion Rate} & \\textbf{Cos Sim $\\downarrow$} & \\textbf{SSIM $\\uparrow$} & \\textbf{PSNR (dB) $\\uparrow$} \\\\")
    print("\\midrule")
    for label, s in all_results.items():
        print(f"${label}$ & {s['evasion_rate_pct']:.1f}\\% & {s['avg_cos_sim']:.4f} "
              f"& {s['avg_ssim']:.4f} & {s['avg_psnr_db']:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # Save JSON
    output_file = output_dir / "epsilon_sweep_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Epsilon Evaluation")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing face images")
    parser.add_argument("--max_images", type=int, default=20,
                        help="Max images per epsilon (20 recommended — 7 epsilons × 20 = 140 runs)")
    args = parser.parse_args()
    run_epsilon_eval(args)
