"""
Transferable Adversarial Attack for Privacy Protection.

Uses an ensemble of pretrained models with smooth perturbation techniques
that are imperceptible to the human eye.

Key techniques:
  1. Ensemble of diverse architectures (ResNet, VGG, DenseNet)
  2. Gaussian-smoothed gradients → no per-pixel speckle noise
  3. Total variation (TV) regularisation → penalises high-frequency patterns
  4. Input diversity: random resize + pad
  5. Momentum-based optimisation (MI-FGSM)
  6. Low L∞ epsilon budget for minimal visibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random
import math


# ---------------------------------------------------------------------------
# Gaussian kernel for gradient smoothing
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 5, sigma: float = 1.0):
    """Create a 2D Gaussian kernel for convolution."""
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = torch.outer(g, g)
    g = g / g.sum()
    return g


def smooth_gradient(grad: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0):
    """Apply Gaussian blur to the gradient to eliminate speckle noise."""
    kernel = _gaussian_kernel(kernel_size, sigma).to(grad.device)
    # Shape: (1, 1, k, k) → apply per-channel via groups
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    pad = kernel_size // 2
    return F.conv2d(grad, kernel, padding=pad, groups=3)


# ---------------------------------------------------------------------------
# Total Variation loss — penalises high-frequency noise
# ---------------------------------------------------------------------------

def total_variation_loss(delta: torch.Tensor):
    """
    Sum of absolute differences between adjacent pixels.
    Minimising this makes the perturbation spatially smooth.
    """
    tv_h = torch.abs(delta[:, :, 1:, :] - delta[:, :, :-1, :]).mean()
    tv_w = torch.abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]).mean()
    return tv_h + tv_w


def color_regularisation_loss(delta: torch.Tensor):
    """
    Penalise the variance of perturbation across RGB channels.
    This pushes the perturbation toward being channel-uniform (grayscale-like),
    preventing rainbow / oil-spill artifacts.
    """
    channel_mean = delta.mean(dim=1, keepdim=True)  # average across RGB
    return ((delta - channel_mean) ** 2).mean()


def correlate_channels(delta: torch.Tensor, mix: float = 0.6):
    """
    Blend each channel's perturbation toward the channel-average.
    mix=0 → fully independent channels (rainbow)
    mix=1 → identical perturbation in all channels (grayscale)
    """
    channel_mean = delta.mean(dim=1, keepdim=True)
    return (1 - mix) * delta + mix * channel_mean


# ---------------------------------------------------------------------------
# Ensemble feature extractor
# ---------------------------------------------------------------------------

class EnsembleFeatureExtractor(nn.Module):
    """
    Wraps ResNet-50, VGG-16 and DenseNet-121 as feature extractors.
    The average cosine similarity across all three is the attack objective.
    """

    def __init__(self, device="cpu", models_to_use=None):
        super().__init__()
        self.device = device
        # models_to_use: list of model names to include, e.g. ["resnet", "vgg", "densenet"]
        # Default: all three
        if models_to_use is None:
            models_to_use = ["resnet", "vgg", "densenet"]
        self.active_models = models_to_use

        if "resnet" in models_to_use:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])

        if "vgg" in models_to_use:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.vgg_features = vgg.features

        if "densenet" in models_to_use:
            densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.densenet_features = densenet.features

        self.to(device)
        self.eval()
        if device == "cuda":
            self.half()  # FP16 for faster inference
        for p in self.parameters():
            p.requires_grad = False

    def _extract(self, x):
        if next(self.parameters()).dtype == torch.float16:
            x = x.half()
        
        features = []

        if "resnet" in self.active_models:
            f = self.resnet_features(x)
            features.append(f.flatten(1))

        if "vgg" in self.active_models:
            f = self.vgg_features(x)
            f = F.adaptive_avg_pool2d(f, 1)
            features.append(f.flatten(1))

        if "densenet" in self.active_models:
            f = self.densenet_features(x)
            f = F.relu(f)
            f = F.adaptive_avg_pool2d(f, 1)
            features.append(f.flatten(1))

        return [f.float() for f in features]

    def ensemble_cosine_similarity(self, x_orig, x_adv):
        feats_orig = self._extract(x_orig)
        feats_adv = self._extract(x_adv)
        sims = []
        for fo, fa in zip(feats_orig, feats_adv):
            sims.append(F.cosine_similarity(fo, fa, dim=1).mean())
        return torch.stack(sims).mean()
        
    def ensemble_cosine_similarity_cached(self, feats_orig, x_adv):
        """Compare pre-computed original features with adversarial features.
        This provides a ~40% speedup by skipping 3 forward passes per iteration.
        """
        feats_adv = self._extract(x_adv)
        sims = []
        for fo, fa in zip(feats_orig, feats_adv):
            sims.append(F.cosine_similarity(fo, fa, dim=1).mean())
        return torch.stack(sims).mean()


# ---------------------------------------------------------------------------
# Input diversity (Xie et al., 2019)
# ---------------------------------------------------------------------------

def input_diversity(x, resize_range=(200, 248), target_size=224, prob=0.5):
    """Random resize + pad to prevent overfitting to one resolution."""
    if random.random() > prob:
        return x
    new_size = random.randint(resize_range[0], resize_range[1])
    x_resized = F.interpolate(x, size=(new_size, new_size),
                              mode="bilinear", align_corners=False)
    pad_total = target_size - new_size
    pad_left = random.randint(0, max(pad_total, 0))
    pad_top = random.randint(0, max(pad_total, 0))
    pad_right = max(pad_total - pad_left, 0)
    pad_bottom = max(pad_total - pad_top, 0)
    x_padded = F.pad(x_resized, (pad_left, pad_right, pad_top, pad_bottom))
    return x_padded[:, :, :target_size, :target_size]


# ---------------------------------------------------------------------------
# Main attack
# ---------------------------------------------------------------------------

def privacy_attack(
    ensemble: EnsembleFeatureExtractor,
    image_tensor: torch.Tensor,
    epsilon: float = 10 / 255,
    alpha: float = 0.5 / 255,
    num_iterations: int = 300,
    momentum: float = 0.9,
    tv_weight: float = 0.3,
    smooth_sigma: float = 1.5,
    smooth_kernel: int = 7,
    use_input_diversity: bool = True,
    use_color_reg: bool = True,
    progress_callback=None,
):
    """
    Smooth ensemble adversarial attack.

    Key differences from vanilla MI-FGSM:
      • Gaussian blur on gradients → removes per-pixel speckle noise
      • TV regularisation → penalises checkerboard/rainbow patterns
      • Smaller alpha & epsilon → finer-grained, less visible perturbation
      • More iterations to compensate for the smaller step size

    Args:
        ensemble:           EnsembleFeatureExtractor instance.
        image_tensor:       (1, C, 224, 224) in [0, 1].
        epsilon:            Max L∞ perturbation (default 10/255 ≈ 0.039).
        alpha:              Step size per iteration.
        num_iterations:     PGD iterations.
        momentum:           MI-FGSM momentum.
        tv_weight:          Weight for total variation regularisation.
        smooth_sigma:       Sigma for Gaussian gradient smoothing.
        smooth_kernel:      Kernel size for Gaussian gradient smoothing.
        use_input_diversity: Apply random resize/pad transforms.
        progress_callback:  callable(current, total, info_dict).
    """

    device = image_tensor.device
    x = image_tensor.clone().detach()

    # Initialise delta near zero (very small random) for smooth convergence
    delta = torch.zeros_like(x).uniform_(-epsilon * 0.01, epsilon * 0.01)
    delta = torch.clamp(x + delta, 0, 1) - x
    delta.requires_grad_(True)

    grad_momentum = torch.zeros_like(x)
    
    # Pre-compute original features ONCE before the loop
    with torch.no_grad():
        feats_orig = ensemble._extract(x)
        # Detach so they don't participate in the graph
        feats_orig = [f.detach() for f in feats_orig]
        
    # Pre-build Gaussian kernel once
    kernel = _gaussian_kernel(smooth_kernel, smooth_sigma).to(device)
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)
    pad = smooth_kernel // 2

    def fast_smooth(grad):
        return F.conv2d(grad, kernel, padding=pad, groups=3)

    for i in range(num_iterations):
        adv = x + delta

        if use_input_diversity:
            adv_t = input_diversity(adv)
        else:
            adv_t = adv

        # Feature disruption loss using cached features
        cos_sim = ensemble.ensemble_cosine_similarity_cached(feats_orig, adv_t)

        # TV regularisation on the perturbation
        tv = total_variation_loss(delta)

        # Colour regularisation — prevent independent per-channel perturbation
        if use_color_reg:
            color_reg = color_regularisation_loss(delta)
            loss = cos_sim + tv_weight * tv + tv_weight * 0.5 * color_reg
        else:
            loss = cos_sim + tv_weight * tv

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.data

            # ---- Smooth the gradient using pre-built kernel ----
            grad = fast_smooth(grad)

            # Normalise
            grad = grad / (grad.abs().mean() + 1e-10)

            # Momentum accumulation
            grad_momentum = momentum * grad_momentum + grad

            # PGD step (decrease cosine similarity)
            delta.data = delta.data - alpha * grad_momentum.sign()

            # Correlate channels to prevent rainbow artifacts
            if use_color_reg:
                delta.data = correlate_channels(delta.data, mix=0.6)

            # Project onto L∞ ε-ball and [0, 1]
            delta.data = torch.clamp(delta.data, -epsilon, epsilon)
            delta.data = torch.clamp(x + delta.data, 0, 1) - x

            # Periodically smooth the delta itself to prevent pattern buildup
            if (i + 1) % 20 == 0:
                # Need a temporary 0.3 sigma kernel for delta smoothing
                delta.data = smooth_gradient(delta.data, kernel_size=smooth_kernel,
                                             sigma=smooth_sigma * 0.3)
                if use_color_reg:
                    delta.data = correlate_channels(delta.data, mix=0.6)
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(x + delta.data, 0, 1) - x

        delta.grad.zero_()

        # Report progress every 5 iterations
        if progress_callback and (i + 1) % 5 == 0:
            current_sim = cos_sim.item()
            progress_callback(i + 1, num_iterations, {
                "iteration": i + 1,
                "cos_sim": round(current_sim, 4),
                "l2": round(torch.norm(delta.data).item(), 4),
            })
            
            # Early stopping if resemblance is sufficiently destroyed
            if current_sim < 0.05 and i > 50:
                break

    # --- Final gentle smoothing pass on perturbation ---
    with torch.no_grad():
        final_delta = delta.detach()
        # Correlate channels and smooth
        if use_color_reg:
            final_delta = correlate_channels(final_delta, mix=0.6)
        final_delta = smooth_gradient(final_delta, kernel_size=5,
                                      sigma=smooth_sigma * 0.5)
        final_delta = torch.clamp(final_delta, -epsilon, epsilon)
        best_adv = torch.clamp(x + final_delta, 0, 1)

    with torch.no_grad():
        final_sim = ensemble.ensemble_cosine_similarity(x, best_adv).item()
        l2 = torch.norm(best_adv - x).item()

    stats = {
        "l2_distance": l2,
        "cos_sim_orig": 1.0,
        "cos_sim_adv": final_sim,
    }

    if progress_callback:
        progress_callback(num_iterations, num_iterations, {
            "iteration": num_iterations,
            "cos_sim": round(final_sim, 4),
            "l2": round(l2, 4),
        })

    return best_adv, stats
