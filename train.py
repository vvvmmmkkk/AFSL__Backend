import sys
from pathlib import Path
import torch
torch.backends.cudnn.benchmark = True
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

# add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from datasets.ffpp_dataset import FFPPDataset
from models.detector import Detector
from attacks.pgd import pgd_attack
from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    DEVICE,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS,
)

# Training mode: "baseline" or "afsl"
MODE = "afsl"

# Baseline checkpoint path
BASELINE_CKPT = "runs/baseline_epoch_5.pth"

# AFSL epoch count
AFSL_EPOCHS = 1

# -----------------------------
# SRL LOSS (real–fake pairing)
# Penalises HIGH similarity between real and fake features
# (pushes them apart to preserve inter-class separation)
# L_SRL = max(0, sim(real, fake) - margin)
# -----------------------------
def similarity_regularization_loss(features, labels, video_ids, margin=0.3):
    device = features.device
    loss = torch.tensor(0.0, device=device)
    count = 0

    labels_list = labels.detach().cpu().tolist()

    for vid in set(video_ids):
        idxs = [i for i, v in enumerate(video_ids) if v == vid]
        if len(idxs) < 2:
            continue

        real_idxs = [i for i in idxs if labels_list[i] == 1]
        fake_idxs = [i for i in idxs if labels_list[i] == 0]

        if not real_idxs or not fake_idxs:
            continue

        i, j = real_idxs[0], fake_idxs[0]
        sim = F.cosine_similarity(
            features[i].unsqueeze(0),
            features[j].unsqueeze(0)
        )
        # Penalise when similarity EXCEEDS margin (push apart)
        loss += torch.clamp(sim - margin, min=0.0)
        count += 1

    return loss / count if count > 0 else loss



# -----------------------------
# TRAINING LOOP (BASELINE OR AFSL)
# -----------------------------
def train():
    device = DEVICE

    # dataset & loader
    train_dataset = FFPPDataset(split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # IMPORTANT on Windows
        drop_last=True,
        pin_memory=(device == "cuda"),
    )

    # model
    model = Detector().to(device)

    if MODE == "afsl":
        # Load pretrained baseline checkpoint
        model.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE))
        print(f"Loaded baseline checkpoint from {BASELINE_CKPT}")

    # Set epochs and learning rate for AFSL
    epochs = AFSL_EPOCHS
    lr = LEARNING_RATE * 0.1
    optimizer = Adam(model.parameters(), lr=lr)

    if MODE == "afsl":
        lambda_asl = 0.3
        lambda_srl = 0.01

    print(f"Starting {MODE.upper()} training...\n")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, labels, video_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            if MODE == "afsl":
                # Generate adversarial images for AFSL training
                images_adv = pgd_attack(
                    model,
                    images,
                    labels,
                    epsilon=PGD_EPSILON,
                    alpha=PGD_ALPHA,
                    steps=PGD_STEPS
                )

                # Compute features and logits
                features, logits = model(images)
                features_adv, _ = model(images_adv)

                # 1. Classification loss
                loss_cls = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    labels.float().view(-1)
                )

                # 2. Adversarial Feature Similarity Loss (ASL)
                loss_asl = 1 - F.cosine_similarity(
                    features,
                    features_adv,
                    dim=1
                ).mean()

                # 3. Similarity Regularization Loss (SRL)
                loss_srl = similarity_regularization_loss(
                    features,
                    labels,
                    video_ids
                )

                # total AFSL loss
                loss = loss_cls + lambda_asl * loss_asl + lambda_srl * loss_srl
            else:
                # Baseline: classification loss only on clean images
                _, logits = model(images)
                loss = F.binary_cross_entropy_with_logits(
                    logits.view(-1),
                    labels.float().view(-1)
                )

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}\n")

        # save checkpoint
        torch.save(
            model.state_dict(),
            f"runs/{MODE}_epoch_{epoch+1}.pth"
        )

    print(f"{MODE.upper()} training finished.")


if __name__ == "__main__":
    train()
