import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# add project root
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from datasets.ffpp_dataset import FFPPDataset
from models.detector import Detector
from attacks.pgd import pgd_attack
from config import (
    BATCH_SIZE,
    DEVICE,
    PGD_EPSILON,
    PGD_ALPHA,
    PGD_STEPS,
    DECISION_THRESHOLD,
)

def evaluate_model(model_path="runs/afsl_epoch_6.pth", adversarial=False):
    # Load model
    model = Detector()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load test dataset
    test_dataset = FFPPDataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    attack_type = "Adversarial" if adversarial else "Clean"
    print(f"Evaluating model: {model_path} ({attack_type})")
    print(f"Test dataset size: {len(test_dataset)}")

    for images, labels, _ in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if adversarial:
            # PGD needs gradients — must be OUTSIDE no_grad()
            images = pgd_attack(
                model,
                images,
                labels,
                epsilon=PGD_EPSILON,
                alpha=PGD_ALPHA,
                steps=PGD_STEPS
            )

        with torch.no_grad():
            _, logits = model(images)
            preds = (torch.sigmoid(logits.squeeze()) > DECISION_THRESHOLD).int()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Fake', 'Real']))

    return accuracy

if __name__ == "__main__":
    # You can change the model path here
    evaluate_model("runs/afsl_epoch_6.pth")