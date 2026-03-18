import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from PIL import Image

from config import cfg
from models.fusion_model import FusionModel
from utils.preprocessing import DeepfakeDataset
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from utils.grad_cam import GradCAM, overlay_heatmap_on_image


def get_test_loader() -> DataLoader:
    test_dataset = DeepfakeDataset(cfg.DATA_DIR, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return test_loader


def evaluate(model: FusionModel, loader: DataLoader, device) -> None:
    model.eval()
    all_labels: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for spatial, freq, labels in tqdm(loader, desc="Testing"):
            spatial = spatial.to(device)
            freq = freq.to(device)
            labels = labels.to(device)

            logits, _ = model(spatial, freq)
            probs = torch.sigmoid(logits)

            all_labels.extend(labels.cpu().numpy().astype(int).tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_labels)
    y_probs = np.array(all_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        roc_auc = float("nan")

    print(
        f"Test: Acc={acc:.4f} | Prec={precision:.4f} | "
        f"Rec={recall:.4f} | F1={f1:.4f} | AUC={roc_auc:.4f}"
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(cfg.RESULTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(cfg.RESULTS_DIR, "roc_curve.png")

    plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
    plot_roc_curve(y_true, y_probs, save_path=roc_path)
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved ROC curve to {roc_path}")


def gradcam_demo(model: FusionModel, device, num_samples: int = 4):
    """
    Generate Grad-CAM visualizations for a few test images.
    """
    model.eval()
    test_dataset = DeepfakeDataset(cfg.DATA_DIR, split="test")

    if len(test_dataset) == 0:
        return

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    # target last conv block of the spatial branch
    target_layer = model.spatial_branch.backbone[-1]
    grad_cam = GradCAM(model, target_layer)

    indices = list(range(min(num_samples, len(test_dataset))))
    for idx in indices:
        spatial, freq, label = test_dataset[idx]

        spatial_batch = spatial.unsqueeze(0).to(device)
        freq_batch = freq.unsqueeze(0).to(device)

        heatmap = grad_cam.generate(spatial_batch, freq_batch)

        # convert spatial tensor back to PIL for overlay
        spatial_img = spatial.permute(1, 2, 0).cpu().numpy()
        spatial_img = np.clip(spatial_img * 255.0, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(spatial_img)

        overlay = overlay_heatmap_on_image(heatmap, pil_img, alpha=0.5)
        save_path = os.path.join(cfg.RESULTS_DIR, f"gradcam_sample_{idx}_label_{int(label.item())}.png")
        overlay.save(save_path)
        print(f"Saved Grad-CAM visualization to {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FusionModel(pretrained_backbones=False).to(device)
    if not os.path.isfile(cfg.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model checkpoint not found at {cfg.BEST_MODEL_PATH}. "
            "Train the model first by running train.py."
        )
    state = torch.load(cfg.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state)

    test_loader = get_test_loader()
    evaluate(model, test_loader, device)
    gradcam_demo(model, device, num_samples=4)


if __name__ == "__main__":
    main()

