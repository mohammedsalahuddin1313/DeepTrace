import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from config import cfg
from models.fusion_model import FusionModel
from utils.preprocessing import DeepfakeDataset


def get_dataloaders() -> Tuple[DataLoader, DataLoader]:
    train_dataset = DeepfakeDataset(cfg.DATA_DIR, split="train")
    val_dataset = DeepfakeDataset(cfg.DATA_DIR, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_metrics(y_true: List[int], y_probs: List[float]) -> Dict[str, float]:
    y_true_arr = np.array(y_true)
    y_probs_arr = np.array(y_probs)
    y_pred_arr = (y_probs_arr >= 0.5).astype(int)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="binary", zero_division=0
    )
    try:
        roc_auc = roc_auc_score(y_true_arr, y_probs_arr)
    except ValueError:
        roc_auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders()

    model = FusionModel(pretrained_backbones=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS} [Train]")
        for spatial, freq, labels in pbar:
            spatial = spatial.to(device)
            freq = freq.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(spatial, freq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"loss": np.mean(train_losses)})

        model.eval()
        val_losses = []
        all_labels: List[int] = []
        all_probs: List[float] = []

        with torch.no_grad():
            for spatial, freq, labels in tqdm(
                val_loader, desc=f"Epoch {epoch}/{cfg.NUM_EPOCHS} [Val]"
            ):
                spatial = spatial.to(device)
                freq = freq.to(device)
                labels = labels.to(device)

                logits, _ = model(spatial, freq)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

                probs = torch.sigmoid(logits)
                all_labels.extend(labels.cpu().numpy().astype(int).tolist())
                all_probs.extend(probs.cpu().numpy().tolist())

        val_loss = float(np.mean(val_losses))
        metrics = compute_metrics(all_labels, all_probs)

        print(
            f"Epoch {epoch}: "
            f"Train Loss={np.mean(train_losses):.4f} | "
            f"Val Loss={val_loss:.4f} | "
            f"Acc={metrics['accuracy']:.4f} | "
            f"Prec={metrics['precision']:.4f} | "
            f"Rec={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"AUC={metrics['roc_auc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), cfg.BEST_MODEL_PATH)
            print(f"  -> Saved new best model to {cfg.BEST_MODEL_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    train()

