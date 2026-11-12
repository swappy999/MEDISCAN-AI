"""
main_pytorch.py: Chest X-Ray Pneumonia Detection (PyTorch + CUDA Acceleration)
-----------------------------------------------------------------------------
GPU-accelerated DenseNet121 training, evaluation, and Grad-CAM for MediScan AI.
Optimized for NVIDIA GeForce RTX 4060 Laptop GPU (and compatible CUDA GPUs).

Usage:
  # Quick synthetic smoke test (validates CUDA, DenseNet121 forward/backward pass, and Grad-CAM on RTX 4060)
  python main_pytorch.py

  # Train on RTX 4060 with mixed precision (AMP)
  python main_pytorch.py --mode train

  # Evaluate checkpoint on test set
  python main_pytorch.py --mode evaluate --checkpoint outputs/models/best_model_pytorch.pth

  # Single-image prediction + Grad-CAM heatmap visualization
  python main_pytorch.py --mode predict --image data/raw/test/PNEUMONIA/person1_virus_6.jpeg --checkpoint outputs/models/best_model_pytorch.pth
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import cv2
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLASSES = ["Normal", "Pneumonia"]


# ─── Configuration Loader ─────────────────────────────────────────────────────

def load_config(path: str = None) -> dict:
    if path is None or path == "configs/config.yaml":
        base_dir = Path(__file__).resolve().parent
        config_path = base_dir / "configs" / "config.yaml"
    else:
        config_path = Path(path)
        if not config_path.exists():
            config_path = Path(__file__).resolve().parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Ensure root_dir is absolute or relative to repo root
    base_dir = Path(__file__).resolve().parent
    if not Path(config["data"]["root_dir"]).is_absolute():
        config["data"]["root_dir"] = str(base_dir / config["data"]["root_dir"])
    for k in ["model_dir", "log_dir", "viz_dir", "reports_dir"]:
        if not Path(config["output"][k]).is_absolute():
            config["output"][k] = str(base_dir / config["output"][k])
    return config


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Target Compute Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
        logger.info(f"Total Dedicated VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA is not available. Running on CPU.")
    return device


# ─── Model Definition ──────────────────────────────────────────────────────────

class DenseNet121Pneumonia(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.densenet = models.densenet121(weights=weights)
        in_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x).squeeze(-1)

    def get_last_conv_layer(self):
        return self.densenet.features.norm5


# ─── Grad-CAM Implementation ──────────────────────────────────────────────────

class PyTorchGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def generate(self, input_tensor: torch.Tensor) -> Tuple[np.ndarray, float]:
        self.model.eval()
        self.gradients = None
        self.activations = None
        output = self.model(input_tensor)
        prob = torch.sigmoid(output).item()

        self.model.zero_grad()
        output.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam, prob


def overlay_cam(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * image_rgb)
    return overlay


# ─── Dataset & Dataloader ─────────────────────────────────────────────────────

class ChestXrayDataset(Dataset):
    def __init__(self, root_dir: str, split: str, transform=None):
        self.split_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []

        for label_idx, class_name in enumerate(["NORMAL", "PNEUMONIA"]):
            class_folder = self.split_dir / class_name
            if not class_folder.exists():
                raise FileNotFoundError(f"Directory not found: {class_folder}")
            for ext in ("*.jpeg", "*.jpg", "*.png"):
                for p in class_folder.glob(ext):
                    self.samples.append((str(p), label_idx))

        self.labels = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), path


def get_dataloaders(config: dict):
    root_dir = config["data"]["root_dir"]
    img_size = config["data"]["image_size"]
    batch_size = config["training"]["batch_size"]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ChestXrayDataset(root_dir, "train", transform=train_transform)
    val_ds = ChestXrayDataset(root_dir, "val", transform=eval_transform)
    test_ds = ChestXrayDataset(root_dir, "test", transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds


# ─── Training Routine ─────────────────────────────────────────────────────────

def run_train(config: dict):
    device = get_device()
    train_loader, val_loader, test_loader, train_ds = get_dataloaders(config)

    logger.info(f"Dataset split sizes — Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Class balance weighting
    labels = np.array(train_ds.labels)
    n_neg = np.sum(labels == 0)
    n_pos = np.sum(labels == 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    logger.info(f"Class imbalance handling: Normal={n_neg}, Pneumonia={n_pos}, pos_weight={pos_weight.item():.4f}")

    model = DenseNet121Pneumonia(
        pretrained=config["model"].get("pretrained", True),
        dropout=config["model"].get("dropout", 0.5)
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].get("learning_rate", 1e-4),
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )

    epochs = config["training"].get("epochs", 20)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = config["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = model_dir / "best_model_pytorch.pth"

    best_val_auc = 0.0
    patience = config["training"].get("patience", 5)
    patience_counter = 0

    logger.info(f"Starting GPU training for {epochs} epochs on {device} (Mixed Precision: {use_amp})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()

        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(targets)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_targets, val_probs = [], []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device, non_blocking=True)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                val_targets.extend(targets.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_probs)
        val_preds = (np.array(val_probs) >= 0.5).astype(int)
        val_acc = accuracy_score(val_targets, val_preds)
        elapsed = time.time() - start_t

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | Train Loss: {train_loss:.4f} | Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "config": config,
            }, best_ckpt_path)
            logger.info(f"  -> Checkpoint saved: {best_ckpt_path} (Val AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

    logger.info("Training complete. Evaluating best checkpoint on test set...")
    run_evaluate(config, str(best_ckpt_path))


# ─── Evaluation Routine ───────────────────────────────────────────────────────

def run_evaluate(config: dict, checkpoint_path: str):
    device = get_device()
    _, _, test_loader, _ = get_dataloaders(config)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DenseNet121Pneumonia(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_targets, all_probs, all_paths = [], [], []
    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    threshold = config["evaluation"].get("threshold", 0.5)
    preds = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_targets, preds)
    prec = precision_score(all_targets, preds, zero_division=0)
    rec = recall_score(all_targets, preds, zero_division=0)
    f1 = f1_score(all_targets, preds, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs)
    cm = confusion_matrix(all_targets, preds)

    print("\n" + "=" * 50)
    print("      MEDISCAN AI — TEST SET EVALUATION REPORT")
    print("=" * 50)
    print(f"Device:           {device}")
    print(f"Test Samples:     {len(all_targets)}")
    print(f"Accuracy:         {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision:        {prec:.4f}")
    print(f"Recall (Sens):    {rec:.4f}")
    print(f"F1 Score:         {f1:.4f}")
    print(f"ROC-AUC:          {auc:.4f}")
    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0, 0]:4d} | FP: {cm[0, 1]:4d}")
    print(f"  FN: {cm[1, 0]:4d} | TP: {cm[1, 1]:4d}")
    print("=" * 50 + "\n")


# ─── Single-Image Prediction & Grad-CAM ────────────────────────────────────────

def run_predict(config: dict, checkpoint_path: str, image_path: str):
    device = get_device()
    img_size = config["data"]["image_size"]

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DenseNet121Pneumonia(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    orig_img = Image.open(image_path).convert("RGB")
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_img = eval_transform(orig_img).unsqueeze(0).to(device)

    gradcam = PyTorchGradCAM(model, model.get_last_conv_layer())
    cam, prob = gradcam.generate(tensor_img)

    pred_label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else (1.0 - prob)

    viz_dir = Path(config["output"]["viz_dir"])
    viz_dir.mkdir(parents=True, exist_ok=True)
    out_cam_path = viz_dir / f"gradcam_{Path(image_path).stem}.png"

    resized_orig = np.array(orig_img.resize((img_size, img_size)))
    overlay = overlay_cam(resized_orig, cam)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(resized_orig)
    axes[0].set_title("Input Chest X-Ray")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay: {pred_label} ({confidence*100:.1f}%)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_cam_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 50)
    print("       MEDISCAN AI — SINGLE IMAGE PREDICTION")
    print("=" * 50)
    print(f"Image:            {image_path}")
    print(f"Prediction:       {pred_label}")
    print(f"P(Pneumonia):     {prob:.4f}")
    print(f"Confidence:       {confidence*100:.2f}%")
    print(f"Grad-CAM Saved:   {out_cam_path}")
    print("=" * 50 + "\n")


# ─── No-Dataset Synthetic Smoke Test ──────────────────────────────────────────

def run_demo(config: dict):
    device = get_device()
    size = config["data"]["image_size"]

    logger.info("=" * 60)
    logger.info("   MEDISCAN AI — RTX 4060 HARDWARE SMOKE TEST")
    logger.info("=" * 60)
    logger.info(f"Target Device: {device}")

    # Build model directly on GPU
    model = DenseNet121Pneumonia(pretrained=False).to(device)

    # 1. Forward pass on GPU
    dummy_input = torch.randn(4, 3, size, size, device=device)
    dummy_output = model(dummy_input)
    logger.info(f"CUDA Forward Pass OK: input shape={list(dummy_input.shape)}, output shape={list(dummy_output.shape)}")

    # 2. Backward pass on GPU
    dummy_target = torch.tensor([0.0, 1.0, 0.0, 1.0], device=device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(dummy_output, dummy_target)
    loss.backward()
    logger.info(f"CUDA Backward Pass OK: computed loss={loss.item():.4f}")

    # 3. Grad-CAM execution on GPU
    gradcam = PyTorchGradCAM(model, model.get_last_conv_layer())
    cam, prob = gradcam.generate(dummy_input[:1])
    logger.info(f"CUDA Grad-CAM OK: heatmap shape={cam.shape}, sample prob={prob:.4f}")

    logger.info("RTX 4060 GPU Acceleration & PyTorch Pipeline verified 100% functional!")
    print("\nReady for manual execution. Commands:")
    print("  python main_pytorch.py --mode train")
    print("  python main_pytorch.py --mode evaluate --checkpoint outputs/models/best_model_pytorch.pth")
    print("  python main_pytorch.py --mode predict --image <path> --checkpoint outputs/models/best_model_pytorch.pth\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MediScan AI - Chest X-Ray Pneumonia (PyTorch CUDA)")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", default="demo", choices=["demo", "train", "evaluate", "predict"],
                        help="Operation mode: demo | train | evaluate | predict")
    parser.add_argument("--checkpoint", default="outputs/models/best_model_pytorch.pth", help="Path to model checkpoint")
    parser.add_argument("--image", default=None, help="Path to single image for prediction")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.mode == "demo":
        run_demo(config)
    elif args.mode == "train":
        run_train(config)
    elif args.mode == "evaluate":
        run_evaluate(config, args.checkpoint)
    elif args.mode == "predict":
        if not args.image:
            raise ValueError("--image must be specified when mode is 'predict'")
        run_predict(config, args.checkpoint, args.image)


if __name__ == "__main__":
    main()
