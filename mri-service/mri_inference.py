"""
mri_inference.py — Isolated Brain MRI ResNet-18 Inference & Grad-CAM Module.
Imports model/checkpoint/Grad-CAM architecture from Brain-MRI-Tumour/src
without sharing any weights, transforms, or classes with the Chest X-ray module.
"""

import logging
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure Brain-MRI-Tumour/src is importable without altering upstream files
_BRAIN_SRC = Path(__file__).resolve().parent.parent / "Brain-MRI-Tumour" / "src"
if str(_BRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_BRAIN_SRC))

from data_loader import build_transforms  # noqa: E402
from gradcam import GradCAM  # noqa: E402
from model import get_model  # noqa: E402
from utils import load_checkpoint  # noqa: E402

logger = logging.getLogger("mediscan.mri.inference")

# Canonicalize checkpoint folder names cleanly while preserving model class order
_CLASS_DISPLAY_MAP = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "notumor": "no_tumor",
    "no_tumor": "no_tumor",
}

_GRADCAM_LOCK = threading.Lock()


def normalize_mri_class(raw_name: str) -> str:
    """Map raw dataset folder label (e.g. 'notumor') to canonical API label ('no_tumor')."""
    key = str(raw_name).strip().lower()
    return _CLASS_DISPLAY_MAP.get(key, key)


def load_mri_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[torch.nn.Module, GradCAM, list, str]:
    """
    Load the trained Brain MRI TumorClassifier and attach Grad-CAM to its final conv layer.
    Returns (model, gradcam, class_names, model_name).
    """
    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    raw_class_names = list(ckpt["class_names"])
    backbone_name = str(ckpt.get("model_name", "resnet18"))

    model = get_model(num_classes=len(raw_class_names), backbone=backbone_name).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    gradcam = GradCAM(model, model.last_conv_layer())
    model_identifier = f"ResNet18BrainMRI ({backbone_name})" if backbone_name == "resnet18" else f"BrainMRI ({backbone_name})"
    logger.info(
        "Loaded Brain MRI model '%s' from %s on %s (classes=%s)",
        model_identifier,
        checkpoint_path,
        device,
        raw_class_names,
    )
    return model, gradcam, raw_class_names, model_identifier


def _render_jet_and_overlay(image_pil: Image.Image, cam: np.ndarray, alpha: float = 0.4):
    """Render clinical jet heatmap and blended overlay PIL RGB images matching UI style."""
    jet = matplotlib.colormaps["jet"]
    heatmap_rgb = (jet(np.clip(cam, 0.0, 1.0))[:, :, :3] * 255.0).astype(np.uint8)

    orig_224 = np.asarray(
        image_pil.convert("RGB").resize((224, 224), Image.BILINEAR),
        dtype=np.float32,
    )
    overlay_np = np.clip((1.0 - alpha) * orig_224 + alpha * heatmap_rgb.astype(np.float32), 0, 255).astype(np.uint8)

    return Image.fromarray(heatmap_rgb), Image.fromarray(overlay_np)


def run_mri_inference(
    model: torch.nn.Module,
    gradcam: GradCAM,
    device: torch.device,
    raw_class_names: list,
    image_pil: Image.Image,
    heatmap_save_path: Optional[str] = None,
    overlay_save_path: Optional[str] = None,
) -> Dict:
    """Run 4-class Brain MRI inference and Grad-CAM visual explanation."""
    _, eval_transform = build_transforms(augment=False)
    rgb_image = image_pil.convert("RGB")
    tensor = eval_transform(rgb_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs_tensor = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs_tensor).item())
    predicted_label = normalize_mri_class(raw_class_names[pred_idx])
    confidence = float(probs_tensor[pred_idx].item())

    probabilities: Dict[str, float] = {}
    for idx, raw_name in enumerate(raw_class_names):
        canonical = normalize_mri_class(raw_name)
        probabilities[canonical] = round(float(probs_tensor[idx].item()), 4)

    # Order probabilities consistently: glioma, meningioma, pituitary, no_tumor
    ordered_probs: Dict[str, float] = {}
    for key in ("glioma", "meningioma", "pituitary", "no_tumor"):
        if key in probabilities:
            ordered_probs[key] = probabilities[key]
    for k, v in probabilities.items():
        if k not in ordered_probs:
            ordered_probs[k] = v

    heatmap_saved = False
    overlay_saved = False

    if heatmap_save_path is not None:
        try:
            with _GRADCAM_LOCK:
                grad_tensor = eval_transform(rgb_image).unsqueeze(0).to(device)
                grad_tensor.requires_grad_(True)
                cam, _ = gradcam(grad_tensor, class_idx=pred_idx)
                model.zero_grad(set_to_none=True)

            heatmap_img, overlay_img = _render_jet_and_overlay(rgb_image, cam, alpha=0.4)
            heatmap_img.save(str(heatmap_save_path), format="PNG")
            heatmap_saved = True

            if overlay_save_path is not None:
                overlay_img.save(str(overlay_save_path), format="PNG")
                overlay_saved = True
        except Exception as exc:
            logger.warning("MRI Grad-CAM generation failed: %s", exc, exc_info=True)
            heatmap_saved = False
            overlay_saved = False

    return {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "probabilities": ordered_probs,
        "heatmap_saved": heatmap_saved,
        "overlay_saved": overlay_saved,
    }

