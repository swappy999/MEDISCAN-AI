"""
inference.py — Standalone ML inference module for MediScan AI.
Imports only from stdlib, torch, torchvision, PIL, cv2, numpy.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class labels
# ---------------------------------------------------------------------------
CLASSES = ["Normal", "Pneumonia"]


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class DenseNet121Pneumonia(nn.Module):
    """DenseNet-121 fine-tuned for binary pneumonia classification."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pretrained:
                self.densenet = models.densenet121(
                    weights=models.DenseNet121_Weights.IMAGENET1K_V1
                )
            else:
                self.densenet = models.densenet121(weights=None)

        # Replace classifier: Dropout(0.5) + Linear(1024, 1)
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x).squeeze(-1)

    def get_last_conv_layer(self) -> nn.Module:
        return self.densenet.features.norm5


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
class PyTorchGradCAM:
    """Gradient-weighted Class Activation Mapping for DenseNet121Pneumonia."""

    def __init__(self, model: DenseNet121Pneumonia, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def _forward_hook(self, module, input, output):  # noqa: A002
        self._activations = output.detach()
        if output.requires_grad:
            def save_grad(grad):
                self._gradients = grad
            output.register_hook(save_grad)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Run a forward+backward pass and return the Grad-CAM heatmap
        as a float32 numpy array with values in [0, 1], shape (224, 224).

        Parameters
        ----------
        tensor : torch.Tensor
            Preprocessed image tensor, shape (1, 3, 224, 224), on the
            correct device, with grad enabled.
        """
        self.model.zero_grad()

        # Forward
        logit = self.model(tensor)  # shape (1,) or scalar
        # Scalar sigmoid output — binary class so backprop w.r.t. logit directly
        logit = logit if logit.dim() > 0 else logit.unsqueeze(0)
        score = logit[0]
        score.backward()

        # activations: (1, C, H, W)
        activations = self._activations  # captured in forward hook
        # gradients: (1, C, H, W)
        gradients = self._gradients

        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Weights: global-average-pool gradients over spatial dims
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=False)  # (1, H, W)
        cam = cam.squeeze(0).cpu().numpy()  # (H, W)

        # ReLU + normalize
        cam = np.maximum(cam, 0)
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Resize to 224×224
        cam = cv2.resize(cam, (224, 224))
        return cam.astype(np.float32)

    def remove_hooks(self):
        """Remove the registered forward hook."""
        self._handle.remove()


# ---------------------------------------------------------------------------
# Overlay utility
# ---------------------------------------------------------------------------
def overlay_cam(
    image_rgb_np: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Blend a Grad-CAM heatmap over an RGB image.

    Parameters
    ----------
    image_rgb_np : np.ndarray
        uint8 RGB image, shape (H, W, 3).
    cam : np.ndarray
        float32 array in [0, 1], shape (H, W).
    alpha : float
        Blending weight for the heatmap (0 = original only, 1 = heatmap only).

    Returns
    -------
    np.ndarray
        uint8 blended image, shape (H, W, 3), RGB.
    """
    cam_uint8 = np.uint8(255 * cam)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
def get_inference_transform() -> transforms.Compose:
    """Return the standard ImageNet-normalised inference transform."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """
    Instantiate and load the DenseNet121Pneumonia model from a checkpoint.

    Returns
    -------
    (DenseNet121Pneumonia, PyTorchGradCAM)
    """
    model = DenseNet121Pneumonia(pretrained=False)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(device)

    gradcam = PyTorchGradCAM(model, model.get_last_conv_layer())
    logger.info("Model loaded from %s on %s", checkpoint_path, device)
    return model, gradcam


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(
    model: DenseNet121Pneumonia,
    gradcam: PyTorchGradCAM,
    device: torch.device,
    image_pil: Image.Image,
    heatmap_save_path: Optional[str] = None,
    overlay_save_path: Optional[str] = None,
) -> dict:
    """
    Run prediction (and optional Grad-CAM) on a PIL RGB image.

    Returns
    -------
    dict with keys: prediction, confidence, probabilities,
                    heatmap_saved, overlay_saved
    """
    transform = get_inference_transform()
    tensor = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)

    # ----------------------------------------------------------------
    # No-grad forward pass for prediction
    # ----------------------------------------------------------------
    with torch.no_grad():
        logit = model(tensor)

    prob: float = torch.sigmoid(logit).item()
    prediction = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    confidence = prob if prob >= 0.5 else 1.0 - prob
    probabilities = {
        "NORMAL": round(1.0 - prob, 4),
        "PNEUMONIA": round(prob, 4),
    }

    heatmap_saved = False
    overlay_saved = False

    # ----------------------------------------------------------------
    # Grad-CAM (requires gradient-enabled forward pass)
    # ----------------------------------------------------------------
    if heatmap_save_path is not None:
        try:
            # Fresh tensor with grad enabled
            grad_tensor = (
                transform(image_pil.convert("RGB")).unsqueeze(0).to(device)
            )
            grad_tensor.requires_grad_(True)

            cam = gradcam.generate(grad_tensor)

            # Save raw heatmap
            heatmap_uint8 = np.uint8(255 * cam)
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(str(heatmap_save_path), heatmap_bgr)
            heatmap_saved = True

            # Save overlay
            if overlay_save_path is not None:
                img_224 = np.array(
                    image_pil.convert("RGB").resize((224, 224), Image.BILINEAR)
                )
                overlay = overlay_cam(img_224, cam, alpha=0.4)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(overlay_save_path), overlay_bgr)
                overlay_saved = True

        except Exception as exc:
            logger.warning("Grad-CAM generation failed: %s", exc, exc_info=True)
            heatmap_saved = False
            overlay_saved = False

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "heatmap_saved": heatmap_saved,
        "overlay_saved": overlay_saved,
    }
