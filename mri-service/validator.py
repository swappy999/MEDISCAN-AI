"""
validator.py — Dedicated Brain MRI Input Validation & Out-of-Distribution (OOD) Gate.
-------------------------------------------------------------------------------------
Ensures that ONLY genuine Brain MRI images reach the Brain MRI tumour classifier.
Rejects chest X-rays, extremity/dental/spine X-rays, CT scans, other body-part
MRIs, ordinary photographs, screenshots, medical reports, blank frames, and
corrupted files.

Fails closed whenever the validator has not been trained, calibrated, and verified.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.ndimage import sobel

logger = logging.getLogger("mediscan.mri.validator")

MSG_ACCEPTED = "Brain MRI Verified — Image accepted for tumour classification."
MSG_NOT_READY = (
    "Brain MRI validator is not trained and verified. Service failed closed for safety."
)
MSG_CORRUPTED = (
    "Invalid Upload — Image file is corrupted, blank, or cannot be decoded. "
    "Please upload a valid Brain MRI image."
)
MSG_PHOTO = (
    "Unsupported Image — Natural photographs or color graphics are not accepted. "
    "Please upload an authentic Brain MRI scan."
)
MSG_DOCUMENT = (
    "Invalid Upload — Document or screenshot detected. Please upload an authentic Brain MRI image."
)
MSG_CHEST_XRAY = (
    "Unsupported Modality — Chest X-rays are not accepted by the Brain MRI module. "
    "Please use the Chest X-Ray analysis module."
)
MSG_NON_BRAIN_MRI = (
    "Unsupported Image — Input does not match a valid Brain MRI scan. "
    "Please upload a Brain MRI image (JPG, JPEG, or PNG)."
)
MSG_UNCERTAIN = (
    "Unable to confirm Brain MRI scan with sufficient confidence. "
    "Please upload another clear Brain MRI image."
)


class BrainMRIValidator:
    """
    Two-tier Brain MRI validator combining:
      1. Structural, resolution, photometric saturation, and text-document screening.
      2. Calibrated deep-feature (MobileNetV3-Small 576-D) OOD & binary gating:
         - Class-conditional cosine similarity to 4 Brain MRI prototypes
         - Class-conditional Mahalanobis distance (Ledoit-Wolf shared precision)
         - Supervised L2 logistic probability P(BRAIN_MRI | embedding)
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base_dir = Path(__file__).resolve().parent
        self.artifacts_path = self.base_dir / "mri_validator_artifacts.npz"
        self.report_path = self.base_dir / "mri_validator_report.json"

        self.is_ready: bool = False
        self.ready_reason: str = MSG_NOT_READY
        self.report_summary: Dict[str, Any] = {}

        self._load_artifacts_and_readiness()
        if self.is_ready:
            self._load_feature_extractor()

    def _load_artifacts_and_readiness(self) -> None:
        if not self.artifacts_path.exists():
            self.ready_reason = f"Validator artifacts missing at {self.artifacts_path}"
            logger.error(self.ready_reason)
            return

        if not self.report_path.exists():
            self.ready_reason = (
                f"Validator verification report missing at {self.report_path}. "
                "Run verify_mri_validator.py before serving."
            )
            logger.error(self.ready_reason)
            return

        try:
            with open(self.report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            if not report.get("validator_ready", False):
                self.ready_reason = (
                    "Validator verification report marked validator_ready=False."
                )
                logger.error(self.ready_reason)
                return
            self.report_summary = report
        except Exception as exc:
            self.ready_reason = f"Failed to read validator verification report: {exc}"
            logger.error(self.ready_reason)
            return

        try:
            data = np.load(self.artifacts_path, allow_pickle=False)
            self.mri_class_names = [str(x) for x in data["mri_class_names"]]
            self.mri_centroids = data["mri_centroids"].astype(np.float64)        # (4, 576)
            self.precision_matrix = data["precision_matrix"].astype(np.float64)  # (576, 576)
            self.cxr_centroid = data["cxr_centroid"].astype(np.float64)          # (576,)
            self.scaler_mean = data["scaler_mean"].astype(np.float64)            # (576,)
            self.scaler_scale = data["scaler_scale"].astype(np.float64)          # (576,)
            self.clf_coef = data["clf_coef"].astype(np.float64)                  # (576,)
            self.clf_intercept = float(data["clf_intercept"])

            self.cos_accept = float(data["cos_accept"])
            self.cos_uncertain = float(data["cos_uncertain"])
            self.maha_accept = float(data["maha_accept"])
            self.maha_uncertain = float(data["maha_uncertain"])
            self.prob_accept = float(data["prob_accept"])
            self.prob_uncertain = float(data["prob_uncertain"])
            self.sat_mean_max = float(data["sat_mean_max"])
            self.white_ratio_max = float(data["white_ratio_max"])

            self.is_ready = True
            self.ready_reason = "Trained, calibrated, and verified Brain MRI validator ready."
            logger.info(
                "Brain MRI validator ready (cos>=%.4f, maha<=%.2f, prob>=%.4f)",
                self.cos_accept,
                self.maha_accept,
                self.prob_accept,
            )
        except Exception as exc:
            self.is_ready = False
            self.ready_reason = f"Failed to load validator artifacts: {exc}"
            logger.error(self.ready_reason, exc_info=True)

    def _load_feature_extractor(self) -> None:
        cached_weights = (
            Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
        )
        if cached_weights.exists():
            state_dict = torch.load(
                cached_weights, map_location=self.device, weights_only=True
            )
            backbone = models.mobilenet_v3_small(weights=None)
            backbone.load_state_dict(state_dict)
            logger.info("Loaded MobileNetV3-Small cache: %s", cached_weights)
        else:
            backbone = models.mobilenet_v3_small(
                weights=models.MobileNet_V3_Small_Weights.DEFAULT
            )
        backbone.classifier = torch.nn.Identity()
        self.backbone = backbone.to(self.device).eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _extract_embedding(self, image: Image.Image) -> np.ndarray:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(tensor).squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(feat) + 1e-12
        return feat / norm

    def _reject(
        self,
        image_type: str,
        reason: str,
        message: str,
        score: float = 0.0,
        status: str = "REJECTED",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "is_valid": False,
            "image_type": image_type,
            "validation_status": status,
            "reason": reason,
            "message": message,
            "score": round(float(score), 4),
        }
        if extra:
            payload.update(extra)
        return payload

    def validate(self, image: Image.Image) -> Dict[str, Any]:
        """Run fail-closed multi-stage validation on a PIL image."""
        if not self.is_ready:
            return self._reject(
                image_type="VALIDATOR_NOT_READY",
                reason=self.ready_reason,
                message=MSG_NOT_READY,
                score=0.0,
                status="NOT_READY",
            )

        # Stage 1: Structural & dimension validation
        try:
            w, h = image.size
        except Exception as exc:
            return self._reject(
                "CORRUPTED_FILE", f"Unable to inspect dimensions: {exc}", MSG_CORRUPTED
            )

        if w < 100 or h < 100:
            return self._reject(
                "LOW_RESOLUTION",
                f"Resolution ({w}x{h}) is below the minimum 100x100 requirement for MRI analysis.",
                MSG_NON_BRAIN_MRI,
            )

        aspect_ratio = w / float(h)
        if aspect_ratio > 2.2 or aspect_ratio < 0.42:
            return self._reject(
                "EXTREMITY_OR_PANORAMIC_XRAY",
                f"Extreme aspect ratio ({aspect_ratio:.2f}) is incompatible with diagnostic brain MRI slices.",
                MSG_NON_BRAIN_MRI,
                score=0.1,
            )

        # Stage 2: Photometric / saturation & blank checks on RGB array
        rgb_small = np.asarray(
            image.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32
        ) / 255.0
        gray = (
            0.299 * rgb_small[:, :, 0]
            + 0.587 * rgb_small[:, :, 1]
            + 0.114 * rgb_small[:, :, 2]
        )
        if float(np.std(gray)) < 0.02:
            return self._reject(
                "CORRUPTED_OR_BLANK",
                "Image contains near-zero pixel variance (blank or flat frame).",
                MSG_CORRUPTED,
            )

        sat = rgb_to_hsv(rgb_small)[:, :, 1]
        sat_mean = float(np.mean(sat))
        if sat_mean > self.sat_mean_max:
            return self._reject(
                "ORDINARY_PHOTOGRAPH",
                f"Mean color saturation ({sat_mean:.3f}) exceeds medical MRI calibration bound ({self.sat_mean_max:.3f}).",
                MSG_PHOTO,
                score=0.01,
            )

        # Stage 3: Document / medical report / light-mode screenshot screening
        white_ratio = float(np.mean(gray >= 240.0 / 255.0))
        if white_ratio > self.white_ratio_max:
            gx = np.abs(sobel(gray, axis=1))
            edge_density = float(np.mean(gx > 0.12))
            if white_ratio > 0.60 or edge_density > 0.04:
                return self._reject(
                    "DOCUMENT_OR_REPORT",
                    f"Image matches light-background document/screenshot layout (white area {white_ratio:.1%}, edge density {edge_density:.3f}).",
                    MSG_DOCUMENT,
                    score=0.02,
                )

        # Stage 4: Deep OOD & trained supervised Brain-MRI gate
        z = self._extract_embedding(image)
        cos_sims = np.dot(self.mri_centroids, z)
        best_proto_idx = int(np.argmax(cos_sims))
        max_cos_sim = float(cos_sims[best_proto_idx])

        diffs = self.mri_centroids - z[None, :]  # (4, 576)
        # Mahalanobis squared distances via quadratic form
        maha_dists = np.sqrt(
            np.maximum(0.0, np.einsum("ki,ij,kj->k", diffs, self.precision_matrix, diffs))
        )
        min_maha = float(np.min(maha_dists))

        z_scaled = (z - self.scaler_mean) / np.where(
            self.scaler_scale > 1e-8, self.scaler_scale, 1.0
        )
        logit = float(np.dot(self.clf_coef, z_scaled) + self.clf_intercept)
        mri_prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -40.0, 40.0))))

        cxr_cos_sim = float(np.dot(self.cxr_centroid, z))

        metrics = {
            "cosine_similarity": round(max_cos_sim, 4),
            "mahalanobis_distance": round(min_maha, 2),
            "mri_classifier_prob": round(mri_prob, 4),
            "cxr_similarity": round(cxr_cos_sim, 4),
            "nearest_prototype": self.mri_class_names[best_proto_idx],
            "thresholds": {
                "cos_accept": round(self.cos_accept, 4),
                "maha_accept": round(self.maha_accept, 2),
                "prob_accept": round(self.prob_accept, 4),
            },
        }

        logger.info(
            "MRI Validation metrics: cos=%.4f (min %.4f), maha=%.2f (max %.2f), prob=%.4f (min %.4f), cxr_sim=%.4f",
            max_cos_sim,
            self.cos_accept,
            min_maha,
            self.maha_accept,
            mri_prob,
            self.prob_accept,
            cxr_cos_sim,
        )

        accepted = (
            max_cos_sim >= self.cos_accept
            and min_maha <= self.maha_accept
            and mri_prob >= self.prob_accept
        )
        if accepted:
            return {
                "is_valid": True,
                "image_type": "BRAIN_MRI",
                "validation_status": "ACCEPTED",
                "reason": (
                    f"Input conforms to calibrated Brain MRI embedding distribution "
                    f"(cos={max_cos_sim:.3f}, maha={min_maha:.1f}, p={mri_prob:.3f})."
                ),
                "message": MSG_ACCEPTED,
                "score": round(max_cos_sim, 4),
                "distance": round(min_maha, 2),
                "metrics": metrics,
            }

        definitely_ood = (
            max_cos_sim < self.cos_uncertain
            or min_maha > self.maha_uncertain
            or mri_prob < self.prob_uncertain
        )
        if definitely_ood:
            if cxr_cos_sim >= 0.72 and cxr_cos_sim > max_cos_sim + 0.05:
                image_type = "CHEST_XRAY"
                message = MSG_CHEST_XRAY
                reason = (
                    f"Feature embedding aligns with thoracic Chest X-ray (cxr_sim={cxr_cos_sim:.3f}) "
                    f"rather than Brain MRI (mri_cos={max_cos_sim:.3f})."
                )
            else:
                image_type = "NON_BRAIN_MRI"
                message = MSG_NON_BRAIN_MRI
                reason = (
                    f"Out-of-distribution input rejected (mri_cos={max_cos_sim:.3f}, "
                    f"mahalanobis={min_maha:.1f}, clf_prob={mri_prob:.3f})."
                )
            return self._reject(
                image_type=image_type,
                reason=reason,
                message=message,
                score=max_cos_sim,
                status="REJECTED",
                extra={"distance": round(min_maha, 2), "metrics": metrics},
            )

        return self._reject(
            image_type="UNCERTAIN",
            reason=(
                f"Input lies in uncertain boundary zone (mri_cos={max_cos_sim:.3f}, "
                f"mahalanobis={min_maha:.1f}, clf_prob={mri_prob:.3f})."
            ),
            message=MSG_UNCERTAIN,
            score=max_cos_sim,
            status="UNCERTAIN",
            extra={"distance": round(min_maha, 2), "metrics": metrics},
        )


_validator_instance: Optional[BrainMRIValidator] = None


def get_mri_validator(device: Optional[torch.device] = None) -> BrainMRIValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = BrainMRIValidator(device=device)
    return _validator_instance


def validate_brain_mri(
    image: Image.Image, device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Validate whether an uploaded image is a genuine Brain MRI before classification."""
    return get_mri_validator(device=device).validate(image)

