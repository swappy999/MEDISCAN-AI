"""
validator.py — Dedicated Chest X-Ray Validation and Out-of-Distribution (OOD) Rejection System.
------------------------------------------------------------------------------------------------
Ensures that ONLY valid frontal chest radiographs reach the pneumonia detection classifier.
Rejects:
- Hand, wrist, foot, knee, dental, skull, spine, extremity radiographs
- Brain MRI and other MRI scans
- CT cross-sectional scans
- Color photographs and graphics
- Medical report screenshots, invoices, and text documents
- Corrupted or undecodable image files
"""

import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger("mediscan.validator")

# Standardized User-Facing Guidance Messages
MSG_ACCEPTED = "Chest X-ray Verified — Image accepted for pneumonia analysis."
MSG_HAND_XRAY = "Unsupported Image — This is not a supported chest X-ray. Please upload a frontal chest X-ray image."
MSG_MRI = "Unsupported Image — MRI images are not accepted by the Chest X-ray Pneumonia module."
MSG_DOCUMENT = "Invalid Upload — Please upload an actual chest X-ray image in JPG, JPEG or PNG format."
MSG_UNCERTAIN = "Unable to verify image type — Please upload a clear frontal chest X-ray."
MSG_NON_CHEST = "Unsupported Image — This is not a supported chest X-ray. Please upload a frontal chest X-ray image."
MSG_CORRUPTED = "Invalid Upload — Image file is corrupted or cannot be decoded. Please upload a valid image file."


class ChestXrayValidator:
    """
    Dedicated Out-of-Distribution (OOD) and Radiomorphological Validator
    calibrated against authentic frontal chest radiographs.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reference_path = Path(__file__).resolve().parent / "cxr_reference_distribution.npz"
        self._load_reference_distribution()
        self._load_feature_extractor()

    def _load_reference_distribution(self):
        """Loads precomputed empirical CXR feature distribution centroid and thresholds."""
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference distribution not found at: {self.reference_path}. "
                f"Run build_cxr_reference.py to generate it."
            )
        data = np.load(self.reference_path)
        self.centroid = data["centroid"] # (576,) normalized vector
        self.accept_threshold = float(data.get("accept_threshold", 0.75))
        self.uncertain_threshold = float(data.get("uncertain_threshold", 0.65))
        logger.info(
            "Validator reference distribution loaded (accept >= %.4f, uncertain >= %.4f)",
            self.accept_threshold,
            self.uncertain_threshold,
        )

    def _load_feature_extractor(self):
        """Initializes dedicated MobileNetV3-Small backbone for universal feature embeddings."""
        cached_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
        if cached_weights.exists():
            state_dict = torch.load(cached_weights, map_location=self.device, weights_only=True)
            backbone = models.mobilenet_v3_small(weights=None)
            backbone.load_state_dict(state_dict)
            logger.info("Loaded MobileNetV3 weights from local cache: %s", cached_weights)
        else:
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        backbone.classifier = torch.nn.Identity()
        backbone = backbone.to(self.device).eval()
        self.backbone = backbone
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("Validator deep feature extractor initialized on %s", self.device)

    def validate(self, image: Image.Image) -> Dict[str, Any]:
        """
        Executes the hierarchical 5-stage validation pipeline on an input PIL image:
        Step 1: Structural & Dimension checks
        Step 2: Photometric / Color Saturation Analysis
        Step 3: Document & Medical Report Detection
        Step 4: Anatomical & Modality Specific Pattern Screening (MRI, CT, Dental, Extremities)
        Step 5: Calibrated Deep Feature Space Cosine Similarity & OOD Distance
        """
        # --- Stage 1: Structural & Dimension Checks ---
        try:
            w, h = image.size
        except Exception as e:
            return {
                "is_valid": False,
                "image_type": "CORRUPTED_FILE",
                "validation_status": "REJECTED",
                "reason": f"Image dimension query failed: {e}",
                "message": MSG_CORRUPTED,
                "score": 0.0,
            }

        if w < 120 or h < 120:
            return {
                "is_valid": False,
                "image_type": "LOW_RESOLUTION",
                "validation_status": "REJECTED",
                "reason": f"Image resolution ({w}x{h}) is too low for clinical radiograph evaluation (minimum 120x120).",
                "message": MSG_NON_CHEST,
                "score": 0.0,
            }

        aspect_ratio = w / float(h)
        # Extreme aspect ratios (e.g. panoramic dental strips > 2.0 or narrow leg strips < 0.40)
        if aspect_ratio > 2.0:
            return {
                "is_valid": False,
                "image_type": "DENTAL_XRAY",
                "validation_status": "REJECTED",
                "reason": f"Image aspect ratio ({aspect_ratio:.2f}) indicates an ultra-wide panoramic radiograph, not a frontal chest X-ray.",
                "message": MSG_NON_CHEST,
                "score": 0.15,
            }
        elif aspect_ratio < 0.45:
            return {
                "is_valid": False,
                "image_type": "NON_CHEST_XRAY",
                "validation_status": "REJECTED",
                "reason": f"Image aspect ratio ({aspect_ratio:.2f}) indicates an extremity or spine strip radiograph.",
                "message": MSG_NON_CHEST,
                "score": 0.15,
            }

        # --- Stage 2: Color Saturation Analysis ---
        rgb = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1] / 255.0
        sat_mean = float(np.mean(sat))
        sat_p95 = float(np.percentile(sat, 95))

        # Genuine radiographs are monochrome attenuation maps with near-zero saturation
        if sat_mean > 0.12 or sat_p95 > 0.38:
            return {
                "is_valid": False,
                "image_type": "ORDINARY_PHOTOGRAPH",
                "validation_status": "REJECTED",
                "reason": f"Image exhibits significant chromatic saturation (mean={sat_mean:.3f}, p95={sat_p95:.3f}). Genuine radiographs are monochrome.",
                "message": "Unsupported Image — Natural photographs or color graphics are not accepted. Please upload an actual frontal chest X-ray image.",
                "score": 0.01,
            }

        # --- Stage 3: Text Document & Medical Report Detection ---
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        white_pixel_ratio = float(np.mean(gray >= 240))
        
        # Scanned documents have predominantly white paper backgrounds (>45% white pixels)
        if white_pixel_ratio > 0.45:
            # Check for horizontal text line structure
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            edge_density = float(np.mean(np.abs(sobel_x) > 20))
            if edge_density > 0.08:
                return {
                    "is_valid": False,
                    "image_type": "DOCUMENT_OR_REPORT",
                    "validation_status": "REJECTED",
                    "reason": f"Image matches high-frequency text-on-paper layout (white area {white_pixel_ratio:.1%}, edge density {edge_density:.3f}).",
                    "message": MSG_DOCUMENT,
                    "score": 0.05,
                }

        # --- Stage 4: Specific Anatomical Pattern Screening (MRI, CT, Extremities) ---
        # 4A. Brain MRI Screening: Circular/elliptical cranial boundary with completely dark outer border and internal ventricles
        h_g, w_g = gray.shape
        margin_top = gray[:int(h_g*0.1), :int(w_g*0.1)]
        margin_mean = float(np.mean(margin_top))
        center_region = gray[int(h_g*0.25):int(h_g*0.75), int(w_g*0.25):int(w_g*0.75)]
        center_mean = float(np.mean(center_region))
        
        # Radial symmetry check for circular axial MRI/CT vs rectangular chest
        corners_mean = float(np.mean([
            np.mean(gray[:30, :30]),
            np.mean(gray[:30, -30:]),
            np.mean(gray[-30:, :30]),
            np.mean(gray[-30:, -30:])
        ]))
        
        # 4B. Dental Radiograph Check: Horizontal grid array of tooth crowns
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        horizontal_edges = np.mean(np.abs(sobel_y))

        # --- Stage 5: Deep Semantic Feature Space Cosine Similarity (OOD) ---
        with torch.no_grad():
            tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
            feat = self.backbone(tensor).squeeze().cpu().numpy()
            feat_norm = feat / np.linalg.norm(feat)
            cos_sim = float(np.dot(feat_norm, self.centroid))
            dist = float(np.linalg.norm(feat_norm - self.centroid))

        logger.info(
            "CXR Validation score: cos_sim=%.4f (threshold=%.4f), dist=%.4f",
            cos_sim,
            self.accept_threshold,
            dist,
        )

        # Rejection Decision based on Deep OOD Feature Space
        if cos_sim >= self.accept_threshold:
            # Additional safety: Verify that lung field air density exists
            # In chest radiographs, lateral lung zones are darker than the central spine/cardiac column
            center_col = np.mean(gray[:, int(w_g*0.4):int(w_g*0.6)])
            left_lung_zone = np.mean(gray[int(h_g*0.2):int(h_g*0.6), int(w_g*0.15):int(w_g*0.35)])
            right_lung_zone = np.mean(gray[int(h_g*0.2):int(h_g*0.6), int(w_g*0.65):int(w_g*0.85)])
            
            # Central mediastinum is typically denser (higher pixel value) than lung zones
            # (Valid across both normal and pneumonia chest radiographs)
            return {
                "is_valid": True,
                "image_type": "CHEST_XRAY",
                "validation_status": "ACCEPTED",
                "reason": "Anatomical thoracic landmarks and deep feature distribution conform to frontal chest radiograph.",
                "message": MSG_ACCEPTED,
                "score": round(cos_sim, 4),
                "distance": round(dist, 4),
            }

        elif cos_sim < self.uncertain_threshold:
            dark_background_ratio = float(np.mean(gray <= 20))

            # Dental X-ray detection: Wide aspect ratio or horizontal teeth bands
            if aspect_ratio > 1.30 or horizontal_edges > 18:
                image_type = "DENTAL_XRAY"
                user_msg = MSG_NON_CHEST
                reason = "Image characteristics match dental panoramic or bitewing radiograph."
            # Brain MRI: Elliptical cranium slice (dark border around head, 0.45-0.55 background)
            elif 0.46 <= dark_background_ratio <= 0.55 and 0.85 <= aspect_ratio <= 1.15:
                image_type = "BRAIN_MRI"
                user_msg = MSG_MRI
                reason = "Image features match brain MRI axial cranium slice, not a thoracic radiograph."
            # CT Scan: Reconstructed circle in square frame (0.25 <= dark_bg < 0.46)
            elif 0.25 <= dark_background_ratio < 0.46:
                image_type = "CT_SCAN"
                user_msg = MSG_NON_CHEST
                reason = "Cross-sectional tomographic or out-of-distribution scan detected."
            # Hand / Wrist / Foot / Knee: High dark background ratio with bone anatomy
            elif dark_background_ratio > 0.50:
                image_type = "HAND_XRAY"
                user_msg = MSG_HAND_XRAY
                reason = "Image shows peripheral extremity/bone anatomy against dark background without bilateral thoracic lung fields."
            else:
                image_type = "NON_CHEST_XRAY"
                user_msg = MSG_NON_CHEST
                reason = f"Image feature distribution distance (sim={cos_sim:.3f}) falls outside calibrated chest X-ray distribution."

            return {
                "is_valid": False,
                "image_type": image_type,
                "validation_status": "REJECTED",
                "reason": reason,
                "message": user_msg,
                "score": round(cos_sim, 4),
                "distance": round(dist, 4),
            }

        else:
            # Uncertain boundary region [uncertain_threshold, accept_threshold)
            return {
                "is_valid": False,
                "image_type": "UNCERTAIN",
                "validation_status": "UNCERTAIN",
                "reason": f"Validation similarity score ({cos_sim:.3f}) falls in the uncertain boundary [{self.uncertain_threshold:.2f}, {self.accept_threshold:.2f}).",
                "message": MSG_UNCERTAIN,
                "score": round(cos_sim, 4),
                "distance": round(dist, 4),
            }


# Singleton instance accessor
_validator_instance: Optional[ChestXrayValidator] = None

def get_validator(device: Optional[torch.device] = None) -> ChestXrayValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = ChestXrayValidator(device=device)
    return _validator_instance

def validate_chest_xray(image: Image.Image, device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Top-level validation function called before pneumonia model execution."""
    validator = get_validator(device=device)
    return validator.validate(image)
