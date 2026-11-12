"""
test_ood_calibration.py
-----------------------
Tests feature-space distance, morphology analysis, and calibrated thresholds
across the 12 benchmark test cases.
"""

import sys
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing feature backbone on", DEVICE, "...")
# Use MobileNetV3-Small or ResNet18 as dedicated lightweight validator feature extractor
backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
backbone.classifier = torch.nn.Identity() # output 576-d feature vector
backbone = backbone.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 1. Compute Chest X-Ray In-Distribution Centroid from 50 training images
chest_train_dir = ROOT.parent / "dataset" / "train"
if not chest_train_dir.exists():
    chest_train_dir = ROOT.parent / "data" / "raw" / "train"
normal_imgs = list((chest_train_dir / "NORMAL").glob("*.jpeg"))[:25]
pneumonia_imgs = list((chest_train_dir / "PNEUMONIA").glob("*.jpeg"))[:25]
train_cxr_paths = normal_imgs + pneumonia_imgs

print(f"Extracting embeddings from {len(train_cxr_paths)} reference chest X-rays...")
embeddings = []
with torch.no_grad():
    for p in train_cxr_paths:
        img = Image.open(p).convert("RGB")
        t = transform(img).unsqueeze(0).to(DEVICE)
        feat = backbone(t).squeeze().cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        embeddings.append(feat)

centroid = np.mean(embeddings, axis=0)
centroid = centroid / np.linalg.norm(centroid)
print("Centroid computed. Shape:", centroid.shape)

# 2. Evaluate all 12 test benchmark cases
test_dir = ROOT / "dataset" / "test"
test_cases = [
    ("Normal CXR", test_dir / "CHEST_XRAY/01_normal_chest_xray.jpeg", True),
    ("Pneumonia CXR", test_dir / "CHEST_XRAY/02_pneumonia_chest_xray.jpeg", True),
    ("Hand X-ray", test_dir / "NON_CHEST/03_hand_xray.png", False),
    ("Wrist X-ray", test_dir / "NON_CHEST/04_wrist_xray.png", False),
    ("Foot X-ray", test_dir / "NON_CHEST/05_foot_xray.png", False),
    ("Knee X-ray", test_dir / "NON_CHEST/06_knee_xray.png", False),
    ("Dental X-ray", test_dir / "NON_CHEST/07_dental_xray.png", False),
    ("Brain MRI", test_dir / "NON_CHEST/08_brain_mri.png", False),
    ("CT Scan", test_dir / "NON_CHEST/09_ct_scan.png", False),
    ("Photograph", test_dir / "NON_CHEST/10_photograph.jpg", False),
    ("Medical Report", test_dir / "NON_CHEST/11_medical_report.png", False),
    ("Corrupted", test_dir / "NON_CHEST/12_corrupted_image.jpg", False),
]

print("\n" + "=" * 75)
print(f"{'Test Case':<18} | {'Valid?':<7} | {'Cosine Sim':<12} | {'Euclid Dist':<12} | {'Saturation':<10}")
print("=" * 75)

for name, path, expected_valid in test_cases:
    if not path.exists():
        print(f"{name:<18} | MISSING FILE")
        continue
    
    # Try decoding
    try:
        pil_img = Image.open(path)
        pil_img.verify() # verify integrity
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"{name:<18} | {str(expected_valid):<7} | {'CORRUPT/FAIL':<12} | {'-':<12} | {'-':<10}")
        continue
    
    # Color saturation
    rgb_np = np.array(pil_img)
    hsv = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]) / 255.0)
    
    # Embedding similarity
    with torch.no_grad():
        t = transform(pil_img).unsqueeze(0).to(DEVICE)
        feat = backbone(t).squeeze().cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        cos_sim = float(np.dot(feat, centroid))
        dist = float(np.linalg.norm(feat - centroid))
    
    print(f"{name:<18} | {str(expected_valid):<7} | {cos_sim:12.4f} | {dist:12.4f} | {sat_mean:10.4f}")

print("=" * 75)

