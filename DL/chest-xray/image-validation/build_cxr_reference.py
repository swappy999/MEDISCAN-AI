"""
build_cxr_reference.py
----------------------
Computes the empirical chest X-ray feature centroid and distribution bounds
from 200 authentic training chest radiographs and saves to:
ml-service/cxr_reference_distribution.npz
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = Path(__file__).resolve().parent.parent.parent / "ml-services" / "chest-xray" / "cxr_reference_distribution.npz"

print("Loading MobileNetV3-Small backbone on", DEVICE, "...")
backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
backbone.classifier = torch.nn.Identity()
backbone = backbone.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dir = ROOT / "dataset" / "train"
if not train_dir.exists():
    train_dir = ROOT / "data" / "raw" / "train"
normal_paths = sorted(list((train_dir / "NORMAL").glob("*.jpeg")))[:100]
pneumonia_paths = sorted(list((train_dir / "PNEUMONIA").glob("*.jpeg")))[:100]
all_paths = normal_paths + pneumonia_paths

print(f"Extracting embeddings from {len(all_paths)} training images...")
embeddings = []
with torch.no_grad():
    for i, p in enumerate(all_paths):
        img = Image.open(p).convert("RGB")
        t = transform(img).unsqueeze(0).to(DEVICE)
        feat = backbone(t).squeeze().cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        embeddings.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(all_paths)}...")

embeddings = np.array(embeddings) # (200, 576)
centroid = np.mean(embeddings, axis=0)
centroid = centroid / np.linalg.norm(centroid)

# Compute within-distribution similarity distribution
sims = np.dot(embeddings, centroid)
min_sim = float(np.min(sims))
mean_sim = float(np.mean(sims))
p5_sim = float(np.percentile(sims, 5))
p1_sim = float(np.percentile(sims, 1))

print(f"Within-distribution similarity stats on 200 training CXRs:")
print(f"  Mean:  {mean_sim:.4f}")
print(f"  5th %: {p5_sim:.4f}")
print(f"  1st %: {p1_sim:.4f}")
print(f"  Min:   {min_sim:.4f}")

# Threshold calibration:
# Allow 99% of genuine chest X-rays while rejecting non-chest (<0.50)
accept_threshold = round(max(0.68, p1_sim - 0.05), 4) # ~0.70
uncertain_threshold = round(accept_threshold - 0.08, 4) # ~0.62

print(f"Calibrated acceptance threshold: {accept_threshold}")
print(f"Calibrated uncertainty threshold: {uncertain_threshold}")

np.savez_compressed(
    OUT_PATH,
    centroid=centroid,
    accept_threshold=accept_threshold,
    uncertain_threshold=uncertain_threshold,
    mean_sim=mean_sim,
    min_sim=min_sim,
    n_samples=len(all_paths)
)
print(f"Saved distribution reference to: {OUT_PATH}")

