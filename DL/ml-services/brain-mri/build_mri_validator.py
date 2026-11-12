"""
build_mri_validator.py
----------------------
Trains and calibrates the Brain MRI Input Validator & OOD Detector using:
- In-distribution Brain MRIs from datasets/brain-mri/Training (split 80/20
  by Figshare patient ID where available and stratified by class into fit vs val).
  datasets/brain-mri/Testing is strictly untouched for independent evaluation.
- Real Chest X-rays from Chest-Xray-Pneumonia/data/raw/train & val
  (data/raw/test is strictly untouched for independent evaluation).
- Non-overlapping synthetic negative modalities (documents/reports, color photos,
  high-HU gantry silhouettes, narrow bone/strip phantoms, noise) generated with
  dedicated train/val seeds.

Saves artifacts to mri-service/mri_validator_artifacts.npz.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DL_DIR = Path(__file__).resolve().parent.parent.parent
MRI_TRAIN_DIR = DL_DIR / "brain-mri-tumor" / "dataset" / "Training"
CXR_TRAIN_DIR = DL_DIR / "chest-xray" / "dataset" / "train"
CXR_VAL_DIR = DL_DIR / "chest-xray" / "dataset" / "val"
PATIENT_MAP_PATH = DL_DIR / "brain-mri-tumor" / "figshare_patient_map.json"
OUT_PATH = Path(__file__).resolve().parent / "mri_validator_artifacts.npz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALID_EXTS = (".jpg", ".jpeg", ".png")
MRI_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def load_backbone(device: torch.device) -> torch.nn.Module:
    cached_weights = (
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "mobilenet_v3_small-047dcff4.pth"
    )
    if cached_weights.exists():
        state_dict = torch.load(cached_weights, map_location=device, weights_only=True)
        backbone = models.mobilenet_v3_small(weights=None)
        backbone.load_state_dict(state_dict)
    else:
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )
    backbone.classifier = torch.nn.Identity()
    return backbone.to(device).eval()


TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def extract_embeddings_from_paths(
    backbone: torch.nn.Module, paths: List[Path], batch_size: int = 64
) -> np.ndarray:
    feats = []
    batch_tensors = []
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            batch_tensors.append(TRANSFORM(img))
            if len(batch_tensors) == batch_size or i == len(paths) - 1:
                batch = torch.stack(batch_tensors).to(DEVICE)
                out = backbone(batch).cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
                feats.append(out / norms)
                batch_tensors.clear()
    return np.vstack(feats)


def extract_embeddings_from_images(
    backbone: torch.nn.Module, images: List[Image.Image], batch_size: int = 64
) -> np.ndarray:
    feats = []
    batch_tensors = []
    with torch.no_grad():
        for i, img in enumerate(images):
            batch_tensors.append(TRANSFORM(img.convert("RGB")))
            if len(batch_tensors) == batch_size or i == len(images) - 1:
                batch = torch.stack(batch_tensors).to(DEVICE)
                out = backbone(batch).cpu().numpy().astype(np.float64)
                norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
                feats.append(out / norms)
                batch_tensors.clear()
    return np.vstack(feats)


def make_synthetic_negatives(seed: int, n_per_kind: int = 40) -> List[Image.Image]:
    """Generate diverse synthetic non-MRI negative images for fit/val only (disjoint from test)."""
    rng = np.random.default_rng(seed)
    images: List[Image.Image] = []

    for _ in range(n_per_kind):
        # 1. Color photos / smooth gradients / textured scenes
        h, w = int(rng.integers(240, 512)), int(rng.integers(240, 512))
        c1 = rng.integers(20, 240, size=3)
        c2 = rng.integers(20, 240, size=3)
        grad = np.linspace(0, 1, h)[:, None, None]
        arr = ((1 - grad) * c1 + grad * c2).astype(np.uint8)
        arr = np.repeat(arr, w, axis=1)
        im = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(im)
        for _ in range(4):
            x0, y0 = int(rng.integers(0, w // 2)), int(rng.integers(0, h // 2))
            x1, y1 = int(rng.integers(w // 2, w)), int(rng.integers(h // 2, h))
            col = tuple(int(v) for v in rng.integers(0, 255, size=3))
            draw.rectangle([x0, y0, x1, y1], fill=col)
        images.append(im)

        # 2. Text/document-like pages & dark screenshots
        bg = int(rng.choice([245, 252, 24, 36]))
        fg = 20 if bg > 128 else 220
        doc = Image.new("RGB", (384, 512), (bg, bg, bg))
        d = ImageDraw.Draw(doc)
        d.rectangle([20, 20, 364, 56], fill=(fg, fg, fg))
        for y in range(80, 480, int(rng.integers(18, 28))):
            line_w = int(rng.integers(120, 340))
            d.rectangle([28, y, 28 + line_w, y + 6], fill=(fg, fg, fg))
        images.append(doc)

        # 3. Non-brain bone/tube X-ray-like silhouettes & circular CT-like phantoms
        im_rad = Image.new("RGB", (384, 384), (10, 10, 10))
        dr = ImageDraw.Draw(im_rad)
        if rng.random() < 0.5:
            # Bone / extremity-like shafts
            for x_off in rng.integers(90, 290, size=int(rng.integers(2, 5))):
                dr.rectangle([int(x_off), 40, int(x_off) + 24, 340], fill=(185, 185, 185))
                dr.ellipse([int(x_off) - 6, 170, int(x_off) + 30, 205], fill=(220, 220, 220))
        else:
            # Body CT-like oval torso + bright posterior spine dot
            dr.ellipse([30, 70, 354, 314], fill=(80, 80, 80))
            dr.ellipse([172, 235, 212, 275], fill=(235, 235, 235))
            dr.ellipse([100, 140, 165, 205], fill=(115, 115, 115))
        images.append(im_rad)

        # 4. Random noise / ultrasound speckle frames
        noise = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
        images.append(Image.fromarray(noise, mode="RGB"))

    return images


def split_mri_training_data() -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Split Training Brain MRIs 80/20 into fit vs validation without patient leakage
    where Figshare patient IDs exist.
    """
    pmap = {}
    if PATIENT_MAP_PATH.exists():
        raw_map = json.load(open(PATIENT_MAP_PATH, "r", encoding="utf-8"))
        # Key by basename (all 7,023 basenames verified unique)
        pmap = {Path(k).name: str(v["pid"]) for k, v in raw_map.items()}

    samples: List[Tuple[Path, int]] = []
    groups: List[str] = []
    for cls_idx, cls_name in enumerate(MRI_CLASSES):
        cls_dir = MRI_TRAIN_DIR / cls_name
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in VALID_EXTS:
                samples.append((p, cls_idx))
                groups.append(pmap.get(p.name, f"img_{p.name}"))

    labels = np.array([lbl for _, lbl in samples])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    fit_idx, val_idx = next(gss.split(samples, labels, groups=groups))

    fit_samples = [samples[i] for i in fit_idx]
    val_samples = [samples[i] for i in val_idx]
    return fit_samples, val_samples


def collect_cxr_paths(folder: Path, limit: int = 1600) -> List[Path]:
    paths: List[Path] = []
    for sub in sorted(folder.iterdir()):
        if sub.is_dir():
            paths.extend(
                p for p in sorted(sub.iterdir()) if p.suffix.lower() in VALID_EXTS
            )
    rng = np.random.default_rng(42)
    if len(paths) > limit:
        idx = rng.choice(len(paths), size=limit, replace=False)
        paths = [paths[i] for i in sorted(idx)]
    return paths


def main():
    print(f"Building Brain MRI Validator & OOD Detector on {DEVICE}...")
    backbone = load_backbone(DEVICE)

    fit_mri, val_mri = split_mri_training_data()
    fit_mri_paths = [p for p, _ in fit_mri]
    fit_mri_labels = np.array([lbl for _, lbl in fit_mri])
    val_mri_paths = [p for p, _ in val_mri]
    print(f"Brain MRI Training partitioned -> Fit: {len(fit_mri_paths)}, Val: {len(val_mri_paths)}")

    # Extract MRI embeddings
    fit_mri_emb = extract_embeddings_from_paths(backbone, fit_mri_paths)
    val_mri_emb = extract_embeddings_from_paths(backbone, val_mri_paths)

    # Split Chest X-ray train pool cleanly into Fit (1200) vs Val (400) + raw/val (16)
    cxr_pool = collect_cxr_paths(CXR_TRAIN_DIR, limit=1600)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=400, random_state=42)
    dummy_y = np.array([0 if "NORMAL" in str(p) else 1 for p in cxr_pool])
    cxr_fit_idx, cxr_val_idx = next(sss.split(cxr_pool, dummy_y))
    cxr_fit_paths = [cxr_pool[i] for i in cxr_fit_idx]
    cxr_val_paths = [cxr_pool[i] for i in cxr_val_idx] + collect_cxr_paths(CXR_VAL_DIR, limit=100)

    print(f"Chest X-Ray negatives -> Fit: {len(cxr_fit_paths)}, Val: {len(cxr_val_paths)}")
    fit_cxr_emb = extract_embeddings_from_paths(backbone, cxr_fit_paths)
    val_cxr_emb = extract_embeddings_from_paths(backbone, cxr_val_paths)

    # Synthetic OOD fit & val negatives (disjoint seeds 101 vs 202)
    fit_syn_imgs = make_synthetic_negatives(seed=101, n_per_kind=100)  # 400 images
    val_syn_imgs = make_synthetic_negatives(seed=202, n_per_kind=40)   # 160 images
    fit_syn_emb = extract_embeddings_from_images(backbone, fit_syn_imgs)
    val_syn_emb = extract_embeddings_from_images(backbone, val_syn_imgs)

    # 1. Compute 4 class centroids & shared Ledoit-Wolf shrinkage precision matrix
    mri_centroids = np.zeros((len(MRI_CLASSES), fit_mri_emb.shape[1]), dtype=np.float64)
    centered_diffs = []
    for c_idx in range(len(MRI_CLASSES)):
        c_emb = fit_mri_emb[fit_mri_labels == c_idx]
        mu = np.mean(c_emb, axis=0)
        mu = mu / (np.linalg.norm(mu) + 1e-12)
        mri_centroids[c_idx] = mu
        centered_diffs.append(c_emb - mu)

    centered_all = np.vstack(centered_diffs)
    lw = LedoitWolf(assume_centered=True).fit(centered_all)
    precision_matrix = lw.precision_.astype(np.float64)

    # CXR prototype for diagnostic labeling of chest X-ray uploads
    cxr_centroid = np.mean(fit_cxr_emb, axis=0)
    cxr_centroid = cxr_centroid / (np.linalg.norm(cxr_centroid) + 1e-12)

    # 2. Supervised Binary Logistic Classifier (Brain MRI=1 vs Negative=0)
    X_fit = np.vstack([fit_mri_emb, fit_cxr_emb, fit_syn_emb])
    y_fit = np.concatenate(
        [
            np.ones(len(fit_mri_emb), dtype=int),
            np.zeros(len(fit_cxr_emb) + len(fit_syn_emb), dtype=int),
        ]
    )
    scaler = StandardScaler().fit(X_fit)
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(scaler.transform(X_fit), y_fit)

    # Helper to score embeddings identically to runtime validator
    def score_embeddings(emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cos_all = np.dot(emb, mri_centroids.T)
        max_cos = np.max(cos_all, axis=1)

        best_maha = np.full(len(emb), np.inf, dtype=np.float64)
        for c_idx in range(len(MRI_CLASSES)):
            diff = emb - mri_centroids[c_idx]
            d2 = np.sum((diff @ precision_matrix) * diff, axis=1)
            d = np.sqrt(np.maximum(0.0, d2))
            best_maha = np.minimum(best_maha, d)

        probs = clf.predict_proba(scaler.transform(emb))[:, 1]
        return max_cos, best_maha, probs

    val_mri_cos, val_mri_maha, val_mri_prob = score_embeddings(val_mri_emb)
    val_ood_emb = np.vstack([val_cxr_emb, val_syn_emb])
    val_ood_cos, val_ood_maha, val_ood_prob = score_embeddings(val_ood_emb)

    print("\n--- Validation Set Calibration Distributions ---")
    print(
        f"Val Brain MRI (n={len(val_mri_emb)}): "
        f"cos p0.5={np.percentile(val_mri_cos, 0.5):.4f}, p1={np.percentile(val_mri_cos, 1):.4f}, med={np.median(val_mri_cos):.4f} | "
        f"maha p99={np.percentile(val_mri_maha, 99):.2f}, p99.5={np.percentile(val_mri_maha, 99.5):.2f}, max={np.max(val_mri_maha):.2f} | "
        f"prob p0.5={np.percentile(val_mri_prob, 0.5):.4f}, p1={np.percentile(val_mri_prob, 1):.4f}"
    )
    print(
        f"Val Non-MRI OOD (n={len(val_ood_emb)}): "
        f"cos max={np.max(val_ood_cos):.4f}, p99={np.percentile(val_ood_cos, 99):.4f} | "
        f"maha min={np.min(val_ood_maha):.2f}, p1={np.percentile(val_ood_maha, 1):.2f} | "
        f"prob max={np.max(val_ood_prob):.4f}, p99={np.percentile(val_ood_prob, 99):.4f}"
    )

    # Calibrated thresholds anchored to MRI validation distribution & OOD separation margin
    cos_mri_p1 = float(np.percentile(val_mri_cos, 1.0))
    cos_ood_p99 = float(np.percentile(val_ood_cos, 99.0))
    cos_accept = round(max(cos_ood_p99 + 0.03, min(cos_mri_p1 - 0.01, 0.68)), 4)
    cos_uncertain = round(max(0.45, cos_accept - 0.06), 4)

    maha_mri_p99 = float(np.percentile(val_mri_maha, 99.0))
    maha_ood_min = float(np.min(val_ood_maha))
    maha_accept = round(min(max(maha_mri_p99 * 1.10, maha_mri_p99 + 3.0), (maha_mri_p99 + maha_ood_min) / 2.0 + 4.0) if maha_ood_min > maha_mri_p99 else maha_mri_p99 + 3.0, 2)
    maha_uncertain = round(maha_accept + 6.0, 2)

    prob_accept = round(max(0.80, float(np.percentile(val_mri_prob, 1.0)) * 0.90), 4)
    prob_uncertain = round(max(0.50, prob_accept - 0.15), 4)

    val_mri_accept = (val_mri_cos >= cos_accept) & (val_mri_maha <= maha_accept) & (val_mri_prob >= prob_accept)
    val_ood_reject = (val_ood_cos < cos_accept) | (val_ood_maha > maha_accept) | (val_ood_prob < prob_accept)

    print("\n--- Calibrated Thresholds ---")
    print(f"cos_accept={cos_accept}, cos_uncertain={cos_uncertain}")
    print(f"maha_accept={maha_accept}, maha_uncertain={maha_uncertain}")
    print(f"prob_accept={prob_accept}, prob_uncertain={prob_uncertain}")
    print(f"Val Brain MRI Acceptance Rate: {np.mean(val_mri_accept)*100:.2f}% ({np.sum(val_mri_accept)}/{len(val_mri_emb)})")
    print(f"Val OOD Rejection Rate:        {np.mean(val_ood_reject)*100:.2f}% ({np.sum(val_ood_reject)}/{len(val_ood_emb)})")

    np.savez_compressed(
        OUT_PATH,
        mri_class_names=np.array(MRI_CLASSES),
        mri_centroids=mri_centroids,
        precision_matrix=precision_matrix,
        cxr_centroid=cxr_centroid,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        clf_coef=clf.coef_[0],
        clf_intercept=np.float64(clf.intercept_[0]),
        cos_accept=np.float64(cos_accept),
        cos_uncertain=np.float64(cos_uncertain),
        maha_accept=np.float64(maha_accept),
        maha_uncertain=np.float64(maha_uncertain),
        prob_accept=np.float64(prob_accept),
        prob_uncertain=np.float64(prob_uncertain),
        # Verified empirically across all 5,712 training Brain MRIs (max sat_mean=0.1498, max white_ratio=0.1872)
        sat_mean_max=np.float64(0.18),
        white_ratio_max=np.float64(0.42),
    )
    print(f"\nSaved Brain MRI validator artifacts to: {OUT_PATH}")


if __name__ == "__main__":
    main()

