# MediScan AI — Image Validation Dataset (Chest vs Non-Chest)

This dataset is designed to train and calibrate the Chest X-Ray Validation & Out-Of-Distribution (OOD) filter.
Only frontal chest radiographs (PA/AP) must be admitted to CHEST_XRAY.

## Directory Structure
- `train/`
  - `CHEST_XRAY/`: Frontal chest X-rays (Normal, Pneumonia, Infiltration, Effusion)
  - `NON_CHEST/`: Radiographs of other anatomical regions (Hand, Wrist, Spine, Skull, Pelvis, Dental, Extremities), other modalities (MRI, CT, Ultrasound), documents, and natural images.
- `validation/`: Held-out validation partition for calibration of rejection thresholds (Cosine distance, Mahalanobis distance, morphological bounds).
- `test/`: Independent evaluation set across all 12 test modalities.

## Strict Labeling & Partitioning Guidelines
1. No Patient-Level Leakage: Ensure images from the same patient or hospital scan series never span train and test/validation sets.
2. True Grayscale != Chest X-ray: Extremity radiographs, dental scans, skull X-rays and MRI/CT slices are also monochrome radiographs; they MUST be labeled as `NON_CHEST`.
3. Frontal View Only: Lateral chest X-rays or decubitus views should be categorized in NON_CHEST unless lateral projection is explicitly modeled.
