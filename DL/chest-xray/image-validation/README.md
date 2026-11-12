# MediScan AI — Chest X-Ray Image Validation & OOD Screening

This directory contains the dataset skeleton, benchmark test cases, and calibration scripts for the **Chest X-Ray Validation & Out-Of-Distribution (OOD) filter**.

---

## Directory Organization

```text
DL/chest-xray/image-validation/
├── dataset/
│   ├── train/               # Train partition (CHEST_XRAY, NON_CHEST)
│   ├── validation/          # Held-out calibration partition
│   └── test/                # 12-probe benchmark test cases
│       ├── CHEST_XRAY/      # 01_normal_chest_xray.jpeg, 02_pneumonia_chest_xray.jpeg
│       └── NON_CHEST/       # Hand, wrist, foot, knee, dental, brain MRI, CT, photo, report, corrupt
│
├── benchmark_pipeline.py    # Benchmarks the 12 test cases against FastAPI endpoints
├── verify_validator_pipeline.py # Verifies validator logic directly without HTTP overhead
├── build_cxr_reference.py   # Extracts MobileNetV3-Small reference distribution stats
├── test_ood_calibration.py  # Tests cosine similarity & Mahalanobis calibration
├── create_validation_dataset.py # Generates synthetic test fixtures & structure
└── README.md
```

---

## Running Verification

To test the image validation pipeline against the 12 benchmark cases:

```powershell
# Using the Chest X-Ray virtual environment
cd DL\chest-xray\image-validation
..\venv\Scripts\python.exe verify_validator_pipeline.py
```
