# MediScan AI — ML Inference Services

This directory hosts the isolated Python FastAPI microservices that provide real-time diagnostic inference, out-of-distribution (OOD) validation gating, and Grad-CAM spatial activation mapping.

---

## Architecture Overview

```text
DL/ml-services/
├── chest-xray/          # FastAPI inference service on port 8000
│   ├── app.py           # FastAPI endpoints (/health, /validate, /predict)
│   ├── inference.py     # DenseNet-121 model loading & Grad-CAM generator
│   ├── validator.py     # MobileNetV3-Small OOD chest X-ray screening gate
│   ├── cxr_reference_distribution.npz # Precomputed CXR embedding distribution
│   ├── requirements.txt # Service dependencies
│   ├── .env.example     # Port & checkpoint path configuration
│   └── README.md
│
├── brain-mri/           # FastAPI inference service on port 8001
│   ├── app.py           # FastAPI endpoints (/health, /validate, /predict)
│   ├── mri_inference.py # ResNet-18 4-class classifier & Grad-CAM generator
│   ├── validator.py     # MobileNetV3-Small OOD brain MRI screening gate
│   ├── mri_validator_artifacts.npz # Calibrated centroid & covariance stats
│   ├── mri_validator_report.json   # Held-out evaluation verification report
│   ├── requirements.txt # Service dependencies
│   ├── .env.example     # Port & checkpoint path configuration
│   └── README.md
│
└── shared/              # Common schemas & image utility contracts
    └── README.md
```

---

## Service Endpoints & Configuration

### 1. Chest X-Ray ML Service (`port 8000`)

- **Root Directory**: `DL/ml-services/chest-xray`
- **Canonical Checkpoint**: `DL/chest-xray/models/checkpoints/best_model_pytorch.pth`
- **Endpoints**:
  - `GET /health`: Model status & device availability.
  - `POST /validate`: Fail-closed chest X-ray validation pre-check.
  - `POST /predict`: Complete validation + DenseNet-121 inference + Grad-CAM generation.
- **Run Command**:
  ```powershell
  cd DL/ml-services/chest-xray
  ..\..\chest-xray\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
  ```

### 2. Brain MRI ML Service (`port 8001`)

- **Root Directory**: `DL/ml-services/brain-mri`
- **Canonical Checkpoint**: `DL/brain-mri-tumor/models/checkpoints/tumor_model.pth`
- **Endpoints**:
  - `GET /health`: Model status, device availability & validator readiness.
  - `POST /validate`: Fail-closed brain MRI validation pre-check.
  - `POST /predict`: Complete validation + ResNet-18 4-class inference + Grad-CAM generation.
- **Run Command**:
  ```powershell
  cd DL/ml-services/brain-mri
  ..\..\brain-mri-tumor\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
  ```

---

## Fail-Closed Validation Policy

Neither diagnostic classifier runs until the image passes its dedicated validation stage:
- Non-chest radiographs, photographs, MRI slices, CT scans, and corrupted images sent to `:8000` return HTTP `422` with a structured rejection payload.
- Non-brain scans, photographs, chest X-rays, and corrupt images sent to `:8001` return HTTP `422` with a structured rejection payload.
- In both services, diagnostic inference and Grad-CAM generation are never executed on rejected images.

