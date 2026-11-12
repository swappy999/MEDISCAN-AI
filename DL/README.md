# Deep Learning (DL) Architecture — MediScan AI

The `DL/` directory contains all deep learning research, model architectures, training and evaluation pipelines, image validation filters, and FastAPI inference microservices for **MediScan AI**.

---

## Directory Organization

```text
DL/
├── chest-xray/
│   ├── dataset/             # In-distribution Chest X-Ray images (train, val, test)
│   ├── models/              # Checkpoints & architecture definitions
│   │   ├── checkpoints/     # Canonical best_model_pytorch.pth (~80.5 MB)
│   │   └── architecture/    # DenseNet121Pneumonia PyTorch implementation
│   ├── training/            # PyTorch & Keras training scripts
│   ├── testing/             # Single-image prediction & testing
│   ├── preprocessing/       # Data loaders, transforms & augmentation
│   ├── evaluation/          # Evaluation metrics, ROC-AUC, Grad-CAM saliency
│   │   ├── reports/         # metrics.json & evaluation_report.md
│   │   └── visualizations/  # Confusion matrices, ROC curves, Grad-CAM samples
│   ├── image-validation/    # Strict CXR validation & OOD rejection filter
│   │   ├── dataset/         # 12-probe benchmark dataset & partition skeleton
│   │   ├── benchmark_pipeline.py
│   │   ├── verify_validator_pipeline.py
│   │   └── build_cxr_reference.py
│   ├── documentation/       # Setup & execution guide (SETUP_AND_RUN.md)
│   ├── configs/             # Hyperparameter configs (config.yaml)
│   ├── requirements.txt     # Python dependencies for Chest X-Ray pipeline
│   └── README.md            # Module documentation
│
├── brain-mri-tumor/
│   ├── dataset/             # Mendeley Brain Tumor MRI dataset (Training, Testing)
│   ├── models/              # Checkpoints & architecture definitions
│   │   ├── checkpoints/     # Canonical tumor_model.pth (~42.7 MB)
│   │   └── architecture/    # ResNet-18 4-class TumorClassifier implementation
│   ├── training/            # PyTorch training, cross-validation & seed sweeps
│   ├── testing/             # Inference scripts
│   ├── preprocessing/       # Dataset loaders, transforms & patient-group splits
│   ├── evaluation/          # Evaluation, Grad-CAM, leakage audit & calibration
│   ├── documentation/       # Run guide & audit documentation (MRI_MODEL_RUN_GUIDE.md)
│   ├── outputs/             # Generated figures, confusion matrices & logs
│   ├── paper/               # Publication manuscript, figures & LaTeX source
│   ├── src/                 # Full research audit codebase
│   ├── requirements.txt     # Python dependencies for Brain MRI pipeline
│   └── README.md            # Module documentation
│
└── ml-services/
    ├── chest-xray/          # FastAPI inference service for Chest X-Ray (:8000)
    ├── brain-mri/           # FastAPI inference service for Brain MRI (:8001)
    ├── shared/              # Common schemas & API utilities
    ├── requirements.txt     # Inference service dependencies
    └── README.md            # Service running & deployment guide
```

---

## Independent Service Execution

Each inference service inside `DL/ml-services/` runs as an isolated FastAPI application:

### 1. Chest X-Ray ML Service (Port 8000)
```powershell
cd DL/ml-services/chest-xray
..\..\chest-xray\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

### 2. Brain MRI ML Service (Port 8001)
```powershell
cd DL/ml-services/brain-mri
..\..\brain-mri-tumor\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

---

## Core Models

| Modality | Architecture | Checkpoint | Location | Classes |
| :--- | :--- | :--- | :--- | :--- |
| **Chest X-Ray** | DenseNet-121 | `best_model_pytorch.pth` | `DL/chest-xray/models/checkpoints/` | `Normal`, `Pneumonia` |
| **Brain MRI** | ResNet-18 | `tumor_model.pth` | `DL/brain-mri-tumor/models/checkpoints/` | `glioma`, `meningioma`, `no_tumor`, `pituitary` |

