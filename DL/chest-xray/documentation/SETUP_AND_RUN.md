# MediScan AI — Chest X-Ray Pneumonia Model: Setup & Execution Guide

This document provides complete instructions for manually preparing the dataset, training, evaluating, and running inference on the **Chest X-ray Pneumonia Detection Model** inside the **MediScan AI** environment.

---

## 1. Project Overview

- **Module**: Chest X-ray Pneumonia Detection
- **Task**: Binary Classification (`NORMAL` vs `PNEUMONIA`)
- **Backbone Architecture**: DenseNet121 (`keras.applications.DenseNet121`) initialized with ImageNet weights, Global Average Pooling, Dropout (0.5), and a dense sigmoid classification head.
- **Explainability**: Class Activation Mapping via **Grad-CAM** (`tf.GradientTape`) computing gradient saliency over the final dense block convolutional layer (`conv5_block16_concat`).
- **Core Framework**: TensorFlow 2.16+ / Keras 3 with NumPy, OpenCV, Pandas, and Scikit-Learn.
- **Hardware Profile**: 
  - **Primary GPU**: **NVIDIA GeForce RTX 4060 Laptop GPU** (8 GB dedicated GDDR6 VRAM, Driver 592.82, CUDA 12.4 compute capability 8.9 Ada Lovelace architecture).
  - **CPU & RAM**: Intel Core i5 11th Gen (16 GB System RAM).
- **GPU Acceleration**: PyTorch 2.6.0 with CUDA 12.4 acceleration enabled (`cuda:0`). Full mixed-precision training (`torch.amp.autocast`) and tensor core acceleration.

---

## 2. Repository Source Link

- **Original Cloned Repository**: [SWAPI03/Chest_Xray_Disease_Detection](https://github.com/SWAPI03/Chest_Xray_Disease_Detection)
- **Local Directory**: `MediScan AI/Chest-Xray-Pneumonia/`

---

## 3. Python Installation Requirements

- **Python Version**: Python 3.11 (strictly recommended for TensorFlow 2.16+ & Keras 3 compatibility on native Windows).
- **Environment Type**: Dedicated virtual environment (`venv`).
- **Installed Python Binary**: `python (Python 3.11)`

---

## 4. Virtual Environment Activation & Deactivation

The virtual environment is located at:
`<repo-root>\Chest-Xray-Pneumonia\venv`

### How to Activate (Windows PowerShell):
```powershell
# Navigate to the model directory
cd "<repo-root>\Chest-Xray-Pneumonia"

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```

> **Note for PowerShell execution policy**: If script execution is restricted on Windows, run PowerShell as Administrator or set process-level execution:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

### How to Activate (Windows Command Prompt - cmd.exe):
```cmd
cd "<repo-root>\Chest-Xray-Pneumonia"
.\venv\Scripts\activate.bat
```

### How to Deactivate:
When finished working in the environment:
```powershell
deactivate
```

---

## 5. Dependency Installation & Status

All required dependencies have been installed:

- **PyTorch**: `2.6.0+cu124` (NVIDIA CUDA 12.4 enabled, RTX 4060 GPU acceleration)
- **Torchvision**: `0.21.0+cu124` (CUDA enabled)
- **CUDA Device**: `cuda:0` — `NVIDIA GeForce RTX 4060 Laptop GPU` (8.0 GB VRAM)
- **TensorFlow**: `2.21.0` (Native Windows CPU execution)
- **Keras**: `3.15.1` (Multi-backend Keras 3)
- **NumPy**: `2.4.6`
- **OpenCV**: `5.0.0.93` (`opencv-python`)
- **Pillow**: `13.0.0`
- **Pandas**: `3.1.1`
- **Scikit-Learn**: `1.9.1`
- **Matplotlib**: `3.11.2`
- **Seaborn**: `0.13.2`
- **PyYAML**: `6.0.3`
- **Package Installation**: `chest_xray_detection` installed in editable mode (`pip install -e .`)

To reinstall or verify dependencies at any time:
```powershell
pip install -r requirements.txt
pip install -e .
```

---

## 6. Dataset Status & Download Details

The official **Kaggle Chest X-Ray Images (Pneumonia)** dataset has been downloaded and verified locally in the workspace:
- **Local Location**: `<repo-root>\Chest-Xray-Pneumonia\data\raw\`
- **Workspace Mirror Link**: `<repo-root>\datasets\chest-xray-pneumonia\raw\`
- **Total Ingested Images**: 5,856 radiographs
- **Kaggle Reference**: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Verified Partition Counts:
| Split | NORMAL | PNEUMONIA | Total |
| :--- | :---: | :---: | :---: |
| **Train** | 1,207 | 3,488 | 4,695 |
| **Validation** | 142 | 395 | 537 |
| **Test (Held-out)** | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

The validation partition was balanced from the original 16-image set to 537 images using `python data/prepare_data.py`. The held-out 624-image test set remains untouched.

---

## 7. Dataset Extraction Instructions

The repository's configuration (`configs/config.yaml`) expects raw data at `data/raw`:

```
Chest-Xray-Pneumonia/
│
└── data/
    └── raw/
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        └── test/
            ├── NORMAL/
            └── PNEUMONIA/
```

### Steps to Extract on Windows:
1. Extract the downloaded ZIP archive. The extracted root folder is typically named `chest_xray`.
2. Move the contents of `chest_xray` directly into `<repo-root>\Chest-Xray-Pneumonia\data\raw`.
3. Alternatively, copy the images to the central repository at `<repo-root>\datasets\chest-xray-pneumonia`.

Verify that the following directories exist and contain images:
- `data/raw/train/NORMAL`
- `data/raw/train/PNEUMONIA`
- `data/raw/val/NORMAL`
- `data/raw/val/PNEUMONIA`
- `data/raw/test/NORMAL`
- `data/raw/test/PNEUMONIA`

---

## 8. Dataset Preparation Command

The original Kaggle dataset has an uneven validation split (`val/` contains only 16 images). The repository provides an automated verification and balancing script:

```powershell
# Run from Chest-Xray-Pneumonia directory with venv activated
python data/prepare_data.py
```

### What this script does:
1. Scans `data/raw/` to verify folder integrity and image counts.
2. If `val/` has only 16 images, it deterministically moves 10% of `train/` images into `val/` using fixed random seed 42.
3. Leaves the 624-image `test/` partition completely untouched to preserve unbiased evaluation.

### Testing Without Downloading (Synthetic Sample Generator):
If you want to test the full pipeline before downloading the 1.2 GB Kaggle dataset:
```powershell
python data/prepare_data.py --sample --n 30
```
This generates synthetic X-ray images in `data/sample/`.

---

## 9. Model Training Command

You have two execution modes available:

### Option A: GPU-Accelerated Training on NVIDIA GeForce RTX 4060 (Recommended)
This uses your dedicated **RTX 4060 Laptop GPU** (8 GB VRAM) with **CUDA 12.4** and Automatic Mixed Precision (`torch.amp.autocast`). Training is approximately **15x-25x faster** than CPU execution.

```powershell
# Navigate to Chest-Xray-Pneumonia with venv activated
cd "<repo-root>\Chest-Xray-Pneumonia"
.\venv\Scripts\Activate.ps1

# Start training on NVIDIA RTX 4060 GPU
python main_pytorch.py --mode train
```

- **Checkpoint**: Saved to `outputs/models/best_model_pytorch.pth`
- **Estimated Epoch Time on RTX 4060**: ~15–25 seconds per epoch (vs ~5–8 minutes on CPU).
- **GPU Mixed Precision**: Enabled automatically for Ada Lovelace Tensor Cores.

### Option B: Original TensorFlow / Keras Training Pipeline
Runs the cloned repository's original TensorFlow/Keras pipeline:

```powershell
python main.py --mode train
```

> **Note**: On native Windows without WSL2, TensorFlow 2.x runs on CPU. If you wish to use GPU with the TensorFlow pipeline, run via Windows Subsystem for Linux (WSL2).

---

## 10. Model Evaluation Command

### Using the RTX 4060 GPU PyTorch Checkpoint:
```powershell
python main_pytorch.py --mode evaluate --checkpoint outputs/models/best_model_pytorch.pth
```

### Using the TensorFlow / Keras Checkpoint:
```powershell
python main.py --mode evaluate --checkpoint outputs/models/best_model.keras
```

---

## 11. Single-Image Prediction & Grad-CAM Command

### Using RTX 4060 GPU PyTorch Checkpoint:
```powershell
python main_pytorch.py --mode predict --image data/raw/test/PNEUMONIA/person1_virus_6.jpeg --checkpoint outputs/models/best_model_pytorch.pth
```

### Using TensorFlow / Keras Checkpoint:
```powershell
python main.py --mode predict --image data/raw/test/PNEUMONIA/person1_virus_6.jpeg --checkpoint outputs/models/best_model.keras
```

To run inference and generate an attention heatmap on a single X-ray image:

```powershell
python main.py --mode predict --checkpoint outputs/models/best_model.keras --image path/to/your/xray.jpeg
```

### What this does:
1. Loads the trained checkpoint.
2. Preprocesses the target image (RGB conversion, 224×224 resize, ImageNet normalization).
3. Computes the probability of Pneumonia vs Normal.
4. Generates a Grad-CAM heatmap overlay highlighting regions contributing to the prediction.
5. Saves the output figure to `outputs/visualizations/`.

---

## 12. Checkpoint Location

- **Best Model Weights**: `Chest-Xray-Pneumonia/outputs/models/best_model.keras`
- **Format**: Native Keras 3 format (`.keras` ZIP archive containing model architecture, weights, and optimizer state).

---

## 13. Expected Output Files & Structure

```
Chest-Xray-Pneumonia/
│
├── configs/
│   └── config.yaml                     # Master settings
├── outputs/
│   ├── logs/                           # Training CSV logs and timestamps
│   ├── models/
│   │   └── best_model.keras            # Checkpoint saved by early stopping
│   ├── reports/
│   │   ├── evaluation_report.md        # Full markdown evaluation metrics
│   │   └── metrics.json                # Structured metrics dictionary
│   └── visualizations/
│       ├── class_distribution.png      # Dataset split balance
│       ├── confusion_matrix.png        # Test confusion matrix
│       ├── gradcam_example.png         # Saliency map on single test image
│       ├── gradcam_examples.png        # Saliency grid on multiple test cases
│       ├── prediction_examples.png     # Sample predictions with true labels
│       ├── roc_curve.png               # ROC curve and AUC calculation
│       └── training_curves.png         # Loss and AUC progression per epoch
```

---

## 14. Windows Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Access is denied` on `Activate.ps1`** | Windows PowerShell execution policy restriction. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` before activating. |
| **`TensorFlow GPU support is not available on native Windows`** | TensorFlow 2.11+ dropped native Windows GPU support. | Expected and normal for CPU. Your Intel Core i5 1135G7 runs on CPU automatically. If GPU acceleration is required, run under WSL2 or Google Colab (notebook included in `notebooks/train_colab.ipynb`). |
| **`oneDNN custom operations are on` warning** | Standard TensorFlow CPU optimization warning. | Harmless. To suppress, set `$env:TF_ENABLE_ONEDNN_OPTS="0"` in PowerShell. |
| **Out of Memory (OOM) during training** | Large batch size on CPU. | Lower batch size: `python main.py --mode train --batch_size 16`. |
| **Dataset path error** | Folder structure mismatch. | Ensure images are at `data/raw/train/NORMAL`, etc., or edit `configs/config.yaml` to point to your exact path. |

---

## 15. Quick Smoke Test (No Dataset Needed)

To confirm your installation is working properly at any time without downloading data:

```powershell
cd "<repo-root>\Chest-Xray-Pneumonia"
.\venv\Scripts\Activate.ps1
python main.py
```

Expected output:
```
INFO | Device: CPU
INFO | Running DEMO mode (synthetic data)...
INFO | Forward pass OK  input=[4, 224, 224, 3]  output=[4, 1]
  Model : DENSENET121  |  Params: 7.0M  |  Trainable: 7.0M
  Pipeline OK.
```

---

## 16. Chest X-Ray Validation & Out-Of-Distribution (OOD) Rejection System

MediScan AI incorporates a dedicated pre-inference validation and out-of-distribution rejection subsystem (`ml-service/validator.py`). This ensures that **only valid frontal chest radiographs** reach the trained DenseNet-121 pneumonia detection model.

### 5-Stage Defense-in-Depth Pipeline:
1. **Structural & Dimension Gating**:
   - Minimum resolution: $120 \times 120$ pixels.
   - Clinical aspect ratio bounds: $0.45 \le W/H \le 2.0$ (filters panoramic dental strips and extremity strips).
2. **Photometric / Color Saturation Analysis**:
   - Rejects color photographs, internet graphics, and screenshots where mean HSV saturation $S_{\text{mean}} > 0.12$ or 95th percentile $S_{95} > 0.38$. Genuine radiographs are strictly monochrome attenuation maps.
3. **Document & Medical Report Screening**:
   - Identifies scanned paper documents with predominant white backgrounds ($I \ge 240$ ratio $> 45\%$) and high horizontal edge gradient density.
4. **Anatomical & Modality Specific Pattern Recognition**:
   - Identifies cranial elliptical perimeters with dark margins (Brain MRI), circular reconstructed fields (CT cross-sectional scans), and peripheral extremity bones against black backgrounds (Hand, Wrist, Foot, Knee X-rays).
5. **Calibrated Deep Feature Space Cosine Distance**:
   - Extracts 576-dimensional semantic embeddings via MobileNetV3-Small.
   - Evaluates cosine similarity against the empirical CXR centroid vector $\mu_{\text{CXR}}$ derived from 200 authentic training chest X-rays.
   - **Calibrated Decision Boundary**:
     - $\cos(\mathbf{f}, \mu_{\text{CXR}}) \ge 0.7824$: **ACCEPTED** $\rightarrow$ Proceeds to pneumonia inference.
     - $0.7024 \le \cos(\mathbf{f}, \mu_{\text{CXR}}) < 0.7824$: **UNCERTAIN** $\rightarrow$ HTTP 422 Rejected (*"Unable to verify image type — Please upload a clear frontal chest X-ray"*).
     - $\cos(\mathbf{f}, \mu_{\text{CXR}}) < 0.7024$: **REJECTED** $\rightarrow$ HTTP 422 Rejected with specific clinical guidance message.

### Running the Validation Benchmark:
To run the automated 12-case benchmark evaluation suite:
```powershell
cd "<repo-root>"
& ".\Chest-Xray-Pneumonia\venv\Scripts\python.exe" benchmark_pipeline.py
```

### Measured Performance:
- **Total Images Tested**: 12 (2 authentic CXRs, 10 varied OOD images: hand, wrist, foot, knee, dental, brain MRI, CT, photo, document, corrupted).
- **False Acceptance Rate (FAR)**: **0.00%**
- **False Rejection Rate (FRR)**: **0.00%**
- **Sensitivity / True Positive Rate**: **100.00%**
- **Specificity / True Negative Rate**: **100.00%**
- **Inference Safety**: 100% verified — zero predictions, confidence scores, or Grad-CAM heatmaps leak for rejected inputs.


