# MediScan AI — Brain MRI Tumour Classification Run Guide (Phase 1 Setup)

> **Current Project Stage — Phase 1 Complete (Setup & Verification Only)**
> - **Completed in Phase 1:** Repository cloned, primary Mendeley dataset downloaded/verified/extracted, dataset integrity and audit checks run, isolated Python virtual environment (`venv`) created, CUDA-enabled PyTorch + `requirements.txt` installed, and CLI scripts + GPU/CPU smoke checks verified.
> - **Not Run Automatically (Phase 2 Manual Actions):** No model training, evaluation, Grad-CAM generation, calibration, or checkpoint creation has been run yet. No files in `frontend/`, `backend/`, `ml-service/`, or `Chest-Xray-Pneumonia/` have been modified.

---

## 1. Repository Source & Git Status

- **Local Repository Path:** `<repo-root>\Brain-MRI-Tumour`
- **GitHub Repository (`origin`):** `https://github.com/imjbassi/Brain-Tumor-MRI-Classification`
- **Title & License:** *Brain Tumor MRI Benchmark Audit* (MIT License, Zenodo DOI `10.5281/zenodo.22755070`)
- **Checked-Out Commit:** `d7b51c9` — `Merge pull request #11 from imjbassi/feat/cover-letter-reviewers`
- **Working Tree Status:** Clean relative to upstream (`requirements.txt` and all `src/` files are unmodified; `venv/` is git-ignored, and this guide `MRI_MODEL_RUN_GUIDE.md` was added at repo root).

---

## 2. Primary Dataset Details

- **Source Portal:** [Mendeley Data — Brain Tumor Data (v1)](https://data.mendeley.com/datasets/w4sw3s9f59/1)
- **DOI:** `10.17632/w4sw3s9f59.1`
- **Dataset Metadata (verified via Mendeley public API):**
  - **Name / Version:** `Brain Tumor Data` — Version `1` (Published `2024-01-22T12:23:01.585Z`)
  - **Contributor:** Ayesha Ghaffar
  - **Licence:** Creative Commons Attribution 4.0 International (`CC BY 4.0`, `http://creativecommons.org/licenses/by/4.0`)
- **Downloaded Archive:** `<repo-root>\datasets\brain-mri\Brain Tumor data.zip`
  - **Size:** `155,749,167` bytes (~`148.5 MB`)
  - **Verified SHA-256:** `3F20516326206E71F870B7C0856FD6E5ABCFE0D80606A30D9DEA93D0AC0151BF` (exact match to Mendeley API checksum)

---

## 3. Extracted Dataset Structure & Verified Counts

Extracted to `<repo-root>\datasets\brain-mri\` (with the outer `Brain Tumor data/` archive prefix stripped so `Training/` and `Testing/` sit directly under `datasets\brain-mri\`, and the original `.zip` preserved alongside them):

```text
<repo-root>\datasets\brain-mri\
├── Brain Tumor data.zip        (preserved original archive)
├── Training\                   (5,712 images total)
│   ├── glioma\                 (1,321 images)
│   ├── meningioma\             (1,339 images)
│   ├── notumor\                (1,595 images)
│   └── pituitary\              (1,457 images)
└── Testing\                    (1,311 images total — DO NOT MODIFY OR MIX INTO TRAINING)
    ├── glioma\                 (300 images)
    ├── meningioma\             (306 images)
    ├── notumor\                (405 images)
    └── pituitary\              (300 images)
```

- **Total Images:** `7,023` (`glioma`: `1,621`, `meningioma`: `1,645`, `notumor`: `2,000`, `pituitary`: `1,757`)
- **Class Folder Naming:** Note that the no-tumor folder is named **`notumor`** (not `no_tumor`). `BrainMRIDataset` (`src/data_loader.py`) discovers subdirectories alphabetically:
  - `class_to_idx`: `{'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}`
- **Validation Split Behaviour:** The raw dataset provides `Training/` and `Testing/` folders without a separate `Validation/` folder. By default, `src/data_loader.py` creates a **stratified 80/20 train/validation split** inside `Training/` (`--val_split 0.2`, `--seed 42`), or a **grouped split (`GroupShuffleSplit`)** when `--group_map` is provided.

---

## 4. Python Version & Isolated Environment Setup

- **Virtual Environment Location:** `<repo-root>\Brain-MRI-Tumour\venv`
- **Python Version (venv):** `3.14.2` (`MSC v.1944 64 bit (AMD64)`)
- **pip Version (venv):** `26.3`

### Activate in Windows PowerShell

```powershell
cd "<repo-root>\Brain-MRI-Tumour"
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation with `ExecutionPolicy`, run once for your current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Or use Command Prompt (`cmd.exe`) instead:

```cmd
cd "<repo-root>\Brain-MRI-Tumour"
venv\Scripts\activate.bat
```

---

## 5. Dependency Installation & PyTorch / CUDA Verification

`requirements.txt` was preserved completely unmodified. Installation used the PyTorch CUDA 12.6 index for `torch` and `torchvision` (matching the detected **NVIDIA GeForce RTX 4060 Laptop GPU**), followed by `pip install -r requirements.txt`.

### Verified Installed Core Packages

| Package | Installed Version | `requirements.txt` Spec |
| :--- | :--- | :--- |
| `torch` | `2.14.0+cu126` | `torch==2.14.0+cu126` |
| `torchvision` | `0.29.0+cu126` | `torchvision==0.29.0+cu126` |
| `numpy` | `2.4.2` | `numpy==2.4.2` |
| `scipy` | `1.17.0` | `scipy==1.17.0` |
| `scikit-learn` | `1.9.0` | `scikit-learn==1.9.0` |
| `matplotlib` | `3.10.8` | `matplotlib==3.10.8` |
| `Pillow` | `12.1.1` | `Pillow==12.1.1` |
| `h5py` | `3.16.0` | `h5py==3.16.0` |

### Verified Runtime & Smoke Check Results

- **`torch.cuda.is_available()`:** `True` (`torch.version.cuda == "12.6"`, `device_count == 1`)
- **Detected GPU:** `NVIDIA GeForce RTX 4060 Laptop GPU` (`8.00 GB` / `8188 MiB` VRAM, NVIDIA driver `592.82`)
- **GPU Smoke Test:** Matrix multiplication on `cuda:0` succeeded.
- **Repository Module Imports:** `data_loader`, `model`, `utils`, `train`, `evaluate`, `gradcam`, `inference`, `visualizer`, `audit_leakage`, `calibrate`, `build_patient_split`, and `build_cv_folds` all imported cleanly.
- **Loader & Architecture Smoke Test (No Training):** `BrainMRIDataset` read `5,712` training samples and `1,311` test samples cleanly (`(3, 224, 224)` tensors), and a 4-class `ResNet18` forward pass succeeded on `cuda:0`.
- **Pretrained Backbone Cache Note:** `%USERPROFILE%\.cache\torch\hub\checkpoints` currently caches only earlier DenseNet/MobileNet weights, so `ResNet18` / `ResNet34` / `EfficientNet-B0` ImageNet weights will download automatically (~45 MB for ResNet-18) on your first manual training run. Keep internet access enabled for the first training run.

---

## 6. Dataset Integrity & Benchmark Audit Summary

All `7,023` images were scanned and decoded with `PIL.Image.verify()` and pixel/hash analysis:

- **Corrupt / Unreadable Files:** `0` (`0` non-image files inside `Training/` or `Testing/`)
- **Extensions & Decoded Formats:** All `7,023` files carry `.jpg` extensions; `7,018` are encoded as `JPEG` and `5` are encoded as `PNG` (opened transparently by Pillow and converted via `.convert("RGB")`).
- **Color Modes:** `RGB`: `3,926`, `L` (grayscale): `3,093`, `RGBA`: `3` (all in `notumor`), `P` (palette): `1` (in `notumor`).
- **Image Dimensions:** Width `150`–`1920` px (median `512`), Height `168`–`1446` px (median `512`). `(512, 512)` accounts for `4,742` images.
- **Class-Distribution & Visual Shortcut Observations:**
  - `glioma` (`1,621`): `100.0%` are `(512, 512)`, `100.0%` have dark borders (`1,426` `L`, `195` `RGB`).
  - `meningioma` (`1,645`): `92.9%` dark borders, `1,387` are `(512, 512)` (`937` `RGB`, `708` `L`).
  - `notumor` (`2,000`): `67.7%` dark borders, diverse cropped resolutions such as `(225, 225)` (`317`), `(630, 630)` (`90`), `(236, 236)` (`81`), mostly `RGB` (`1,967`) with only `29` `L` scans.
  - `pituitary` (`1,757`): `94.2%` dark borders, `1,714` are `(512, 512)` (`930` `L`, `827` `RGB`).
  - *Implication:* Models trained without careful evaluation can partially exploit border/resolution/source differences between `notumor` web-sourced images and Figshare `512x512` tumor scans. Always inspect Grad-CAM overlays (`src\gradcam.py`) to confirm attention falls on intracranial tissue rather than borders.
- **Duplicate & Cross-Split Leakage Audit:**
  - Unique `128x128` grayscale pixel hashes: `6,597 / 7,023` (`426` exact/normalized pixel redundancies across the dataset).
  - Within-`Training` duplicates: `207`; Within-`Testing` duplicates: `30`.
  - Cross-split (`Training` vs `Testing`) overlap: `103 / 1,311` exact file-byte MD5 matches and `216 / 1,311` exact `128x128` pixel-identical matches (`meningioma`: `106`, `notumor`: `105`, `pituitary`: `5`), plus `895 / 1,311` 64-bit perceptual aHash matches.
  - Patient-level overlap (`figshare_patient_map.json`): `4,586 / 7,023` images traced to Figshare patient IDs (`233` patients in `Training`, `206` in `Testing`, **`206` shared**; **`761 / 761` (100%)** of traced tumor test images come from patients also present in `Training`).
  - All `7,023` filenames (basenames) are unique, so `--group_map` basename lookup works cleanly if you use grouped splits later.

> **Important Recommendation on Honest Evaluation:**
> Because the released Mendeley `Training`/`Testing` split leaks patients and exact duplicate images across splits, standard evaluation on `..\datasets\brain-mri\Testing` inflates test accuracy (documented extensively in this repository's own `README.md`). Never edit or pollute `..\datasets\brain-mri\Testing`. When you want a leakage-free clinical benchmark later, generate a **separate** patient-disjoint split folder using the repository's split scripts (Section 10).

---

## 7. Exact Manual Training Commands (Phase 2)

> **Protect Pre-existing Paper Figures:** `paper\figures\` and `runs\` in the cloned repository already contain the original paper's published JSONs and PNG figures. Notice that `src\train.py`, `src\evaluate.py`, `src\gradcam.py`, and `src\visualizer.py` default `--fig_dir` to `"paper/figures"`. **Always pass `--fig_dir outputs\figures`** so your training runs do not overwrite the repository's shipped paper figures.

Activate the environment first (`.\venv\Scripts\Activate.ps1`), ensure your working directory is `<repo-root>\Brain-MRI-Tumour`, and run **one** of the commands below:

### Option A — Standard Recommended Baseline (ResNet-18, Data Augmentation, RTX 4060 GPU)

```powershell
python src\train.py `
  --data_dir "..\datasets\brain-mri" `
  --backbone resnet18 `
  --epochs 30 `
  --batch_size 32 `
  --learning_rate 1e-4 `
  --val_split 0.2 `
  --augment `
  --patience 5 `
  --seed 42 `
  --checkpoint_dir checkpoints `
  --fig_dir outputs\figures
```

*(Single-line equivalent for PowerShell/CMD:)*

```powershell
python src\train.py --data_dir "..\datasets\brain-mri" --backbone resnet18 --epochs 30 --batch_size 32 --learning_rate 1e-4 --val_split 0.2 --augment --patience 5 --seed 42 --checkpoint_dir checkpoints --fig_dir outputs\figures
```

### Option B — Lower Memory / Smaller Batch Size (If Running on CPU Fallback or Tight VRAM)

```powershell
python src\train.py --data_dir "..\datasets\brain-mri" --backbone resnet18 --epochs 30 --batch_size 16 --learning_rate 1e-4 --val_split 0.2 --augment --patience 5 --seed 42 --checkpoint_dir checkpoints --fig_dir outputs\figures
```

### Supported `src\train.py` CLI Options (verified via `--help`)

- `--backbone {resnet18,resnet34,efficientnet_b0}` (default: `resnet18`)
- `--freeze_backbone` (only train classification head)
- `--batch_size` (default: `32`), `--epochs` (default: `30`), `--learning_rate` (default: `1e-4`), `--val_split` (default: `0.2`), `--patience` (default: `5`), `--seed` (default: `42`)
- `--augment` (random horizontal flip `p=0.5`, rotation `+/-10°`, color jitter `0.1`)
- `--balance_classes` (`WeightedRandomSampler` for class imbalance)
- `--checkpoint_dir` (default: `checkpoints`), `--fig_dir` (default: `paper/figures`), `--group_map` (optional JSON mapping image paths to group/patient IDs)

---

## 8. Exact Evaluation Command (Phase 2)

After `checkpoints\tumor_model.pth` has been created by training:

```powershell
python src\evaluate.py --data_dir "..\datasets\brain-mri" --model_path checkpoints\tumor_model.pth --batch_size 32 --fig_dir outputs\figures
```

Prints the full 4-class precision/recall/F1 `classification_report` and test accuracy to console, and writes evaluation plots into `outputs\figures\`.

---

## 9. Exact Grad-CAM Visualization Command (Phase 2)

Generates a 4-class Grad-CAM overlay grid (`gradcam_examples.png`) for one sample per class:

```powershell
python src\gradcam.py --data_dir "..\datasets\brain-mri" --model_path checkpoints\tumor_model.pth --fig_dir outputs\figures
```

---

## 10. Exact Single-Image Inference & Optional Audit/Split Commands

### Single-Image Prediction (`src\inference.py`)

Example using an actual test scan from the extracted dataset (`glioma`):

```powershell
python src\inference.py --model_path checkpoints\tumor_model.pth --image_path "..\datasets\brain-mri\Testing\glioma\Te-gl_0010.jpg" --top_k 3
```

Example using your own external MRI image path:

```powershell
python src\inference.py --model_path checkpoints\tumor_model.pth --image_path "C:\path\to\your_brain_mri.jpg" --top_k 4
```

### Optional Calibration, Visualization & Leakage-Free Split Helpers

- **Dataset Sample Grid & Class Counts (`src\visualizer.py`):**
  ```powershell
  python src\visualizer.py --data_dir "..\datasets\brain-mri" --num_samples 4 --save --fig_dir outputs\figures
  ```
- **Temperature-Scaling Calibration (`src\calibrate.py`):**
  ```powershell
  python src\calibrate.py --data_dir "..\datasets\brain-mri" --model_path checkpoints\tumor_model.pth --batch_size 32 --seed 42
  ```
- **Leakage & Near-Duplicate Audit (`src\audit_leakage.py`):**
  ```powershell
  python src\audit_leakage.py --data_dir "..\datasets\brain-mri" --fig_dir outputs\audit
  ```
- **Patient-Disjoint Benchmark Note (`src\build_patient_split.py` / `src\build_cv_folds.py`):**
  `figshare_patient_map.json` and `cv_folds.json` store keys relative to `data/...` (e.g. `data/Training/glioma/Tr-glTr_0001.jpg`), because those scripts expect a `.\data` path from inside `Brain-MRI-Tumour`. If you later want to build a patient-disjoint fold in `data_patient/` without copying or modifying `datasets\brain-mri`, create a directory junction `mklink /J data ..\datasets\brain-mri` (in `cmd.exe`) before running `python src\build_patient_split.py --data_dir data --folds cv_folds.json --fold_idx 0 --out_dir data_patient`.

---

## 11. Checkpoint & Output File Locations

| File / Folder | Purpose | Notes |
| :--- | :--- | :--- |
| `checkpoints\tumor_model.pth` | Best validation-loss model checkpoint saved by `train.py` | Stores `model_state`, `class_names`, `model_name` |
| `outputs\figures\history.json` | Epoch-by-epoch train/val loss & accuracy | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\training_curves.png` | Train/val loss & accuracy curves (`train.py`) | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\confusion_matrix_val.png` | Normalized validation confusion matrix (`train.py`) | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\confusion_matrix.png` | Normalized test confusion matrix (`evaluate.py`) | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\roc_curves.png` | One-vs-rest per-class ROC/AUC curves (`evaluate.py`) | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\misclassified_examples.png` | Test-set error gallery (`evaluate.py`) | When `--fig_dir outputs\figures` is passed |
| `outputs\figures\gradcam_examples.png` | 4-class Grad-CAM overlay figure (`gradcam.py`) | When `--fig_dir outputs\figures` is passed |
| `paper\figures\` & `runs\` | Original benchmark author's pre-existing paper outputs | Keep untouched by passing `--fig_dir outputs\figures` |

---

## 12. Windows Troubleshooting

1. **PowerShell `Activate.ps1` cannot be loaded (`running scripts is disabled`):**
   - Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, or activate with `venv\Scripts\activate.bat` in `cmd.exe`.
2. **PyTorch DataLoader multiprocessing (`num_workers=4` hardcoded in `get_dataloaders`):**
   - `src\train.py` and `src\evaluate.py` call `get_dataloaders(...)` with its default `num_workers=4` (not exposed as a CLI flag). Both scripts guard execution under `if __name__ == "__main__":`, which is required on Windows `spawn` multiprocessing. Each worker loads PyTorch, so expect higher startup time (~5–15s on first epoch) and higher system RAM usage.
3. **CUDA Out-of-Memory (OOM) or High RAM Pressure:**
   - Lower `--batch_size` from `32` to `16` or `8` (especially if using `resnet34` or `efficientnet_b0`, or if other apps use GPU VRAM).
4. **First Run Network / Weight Download Error:**
   - `TumorClassifier` loads `IMAGENET1K_V1` pretrained weights on initialization (`resnet18` weights are not currently cached locally). Ensure internet access for the first run of `src\train.py`.
5. **Matplotlib Font Cache Warning (`fontlist-*.matplotlib-lock`):**
   - Harmless if another process holds `~/.matplotlib` lock; plots still render via headless `Agg` / disk save (`fig.savefig`). Do not interrupt while plotting finishes.

---

## 13. Compute & Performance Notes

- **Detected Hardware Advantage:** Because this machine has an **NVIDIA GeForce RTX 4060 Laptop GPU (`8.00 GB` VRAM)** with `CUDA 12.6` active in `venv`, a full 30-epoch `resnet18` run over `5,712` images (`224x224`) typically completes in a few minutes rather than an hour+ on CPU.
- **If Falling Back to CPU-Only Execution on Another Machine:** Reduce `--batch_size` to `16` or `8` (and start with `resnet18` + optionally `--freeze_backbone` or fewer `--epochs` first) to keep RAM usage and per-epoch runtime manageable.

---

## 14. Deactivating the Environment

When finished running commands:

```powershell
deactivate
```

