# Brain MRI ML Service (Port 8001)

FastAPI inference service for 4-class brain tumour classification (`glioma`, `meningioma`, `no_tumor`, `pituitary`) from axial/sagittal brain MRI scans.

---

## Quickstart

```powershell
# From repository root
cd DL/ml-services/brain-mri
..\..\brain-mri-tumor\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

- Health Check: `http://127.0.0.1:8001/health`
- Model Checkpoint: `DL/brain-mri-tumor/models/checkpoints/tumor_model.pth`
- Validator Artifacts: `mri_validator_artifacts.npz` and `mri_validator_report.json`

