# Chest X-Ray ML Service (Port 8000)

FastAPI inference service for binary pneumonia classification (`Normal` vs `Pneumonia`) from frontal chest radiographs.

---

## Quickstart

```powershell
# From repository root
cd DL/ml-services/chest-xray
..\..\chest-xray\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

- Health Check: `http://127.0.0.1:8000/health`
- Model Checkpoint: `DL/chest-xray/models/checkpoints/best_model_pytorch.pth`
- Validator Artifacts: `cxr_reference_distribution.npz`

