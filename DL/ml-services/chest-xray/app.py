"""
app.py — FastAPI inference service for MediScan AI.
Loads model once at startup, serves /health and POST /predict.
"""

import io
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

# ---------------------------------------------------------------------------
# Load .env from same directory as this file
# ---------------------------------------------------------------------------
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_base_dir = Path(__file__).resolve().parent
_raw_ckpt = os.getenv("CHECKPOINT_PATH", "../../chest-xray/models/checkpoints/best_model_pytorch.pth")
_resolved_ckpt = (_base_dir / _raw_ckpt).resolve() if not Path(_raw_ckpt).is_absolute() else Path(_raw_ckpt)
if not _resolved_ckpt.exists():
    _fallback_ckpt = (_base_dir / "../../chest-xray/outputs/models/best_model_pytorch.pth").resolve()
    if _fallback_ckpt.exists():
        _resolved_ckpt = _fallback_ckpt
CHECKPOINT_PATH: str = str(_resolved_ckpt)

_raw_results = os.getenv("RESULTS_DIR", "../../../backend/results")
RESULTS_DIR: str = str((_base_dir / _raw_results).resolve() if not Path(_raw_results).is_absolute() else Path(_raw_results))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mediscan.app")

# ---------------------------------------------------------------------------
# Module-level globals (populated at startup)
# ---------------------------------------------------------------------------
_model = None
_gradcam = None
_device = None
_validator = None
_model_loaded: bool = False
_load_error: str = ""

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MediScan AI — ML Inference Service",
    description="Pneumonia detection from chest X-ray images using DenseNet-121.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global _model, _gradcam, _device, _validator, _model_loaded, _load_error

    # Ensure results directory exists
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("Results directory: %s", RESULTS_DIR)

    # Determine device
    try:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", _device)
    except Exception as exc:
        _load_error = f"torch import failed: {exc}"
        logger.error(_load_error)
        return

    # Load model
    try:
        from inference import load_model

        logger.info("Loading model from: %s", CHECKPOINT_PATH)
        _model, _gradcam = load_model(CHECKPOINT_PATH, _device)
        _model_loaded = True
        logger.info("Model loaded successfully.")
    except Exception as exc:
        _load_error = str(exc)
        logger.error("Failed to load model: %s", exc, exc_info=True)

    # Load dedicated Chest X-Ray validator
    try:
        from validator import get_validator

        logger.info("Initializing chest X-ray validator...")
        _validator = get_validator(_device)
        logger.info("Chest X-ray validator loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load validator: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Return service, model, and validator health status."""
    return {
        "status": "ok",
        "model_loaded": _model_loaded,
        "validator_loaded": _validator is not None,
        "checkpoint_path": CHECKPOINT_PATH,
        "device": str(_device) if _device is not None else None,
        "error": _load_error if not _model_loaded else None,
    }


ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.post("/validate")
async def validate_endpoint(file: UploadFile = File(...)):
    """
    Dedicated endpoint to validate whether an uploaded image is a valid frontal chest X-ray.
    Returns 200 with ACCEPTED or 422 with REJECTED / UNCERTAIN without running pneumonia inference.
    """
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=422,
            content={
                "is_valid": False,
                "image_type": "UNSUPPORTED_FORMAT",
                "validation_status": "REJECTED",
                "reason": f"Invalid file type '{content_type}'. Accepted types: image/jpeg, image/png.",
                "message": "Invalid Upload — Please upload an actual chest X-ray image in JPG, JPEG or PNG format.",
                "score": 0.0,
            },
        )

    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={
                "is_valid": False,
                "image_type": "CORRUPTED_FILE",
                "validation_status": "REJECTED",
                "reason": f"Could not decode image: {exc}",
                "message": "Invalid Upload — Image file is corrupted or cannot be decoded. Please upload a valid image file.",
                "score": 0.0,
            },
        )

    from validator import validate_chest_xray

    validation = validate_chest_xray(image_pil, device=_device)
    status_code = 200 if validation.get("is_valid", False) else 422
    return JSONResponse(status_code=status_code, content=validation)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept a chest X-ray image and return a pneumonia prediction.
    Strictly performs Chest X-Ray validation first. If validation fails or is uncertain,
    returns HTTP 422 with rejection details and NEVER calls the pneumonia model.

    Returns JSON with: success, id, prediction, confidence, probabilities,
    heatmap_path, overlay_path, processing_time, model_name, validation.
    """
    # Guard: model not loaded
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": f"Model is not loaded. {_load_error}".strip(),
            },
        )

    # Validate MIME type
    content_type = (file.content_type or "").lower()
    # Strip encoding suffix if present, e.g. "image/jpeg; charset=..."
    content_type = content_type.split(";")[0].strip()
    if content_type not in ALLOWED_MIME_TYPES:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Invalid Upload — Please upload an actual chest X-ray image in JPG, JPEG or PNG format.",
                "validation": {
                    "is_valid": False,
                    "image_type": "UNSUPPORTED_FORMAT",
                    "validation_status": "REJECTED",
                    "reason": f"Invalid file type '{content_type}'. Accepted types: image/jpeg, image/png.",
                    "message": "Invalid Upload — Please upload an actual chest X-ray image in JPG, JPEG or PNG format.",
                    "score": 0.0,
                },
            },
        )

    # Read and decode image bytes
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": "Invalid Upload — Image file is corrupted or cannot be decoded. Please upload a valid image file.",
                "validation": {
                    "is_valid": False,
                    "image_type": "CORRUPTED_FILE",
                    "validation_status": "REJECTED",
                    "reason": f"Could not decode image: {exc}",
                    "message": "Invalid Upload — Image file is corrupted or cannot be decoded. Please upload a valid image file.",
                    "score": 0.0,
                },
            },
        )

    # -----------------------------------------------------------------------
    # STEP 3 & 4: Dedicated Chest X-Ray Validation & OOD Rejection Gating
    # -----------------------------------------------------------------------
    from validator import validate_chest_xray

    validation = validate_chest_xray(image_pil, device=_device)
    logger.info(
        "Validation check: status=%s, type=%s, score=%.4f",
        validation.get("validation_status"),
        validation.get("image_type"),
        validation.get("score", 0.0),
    )

    # Fail closed: If not strictly valid, immediately reject and NEVER call pneumonia model
    if not validation.get("is_valid", False):
        logger.warning(
            "Predict rejected non-chest image: type=%s, reason=%s",
            validation.get("image_type"),
            validation.get("reason"),
        )
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": validation.get("message", "Validation failed."),
                "validation": validation,
            },
        )

    # Generate unique run id and save paths
    uid = uuid4().hex[:12]
    heatmap_filename = f"{uid}_heatmap.png"
    overlay_filename = f"{uid}_overlay.png"
    heatmap_path = str(Path(RESULTS_DIR) / heatmap_filename)
    overlay_path = str(Path(RESULTS_DIR) / overlay_filename)

    # STEP 6: Run pneumonia inference ONLY after validation succeeds
    try:
        from inference import run_inference

        t0 = time.perf_counter()
        result = run_inference(
            model=_model,
            gradcam=_gradcam,
            device=_device,
            image_pil=image_pil,
            heatmap_save_path=heatmap_path,
            overlay_save_path=overlay_path,
        )
        processing_time = round(time.perf_counter() - t0, 4)
    except Exception as exc:
        logger.error("Inference failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )

    return {
        "success": True,
        "id": uid,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "heatmap_path": heatmap_filename if result["heatmap_saved"] else None,
        "overlay_path": overlay_filename if result["overlay_saved"] else None,
        "processing_time": processing_time,
        "model_name": "DenseNet121Pneumonia",
        "validation": validation,
    }
