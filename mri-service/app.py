"""
app.py — Isolated FastAPI inference service for MediScan AI Brain MRI module.
Serves /health, POST /validate, and POST /predict on port 8001.
Strictly validates Brain MRI inputs before running the ResNet-18 tumour classifier.
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

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path)

_base_dir = Path(__file__).resolve().parent
_raw_ckpt = os.getenv(
    "CHECKPOINT_PATH", "../Brain-MRI-Tumour/checkpoints/tumor_model.pth"
)
CHECKPOINT_PATH: str = str(
    (_base_dir / _raw_ckpt).resolve()
    if not Path(_raw_ckpt).is_absolute()
    else Path(_raw_ckpt)
)

_raw_results = os.getenv("RESULTS_DIR", "../backend/results")
RESULTS_DIR: str = str(
    (_base_dir / _raw_results).resolve()
    if not Path(_raw_results).is_absolute()
    else Path(_raw_results)
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mediscan.mri.app")

_model = None
_gradcam = None
_class_names = []
_model_identifier = "ResNet18BrainMRI"
_device = None
_validator = None
_model_loaded: bool = False
_load_error: str = ""

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/jpg"}

app = FastAPI(
    title="MediScan AI — Brain MRI ML Inference Service",
    description="4-class Brain MRI tumour classification and Grad-CAM service with strict OOD input validation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    global _model, _gradcam, _class_names, _model_identifier, _device, _validator, _model_loaded, _load_error

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    logger.info("MRI Results directory: %s", RESULTS_DIR)

    try:
        import torch

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("MRI service using device: %s", _device)
    except Exception as exc:
        _load_error = f"torch import failed: {exc}"
        logger.error(_load_error)
        return

    try:
        from mri_inference import load_mri_model

        logger.info("Loading Brain MRI model from: %s", CHECKPOINT_PATH)
        _model, _gradcam, _class_names, _model_identifier = load_mri_model(
            CHECKPOINT_PATH, _device
        )
        _model_loaded = True
    except Exception as exc:
        _load_error = str(exc)
        logger.error("Failed to load Brain MRI model: %s", exc, exc_info=True)

    try:
        from validator import get_mri_validator

        logger.info("Initializing Brain MRI input validator...")
        _validator = get_mri_validator(_device)
        logger.info(
            "Brain MRI validator initialized (ready=%s): %s",
            _validator.is_ready,
            _validator.ready_reason,
        )
    except Exception as exc:
        logger.error("Failed to initialize Brain MRI validator: %s", exc, exc_info=True)


@app.get("/health")
async def health():
    """Return service, MRI model, and MRI validator health status."""
    validator_ready = bool(_validator is not None and _validator.is_ready)
    return {
        "status": "ok" if (_model_loaded and validator_ready) else "degraded",
        "modality": "brain_mri",
        "model_loaded": _model_loaded,
        "model_name": _model_identifier if _model_loaded else None,
        "classes": ["glioma", "meningioma", "pituitary", "no_tumor"],
        "validator_loaded": _validator is not None,
        "validator_ready": validator_ready,
        "validator_status": _validator.ready_reason if _validator else "Not initialized",
        "validator_report": _validator.report_summary if (_validator and _validator.is_ready) else None,
        "checkpoint_path": CHECKPOINT_PATH,
        "device": str(_device) if _device is not None else None,
        "error": _load_error if not _model_loaded else None,
    }


async def _decode_uploaded_image(file: UploadFile):
    content_type = (file.content_type or "").lower().split(";")[0].strip()
    if content_type not in ALLOWED_MIME_TYPES:
        return None, {
            "is_valid": False,
            "image_type": "UNSUPPORTED_FORMAT",
            "validation_status": "REJECTED",
            "reason": f"Invalid file type '{content_type}'. Accepted types: image/jpeg, image/png.",
            "message": "Invalid Upload — Please upload a Brain MRI image in JPG, JPEG, or PNG format.",
            "score": 0.0,
        }

    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_pil.verify()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image_pil, None
    except Exception as exc:
        return None, {
            "is_valid": False,
            "image_type": "CORRUPTED_FILE",
            "validation_status": "REJECTED",
            "reason": f"Could not decode image: {exc}",
            "message": "Invalid Upload — Image file is corrupted or cannot be decoded. Please upload a valid Brain MRI file.",
            "score": 0.0,
        }


@app.post("/validate")
async def validate_endpoint(file: UploadFile = File(...)):
    """Validate whether uploaded image is a Brain MRI without calling tumour classifier."""
    image_pil, err_validation = await _decode_uploaded_image(file)
    if err_validation is not None:
        return JSONResponse(status_code=422, content=err_validation)

    from validator import validate_brain_mri

    validation = validate_brain_mri(image_pil, device=_device)
    status_code = 200 if validation.get("is_valid", False) else 422
    return JSONResponse(status_code=status_code, content=validation)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Validate Brain MRI input first, fail closed on rejection/uncertainty/unready validator,
    and run 4-class ResNet-18 + Grad-CAM classification only when validated.
    """
    if not _model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "error": f"Brain MRI model is not loaded. {_load_error}".strip(),
            },
        )

    image_pil, err_validation = await _decode_uploaded_image(file)
    if err_validation is not None:
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": err_validation["message"],
                "validation": err_validation,
            },
        )

    from validator import validate_brain_mri

    validation = validate_brain_mri(image_pil, device=_device)
    logger.info(
        "MRI Validation check: status=%s, type=%s, score=%.4f",
        validation.get("validation_status"),
        validation.get("image_type"),
        validation.get("score", 0.0),
    )

    if not validation.get("is_valid", False):
        logger.warning(
            "MRI predict rejected non-MRI/unverified image: status=%s, type=%s, reason=%s",
            validation.get("validation_status"),
            validation.get("image_type"),
            validation.get("reason"),
        )
        return JSONResponse(
            status_code=422,
            content={
                "success": False,
                "error": validation.get("message", "Brain MRI validation failed."),
                "validation": validation,
            },
        )

    uid = uuid4().hex[:12]
    heatmap_filename = f"mri_{uid}_heatmap.png"
    overlay_filename = f"mri_{uid}_overlay.png"
    heatmap_path = str(Path(RESULTS_DIR) / heatmap_filename)
    overlay_path = str(Path(RESULTS_DIR) / overlay_filename)

    try:
        from mri_inference import run_mri_inference

        t0 = time.perf_counter()
        result = run_mri_inference(
            model=_model,
            gradcam=_gradcam,
            device=_device,
            raw_class_names=_class_names,
            image_pil=image_pil,
            heatmap_save_path=heatmap_path,
            overlay_save_path=overlay_path,
        )
        processing_time = round(time.perf_counter() - t0, 4)
    except Exception as exc:
        logger.error("MRI inference failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(exc)},
        )

    return {
        "success": True,
        "modality": "brain_mri",
        "id": uid,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "heatmap_path": heatmap_filename if result["heatmap_saved"] else None,
        "overlay_path": overlay_filename if result["overlay_saved"] else None,
        "processing_time": processing_time,
        "model_name": _model_identifier,
        "validation": validation,
    }

