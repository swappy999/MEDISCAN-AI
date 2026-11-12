"""
benchmark_pipeline.py
=====================
Comprehensive Evaluation and Benchmarking Suite for MediScan AI
Strict Chest X-Ray Validation and Out-Of-Distribution (OOD) Rejection System.

Tests all 12 benchmark test cases directly through FastAPI endpoints:
- POST /validate
- POST /predict
"""

import json
import sys
from pathlib import Path

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add ml-service to path
ml_service_dir = Path(__file__).resolve().parent / "ml-service"
sys.path.insert(0, str(ml_service_dir))

import app
from fastapi.responses import JSONResponse

def run_coro(coro):
    """Executes a coroutine that completes synchronously without async loop blocking."""
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("Coroutine did not finish synchronously!")

class MockUploadFile:
    def __init__(self, filename: str, content: bytes, content_type: str):
        self.filename = filename
        self.content = content
        self.content_type = content_type

    async def read(self) -> bytes:
        return self.content

TEST_BASE = Path(__file__).resolve().parent / "image-validation-dataset" / "test"

BENCHMARK_CASES = [
    # (Label, Relative Path, Expected Valid, Expected Image Type)
    ("1. Normal CXR", "CHEST_XRAY/01_normal_chest_xray.jpeg", True, "CHEST_XRAY"),
    ("2. Pneumonia CXR", "CHEST_XRAY/02_pneumonia_chest_xray.jpeg", True, "CHEST_XRAY"),
    ("3. Hand X-ray", "NON_CHEST/03_hand_xray.png", False, "HAND_XRAY"),
    ("4. Wrist X-ray", "NON_CHEST/04_wrist_xray.png", False, "HAND_XRAY"),
    ("5. Foot X-ray", "NON_CHEST/05_foot_xray.png", False, "HAND_XRAY"),
    ("6. Knee X-ray", "NON_CHEST/06_knee_xray.png", False, "HAND_XRAY"),
    ("7. Dental X-ray", "NON_CHEST/07_dental_xray.png", False, "DENTAL_XRAY"),
    ("8. Brain MRI", "NON_CHEST/08_brain_mri.png", False, "BRAIN_MRI"),
    ("9. CT Scan", "NON_CHEST/09_ct_scan.png", False, "CT_SCAN"),
    ("10. Photograph", "NON_CHEST/10_photograph.jpg", False, "ORDINARY_PHOTOGRAPH"),
    ("11. Medical Report", "NON_CHEST/11_medical_report.png", False, "DOCUMENT_OR_REPORT"),
    ("12. Corrupted Image", "NON_CHEST/12_corrupted_image.jpg", False, "CORRUPTED_FILE"),
]

def parse_response(res):
    if isinstance(res, JSONResponse):
        status_code = res.status_code
        body = json.loads(res.body.decode())
    elif isinstance(res, dict):
        status_code = 200
        body = res
    else:
        status_code = 500
        body = {"error": str(res)}
    return status_code, body

def run_benchmark():
    print("=" * 90)
    print("          MEDISCAN AI — STRICT CHEST X-RAY VALIDATION BENCHMARK SUITE")
    print("=" * 90)
    print("Initializing ML service and preloading PyTorch models...")
    run_coro(app.startup_event())
    print("Models and dedicated CXR validator initialized successfully.\n")

    results = []

    print("-" * 90)
    print(f"{'#':<3} {'Test Case':<20} | {'Expected':<8} | {'Val Status':<10} | {'Pred Status':<11} | {'Result':<8}")
    print("-" * 90)

    for i, (label, rel_path, expected_valid, exp_type) in enumerate(BENCHMARK_CASES, 1):
        file_path = TEST_BASE / rel_path
        if not file_path.exists():
            print(f"ERROR: Missing test file {file_path}")
            continue

        filename = file_path.name
        mime = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else "image/png"

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # 1. Test POST /validate endpoint
        mock_file_val = MockUploadFile(filename, file_bytes, mime)
        val_res = run_coro(app.validate_endpoint(mock_file_val))
        val_status, val_body = parse_response(val_res)

        # 2. Test POST /predict endpoint
        mock_file_pred = MockUploadFile(filename, file_bytes, mime)
        pred_res = run_coro(app.predict(mock_file_pred))
        pred_status, pred_body = parse_response(pred_res)

        # Evaluation
        is_accepted = (val_status == 200 and pred_status == 200)
        passed = (is_accepted == expected_valid)

        val_data = val_body.get("validation", val_body)
        detected_type = val_data.get("image_type", "UNKNOWN")
        cos_score = val_data.get("score")

        # Crucial Safety Verification: Ensure pneumonia model was never executed on rejected images
        safety_passed = True
        if not is_accepted:
            if "prediction" in pred_body or "heatmap_path" in pred_body or "overlay_path" in pred_body:
                safety_passed = False

        status_text = "ACCEPTED" if is_accepted else f"REJECTED({val_status})"
        pred_text = f"HTTP {pred_status}"
        result_tag = "PASS" if (passed and safety_passed) else "FAIL"

        results.append({
            "index": i,
            "label": label,
            "filename": filename,
            "expected_valid": expected_valid,
            "is_accepted": is_accepted,
            "passed": passed and safety_passed,
            "val_status": val_status,
            "pred_status": pred_status,
            "detected_type": detected_type,
            "score": cos_score,
            "val_body": val_body,
            "pred_body": pred_body,
            "safety_passed": safety_passed,
        })

        print(f"{i:<3} {label:<20} | {'ACCEPT' if expected_valid else 'REJECT':<8} | {status_text:<10} | {pred_text:<11} | [{result_tag}]")

    print("-" * 90)
    print("\n" + "=" * 90)
    print("                      DETAILED AUDIT TRAIL PER TEST CASE")
    print("=" * 90)

    for r in results:
        print(f"\n[CASE {r['index']}] {r['label']} ({r['filename']})")
        print(f"  Classification Stage 1 (Validation Endpoint):")
        print(f"    - HTTP Status: {r['val_status']} | Validation Status: {r['val_body'].get('validation_status')}")
        print(f"    - Detected Image Type: {r['detected_type']}")
        print(f"    - CXR Cosine Similarity Score: {r['score']} (Calibrated Threshold >= 0.7824)")
        print(f"    - Diagnostic Reason: {r['val_body'].get('reason')}")
        print(f"  Inference Stage 2 (Predict Endpoint Gating):")
        print(f"    - HTTP Status: {r['pred_status']}")
        if r['is_accepted']:
            print(f"    - Pneumonia Inference: EXECUTED")
            print(f"    - Diagnosis: {r['pred_body'].get('prediction')} (Confidence: {r['pred_body'].get('confidence'):.1%})")
            print(f"    - Probabilities: {r['pred_body'].get('probabilities')}")
            print(f"    - Grad-CAM Heatmap: {r['pred_body'].get('heatmap_path')}")
            print(f"    - Overlay Path: {r['pred_body'].get('overlay_path')}")
        else:
            print(f"    - Pneumonia Inference: BLOCKED (Pre-inference validation guard enforced)")
            print(f"    - User Message: \"{r['pred_body'].get('error')}\"")
            print(f"    - Safety Verification (No prediction or heatmap leaked): {'VERIFIED' if r['safety_passed'] else 'FAILED'}")

    # Compute Final Performance Metrics
    total = len(results)
    total_pos = sum(1 for r in results if r["expected_valid"])
    total_neg = sum(1 for r in results if not r["expected_valid"])

    tp = sum(1 for r in results if r["expected_valid"] and r["is_accepted"])
    tn = sum(1 for r in results if not r["expected_valid"] and not r["is_accepted"])
    fp = sum(1 for r in results if not r["expected_valid"] and r["is_accepted"])
    fn = sum(1 for r in results if r["expected_valid"] and not r["is_accepted"])

    far = (fp / total_neg) * 100 if total_neg else 0.0
    frr = (fn / total_pos) * 100 if total_pos else 0.0
    accuracy = ((tp + tn) / total) * 100
    sensitivity = (tp / total_pos) * 100 if total_pos else 0.0
    specificity = (tn / total_neg) * 100 if total_neg else 0.0

    print("\n" + "=" * 90)
    print("                     FINAL PERFORMANCE METRICS SUMMARY")
    print("=" * 90)
    print(f" Total Evaluation Images Tested : {total}")
    print(f" Authentic Chest X-Rays (Pos)   : {total_pos}  (True Positives: {tp}, False Negatives: {fn})")
    print(f" Non-Chest / OOD Images (Neg)   : {total_neg} (True Negatives: {tn}, False Positives: {fp})")
    print("-" * 90)
    print(f" False Acceptance Rate (FAR)    : {far:.2f}%  (Target: 0.0% for patient safety)")
    print(f" False Rejection Rate (FRR)     : {frr:.2f}%")
    print(f" Validation Sensitivity / TPR   : {sensitivity:.2f}%")
    print(f" Validation Specificity / TNR   : {specificity:.2f}%")
    print(f" Overall Validation Accuracy    : {accuracy:.2f}%")
    print("=" * 90)

if __name__ == "__main__":
    run_benchmark()
