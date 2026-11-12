"""
verify_validator_pipeline.py
----------------------------
Runs the standalone validator on the 12 evaluation cases and reports metrics.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "ml-service"))

from PIL import Image
from validator import validate_chest_xray, ChestXrayValidator
import torch
test_dir = ROOT / "image-validation-dataset/test"

test_cases = [
    ("1. Normal CXR", test_dir / "CHEST_XRAY/01_normal_chest_xray.jpeg", True),
    ("2. Pneumonia CXR", test_dir / "CHEST_XRAY/02_pneumonia_chest_xray.jpeg", True),
    ("3. Hand X-ray", test_dir / "NON_CHEST/03_hand_xray.png", False),
    ("4. Wrist X-ray", test_dir / "NON_CHEST/04_wrist_xray.png", False),
    ("5. Foot X-ray", test_dir / "NON_CHEST/05_foot_xray.png", False),
    ("6. Knee X-ray", test_dir / "NON_CHEST/06_knee_xray.png", False),
    ("7. Dental X-ray", test_dir / "NON_CHEST/07_dental_xray.png", False),
    ("8. Brain MRI", test_dir / "NON_CHEST/08_brain_mri.png", False),
    ("9. CT Scan", test_dir / "NON_CHEST/09_ct_scan.png", False),
    ("10. Photograph", test_dir / "NON_CHEST/10_photograph.jpg", False),
    ("11. Medical Report", test_dir / "NON_CHEST/11_medical_report.png", False),
    ("12. Corrupted Image", test_dir / "NON_CHEST/12_corrupted_image.jpg", False),
]

tp = fp = tn = fn = 0

print("=" * 80)
print(f"{'Test Case':<20} | {'Expected':<8} | {'Status':<10} | {'Type':<18} | {'Pass/Fail':<10}")
print("=" * 80)

for name, path, expected_valid in test_cases:
    is_corrupt = False
    try:
        pil_img = Image.open(path)
        pil_img.verify()
        pil_img = Image.open(path).convert("RGB")
    except Exception as e:
        is_corrupt = True

    if is_corrupt:
        res = {
            "is_valid": False,
            "image_type": "CORRUPTED_FILE",
            "validation_status": "REJECTED",
            "message": "Invalid Upload — Image file is corrupted or cannot be decoded.",
            "score": 0.0
        }
    else:
        res = validate_chest_xray(pil_img)

    actual_valid = res["is_valid"]
    correct = (actual_valid == expected_valid)
    
    if expected_valid and actual_valid:
        tp += 1
    elif not expected_valid and not actual_valid:
        tn += 1
    elif not expected_valid and actual_valid:
        fp += 1
    elif expected_valid and not actual_valid:
        fn += 1

    status_str = "ACCEPTED" if actual_valid else "REJECTED"
    pass_str = "PASS" if correct else "FAIL"
    print(f"{name:<20} | {str(expected_valid):<8} | {status_str:<10} | {res['image_type']:<18} | {pass_str:<10}")

print("=" * 80)

total = len(test_cases)
accuracy = (tp + tn) / total
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # recall for chest X-ray
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # rejection of non-chest
far = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # false acceptance rate

print(f"\nVALIDATION METRICS SUMMARY:")
print(f"  Total Test Cases:       {total}")
print(f"  True Positives (CXR):   {tp}")
print(f"  True Negatives (OOD):   {tn}")
print(f"  False Positives (FAR):  {fp}")
print(f"  False Negatives:        {fn}")
print(f"  Accuracy:               {accuracy * 100:.1f}%")
print(f"  Sensitivity (Recall):   {sensitivity * 100:.1f}%")
print(f"  Specificity (OOD Rej):  {specificity * 100:.1f}%")
print(f"  False Acceptance Rate:  {far * 100:.1f}%")
print("=" * 80)

