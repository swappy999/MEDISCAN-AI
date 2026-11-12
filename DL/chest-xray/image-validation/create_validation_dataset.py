"""
create_validation_dataset.py
----------------------------
Sets up the dataset structure for the Chest X-Ray Validation and Rejection system:
image-validation-dataset/
  ├── train/
  │   ├── CHEST_XRAY/
  │   └── NON_CHEST/
  ├── validation/
  │   ├── CHEST_XRAY/
  │   └── NON_CHEST/
  └── test/
      ├── CHEST_XRAY/
      └── NON_CHEST/

Also generates calibrated benchmark test cases for the 12 evaluation categories:
1. Normal frontal chest X-ray
2. Pneumonia frontal chest X-ray
3. Hand X-ray
4. Wrist X-ray
5. Foot X-ray
6. Knee X-ray
7. Dental X-ray
8. Brain MRI
9. CT scan
10. Ordinary photograph
11. Medical report screenshot
12. Corrupted image
"""

import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"

def setup_directories():
    splits = ["train", "validation", "test"]
    categories = ["CHEST_XRAY", "NON_CHEST"]
    for split in splits:
        for cat in categories:
            d = DATASET_DIR / split / cat
            d.mkdir(parents=True, exist_ok=True)
            (d / ".gitkeep").touch()

    # Create dataset README
    readme_content = """# MediScan AI — Image Validation Dataset (Chest vs Non-Chest)

This dataset is designed to train and calibrate the Chest X-Ray Validation & Out-Of-Distribution (OOD) filter.
Only frontal chest radiographs (PA/AP) must be admitted to CHEST_XRAY.

## Directory Structure
- `train/`
  - `CHEST_XRAY/`: Frontal chest X-rays (Normal, Pneumonia, Infiltration, Effusion)
  - `NON_CHEST/`: Radiographs of other anatomical regions (Hand, Wrist, Spine, Skull, Pelvis, Dental, Extremities), other modalities (MRI, CT, Ultrasound), documents, and natural images.
- `validation/`: Held-out validation partition for calibration of rejection thresholds (Cosine distance, Mahalanobis distance, morphological bounds).
- `test/`: Independent evaluation set across all 12 test modalities.

## Strict Labeling & Partitioning Guidelines
1. No Patient-Level Leakage: Ensure images from the same patient or hospital scan series never span train and test/validation sets.
2. True Grayscale != Chest X-ray: Extremity radiographs, dental scans, skull X-rays and MRI/CT slices are also monochrome radiographs; they MUST be labeled as `NON_CHEST`.
3. Frontal View Only: Lateral chest X-rays or decubitus views should be categorized in NON_CHEST unless lateral projection is explicitly modeled.
"""
    with open(DATASET_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    print("Dataset directory structure created at:", DATASET_DIR)


def populate_test_set():
    test_chest_dir = DATASET_DIR / "test" / "CHEST_XRAY"
    test_nonchest_dir = DATASET_DIR / "test" / "NON_CHEST"
    
    # 1. Normal Frontal Chest X-ray (Real)
    real_normal = ROOT.parent / "dataset" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
    if not real_normal.exists():
        real_normal = ROOT.parent / "data" / "raw" / "test" / "NORMAL" / "IM-0001-0001.jpeg"
    if real_normal.exists():
        shutil.copy(real_normal, test_chest_dir / "01_normal_chest_xray.jpeg")
        print("  [Copied] 01_normal_chest_xray.jpeg")
    
    # 2. Pneumonia Frontal Chest X-ray (Real)
    real_pneumonia = ROOT.parent / "dataset" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"
    if not real_pneumonia.exists():
        real_pneumonia = ROOT.parent / "data" / "raw" / "test" / "PNEUMONIA" / "person100_bacteria_475.jpeg"
    if real_pneumonia.exists():
        shutil.copy(real_pneumonia, test_chest_dir / "02_pneumonia_chest_xray.jpeg")
        print("  [Copied] 02_pneumonia_chest_xray.jpeg")

    # 3. Hand X-ray
    # Distinct bone density: 5 elongated rays (phalanges & metacarpals), dark air background
    img = np.zeros((512, 512), dtype=np.uint8) + 12 # dark radiograph background
    # Soft tissue contour
    cv2.ellipse(img, (256, 320), (140, 160), 0, 0, 360, 45, -1)
    # 5 fingers (phalanges)
    for i, angle in enumerate([-35, -18, 0, 18, 35]):
        rad = np.radians(angle)
        length = 160 if i == 2 else (145 if i in [1, 3] else 110)
        x1, y1 = int(256 + 60 * np.sin(rad)), int(260 - 60 * np.cos(rad))
        x2, y2 = int(256 + length * np.sin(rad)), int(260 - length * np.cos(rad))
        cv2.line(img, (x1, y1), (x2, y2), 180, 18)
        # Joints
        cv2.circle(img, (int((x1+x2)/2), int((y1+y2)/2)), 14, 210, -1)
    # Metacarpals & wrist carpal cluster
    cv2.rectangle(img, (180, 320), (332, 420), 160, -1)
    cv2.circle(img, (256, 440), 45, 190, -1)
    cv2.GaussianBlur(img, (9, 9), 0, dst=img)
    cv2.putText(img, "R HAND PA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 2)
    Image.fromarray(img).save(test_nonchest_dir / "03_hand_xray.png")
    print("  [Generated] 03_hand_xray.png")

    # 4. Wrist X-ray
    # Distal radius and ulna, carpal bones, narrow aspect ratio
    img = np.zeros((512, 512), dtype=np.uint8) + 10
    # Soft tissue forearm
    cv2.rectangle(img, (160, 200), (352, 512), 40, -1)
    # Radius & Ulna bones
    cv2.rectangle(img, (180, 260), (245, 512), 175, -1)
    cv2.rectangle(img, (265, 260), (330, 512), 165, -1)
    # Carpal bones cluster
    cv2.ellipse(img, (256, 220), (80, 35), 0, 0, 360, 195, -1)
    # Metacarpal bases
    for x in [190, 220, 256, 290, 320]:
        cv2.line(img, (x, 185), (x, 110), 170, 14)
    cv2.GaussianBlur(img, (7, 7), 0, dst=img)
    cv2.putText(img, "L WRIST LAT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 2)
    Image.fromarray(img).save(test_nonchest_dir / "04_wrist_xray.png")
    print("  [Generated] 04_wrist_xray.png")

    # 5. Foot X-ray
    # Oblique foot projection: calcaneus, tarsals, metatarsals
    img = np.zeros((512, 512), dtype=np.uint8) + 12
    # Soft tissue foot profile
    pts = np.array([[100, 380], [120, 240], [280, 200], [420, 320], [460, 380], [100, 400]], np.int32)
    cv2.fillPoly(img, [pts], 45)
    # Calcaneus & Talus
    cv2.circle(img, (160, 320), 55, 180, -1)
    cv2.circle(img, (220, 260), 45, 195, -1)
    # Metatarsals fan
    for y_end in [280, 310, 340, 370]:
        cv2.line(img, (270, 270), (410, y_end), 175, 14)
    cv2.GaussianBlur(img, (9, 9), 0, dst=img)
    cv2.putText(img, "FOOT AP/OBL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 2)
    Image.fromarray(img).save(test_nonchest_dir / "05_foot_xray.png")
    print("  [Generated] 05_foot_xray.png")

    # 6. Knee X-ray
    # Distal femur, joint space, proximal tibia & fibula
    img = np.zeros((512, 512), dtype=np.uint8) + 10
    # Soft tissue leg column
    cv2.rectangle(img, (140, 0), (372, 512), 40, -1)
    # Femur condyles
    cv2.rectangle(img, (180, 0), (332, 230), 180, -1)
    cv2.circle(img, (210, 230), 35, 195, -1)
    cv2.circle(img, (300, 230), 35, 195, -1)
    # Joint space gap (black strip)
    # Tibial plateau
    cv2.rectangle(img, (180, 275), (332, 512), 175, -1)
    # Patella
    cv2.ellipse(img, (256, 210), (35, 45), 0, 0, 360, 215, -1)
    cv2.GaussianBlur(img, (9, 9), 0, dst=img)
    cv2.putText(img, "R KNEE AP ERECT", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 120, 2)
    Image.fromarray(img).save(test_nonchest_dir / "06_knee_xray.png")
    print("  [Generated] 06_knee_xray.png")

    # 7. Dental Panoramic / Bitewing X-ray
    # Distinct rectangular teeth grid, roots, mandibular bone
    img = np.zeros((400, 700), dtype=np.uint8) + 15
    # Maxillary arch & Mandibular arch
    for row_y in [140, 230]:
        for tooth_x in range(120, 580, 32):
            cv2.rectangle(img, (tooth_x, row_y), (tooth_x + 24, row_y + 40), 220, -1)
            # Root
            root_dir = -1 if row_y == 140 else 1
            cv2.line(img, (tooth_x + 12, row_y + 20), (tooth_x + 12, row_y + 20 + root_dir * 55), 180, 8)
    cv2.ellipse(img, (350, 340), (280, 60), 0, 0, 180, 110, 22)
    cv2.GaussianBlur(img, (7, 7), 0, dst=img)
    cv2.putText(img, "PANORAMIC DENTAL RADIOGRAPH", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 140, 2)
    Image.fromarray(img).save(test_nonchest_dir / "07_dental_xray.png")
    print("  [Generated] 07_dental_xray.png")

    # 8. Brain MRI (Axial T2)
    # Oval cranium, bilateral cerebral hemispheres, ventricles, sulci
    img = np.zeros((512, 512), dtype=np.uint8) + 5 # black MRI background
    # Outer skull ring
    cv2.ellipse(img, (256, 256), (180, 215), 0, 0, 360, 160, 6) # skull bone
    # Brain tissue parenchyma
    cv2.ellipse(img, (256, 256), (170, 205), 0, 0, 360, 85, -1)
    # Longitudinal fissure
    cv2.line(img, (256, 60), (256, 450), 30, 3)
    # Lateral ventricles (butterfly / horn shapes, hyperintense CSF in T2)
    cv2.ellipse(img, (230, 240), (18, 55), 15, 0, 360, 230, -1)
    cv2.ellipse(img, (282, 240), (18, 55), -15, 0, 360, 230, -1)
    # Cortical sulcal convolutions
    for angle in range(0, 360, 24):
        rad = np.radians(angle)
        cx, cy = int(256 + 130 * np.cos(rad)), int(256 + 160 * np.sin(rad))
        cv2.circle(img, (cx, cy), 16, 50, -1)
    cv2.GaussianBlur(img, (9, 9), 0, dst=img)
    cv2.putText(img, "AXIAL T2 FSE BRAIN MRI", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 160, 2)
    Image.fromarray(img).save(test_nonchest_dir / "08_brain_mri.png")
    print("  [Generated] 08_brain_mri.png")

    # 9. CT Scan (Axial cross-section with circular gantry)
    img = np.zeros((512, 512), dtype=np.uint8) + 8
    # Circular scanner gantry field of view
    cv2.circle(img, (256, 256), 235, 30, -1)
    # Body contour (Abdomen/Pelvis slice)
    cv2.ellipse(img, (256, 256), (185, 140), 0, 0, 360, 75, -1)
    # Spine vertebra at posterior
    cv2.circle(img, (256, 350), 28, 240, -1)
    cv2.circle(img, (256, 350), 10, 15, -1) # spinal canal
    # Liver / Spleen / Aorta
    cv2.ellipse(img, (180, 240), (60, 45), 0, 0, 360, 105, -1) # liver
    cv2.ellipse(img, (330, 240), (35, 30), 0, 0, 360, 95, -1)  # spleen
    cv2.circle(img, (245, 305), 14, 210, -1) # aorta
    cv2.GaussianBlur(img, (7, 7), 0, dst=img)
    cv2.putText(img, "CT ABDOMEN C+ 5mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 180, 2)
    Image.fromarray(img).save(test_nonchest_dir / "09_ct_scan.png")
    print("  [Generated] 09_ct_scan.png")

    # 10. Ordinary Photograph (Full color with high saturation)
    photo = np.zeros((512, 512, 3), dtype=np.uint8)
    # Blue sky top half
    photo[:256, :] = [235, 160, 70] # BGR sky blue
    # Green landscape bottom half
    photo[256:, :] = [45, 165, 35]  # BGR grass green
    # Yellow sun
    cv2.circle(photo, (100, 100), 50, [30, 225, 255], -1)
    cv2.putText(photo, "Outdoor Scene Photo", (150, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, [255, 255, 255], 2)
    Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)).save(test_nonchest_dir / "10_photograph.jpg")
    print("  [Generated] 10_photograph.jpg")

    # 11. Medical Report Screenshot / Document
    doc = np.ones((700, 500, 3), dtype=np.uint8) * 255 # clean white paper
    cv2.rectangle(doc, (30, 30), (470, 670), (220, 220, 220), 1)
    cv2.putText(doc, "METROPOLITAN HOSPITAL", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 40, 80), 2)
    cv2.putText(doc, "DEPARTMENT OF RADIOLOGY REPORT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)
    cv2.line(doc, (50, 115), (450, 115), (180, 180, 180), 1)
    lines = [
        "Patient: Doe, John",
        "MRN: 98452109   DOB: 12/04/1978",
        "Exam: Routine Diagnostic Evaluation",
        "Date of Service: 11/11/2026",
        "",
        "CLINICAL INDICATION:",
        "Evaluation of chronic symptoms.",
        "",
        "FINDINGS:",
        "Lung parenchyma clear without focal consolidation.",
        "Cardiothoracic ratio within normal physiological limits.",
        "No pleural effusion or pneumothorax identified.",
        "Bony thorax intact without acute fracture.",
        "",
        "IMPRESSION:",
        "Unremarkable preliminary study.",
    ]
    y = 150
    for line in lines:
        is_bold = line.endswith(":") or "HOSPITAL" in line
        cv2.putText(doc, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2 if is_bold else 1)
        y += 26
    Image.fromarray(doc).save(test_nonchest_dir / "11_medical_report.png")
    print("  [Generated] 11_medical_report.png")

    # 12. Corrupted Image File
    with open(test_nonchest_dir / "12_corrupted_image.jpg", "wb") as f:
        f.write(b"CORRUPTED_NON_IMAGE_DATA_HEADER_1234567890\x00\xFF\xAA\xBB\xCC\xDD\xEE\xFF" * 100)
    print("  [Generated] 12_corrupted_image.jpg")

    print("\nAll 12 test benchmark images generated successfully in image-validation-dataset/test/!")

if __name__ == "__main__":
    setup_directories()
    populate_test_set()

