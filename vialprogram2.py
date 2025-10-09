import cv2
import numpy as np
import os
from datetime import datetime
import sys
import time
# =============================
# CONFIG
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROI = (2280, 3632, 1970, 5390)  # (y1, y2, x1, x2) for vial ROI

INPUT_DIR = os.path.join(BASE_DIR, "image_process_input")
OUTPUT_DIR = os.path.join(BASE_DIR, "image_process_output")
LOG_FILE = os.path.join(OUTPUT_DIR, "phase_analysis_log.txt")

# Make sure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Thresholds
BRIGHTNESS_THRESHOLD = 19.0   # above this = "high" brightness
# =============================


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def crop_roi(img, roi):
    """Crop region of interest from image"""
    y1, y2, x1, x2 = roi
    return img[y1:y2, x1:x2]


def analyze_brightness(img):
    """Return average brightness (grayscale mean) of ROI"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def classify(b_before, b_after, vial_id):
    """Classify outcome based on brightness thresholds"""
    before_state = "HIGH" if b_before > BRIGHTNESS_THRESHOLD else "LOW"
    after_state = "HIGH" if b_after > BRIGHTNESS_THRESHOLD else "LOW"

    if before_state == "LOW" and after_state == "HIGH":
        return "Phase Separation"
    elif before_state == "LOW" and after_state == "LOW":
        return "Soluable or Nano Emulsion"
    elif before_state == "HIGH" and after_state == "HIGH":
        return "Stable Emulsion"
    else:
        return "Unclassified: Possible Error"


def annotate_image(img, brightness, label, roi):
    """Draw ROI and brightness label on the image"""
    y1, y2, x1, x2 = roi
    annotated = img.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"{label}: {brightness:.1f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    return annotated

def compare_vials(before_img_path, after_img_path, vial_id="Unknown"):
    """Compare brightness of before vs after vial images"""
    img_before = cv2.imread(before_img_path)
    img_after = cv2.imread(after_img_path)

    if img_before is None or img_after is None:
        print(f"❌ Could not load images for {vial_id}, logging FAILED.")

        ensure_dir(OUTPUT_DIR)
        with open(LOG_FILE, "a") as log:
            log.write(f"{datetime.now()} - Vial {vial_id}: Before=NA, After=NA, Result=FAILED (missing image)\n")
        return "FAILED"

    roi_before = crop_roi(img_before, ROI)
    roi_after = crop_roi(img_after, ROI)

    b_brightness = analyze_brightness(roi_before)
    a_brightness = analyze_brightness(roi_after)

    result = classify(b_brightness, a_brightness, vial_id)

    # Annotate images
    ann_before = annotate_image(img_before, b_brightness, "Before", ROI)
    ann_after = annotate_image(img_after, a_brightness, "After", ROI)

    # Save outputs
    ensure_dir(OUTPUT_DIR)
    out_before = os.path.join(OUTPUT_DIR, f"{vial_id}_before_{result}.jpg")
    out_after = os.path.join(OUTPUT_DIR, f"{vial_id}_after_{result}.jpg")
    ok1 = cv2.imwrite(out_before, ann_before)
    ok2 = cv2.imwrite(out_after, ann_after)
    # print(f"Saved before={ok1}, after={ok2} → {OUTPUT_DIR}")

    # Log
    with open(LOG_FILE, "a") as log:
        log.write(f"{datetime.now()} - Vial {vial_id}: "
                  f"Before={b_brightness:.1f}, After={a_brightness:.1f}, "
                  f"Result={result}\n")

    print(f"Vial {vial_id}: {result}")
    print(f"  Before={b_brightness:.1f}, After={a_brightness:.1f}")

    return result


def batch_process(vial_label=None):
    """Process exactly two images in folder. If not exactly two, log FAILED."""
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))])

    # Use passed-in vial label if provided, else fallback to A1
    vial_id = vial_label if vial_label else "A1"

    if len(files) != 2:
        print(f"❌ Expected 2 images, found {len(files)}. Logging FAILED for {vial_id}.")

        for f in files:
            os.remove(os.path.join(INPUT_DIR, f))

        ensure_dir(OUTPUT_DIR)
        with open(LOG_FILE, "a") as log:
            log.write(f"{datetime.now()} - Vial {vial_id}: Before=NA, After=NA, Result=FAILED (wrong count)\n")
        return

    before_path = os.path.join(INPUT_DIR, files[0])
    after_path = os.path.join(INPUT_DIR, files[1])

    compare_vials(before_path, after_path, vial_id=vial_id)
    time.sleep(2) 
    os.remove(before_path)
    os.remove(after_path)


if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)

    # Get vial label from OPC UA (server passes as sys.argv[1])
    VIAL_LABEL = None
    if len(sys.argv) > 1:
        VIAL_LABEL = sys.argv[1]

    print("Starting Phase Separation Analysis...")
    batch_process(vial_label=VIAL_LABEL)
    print("Analysis complete. Results + log saved in", OUTPUT_DIR)
