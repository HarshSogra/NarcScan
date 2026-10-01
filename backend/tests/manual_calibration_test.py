"""
manual_calibration_test.py
==========================

Phase 4 - Manual Grey Reference Calibration Test

Run this script to visually verify that the grey-reference calibration
module works correctly on a real phone photograph of the NarcScan card.

The script runs the full pipeline up to and including Phase 4:
    ArUco detection -> Perspective Correction -> Grey Calibration

Usage
-----
From the backend/ folder with the virtual environment active:

    python tests/manual_calibration_test.py ../test_data/test_data2.png

Output
------
Three images are saved to backend/tests/:
    calibration_input.png    -- the perspective-corrected card (Phase 3 output)
    calibration_roi.png      -- the corrected card with the grey ROI highlighted
    calibration_output.png   -- the colour-cast-corrected card (Phase 4 output)

Open these files in an image viewer to verify:
  - calibration_roi.png   : the green rectangle should sit in the central
                            grey region, well away from the ArUco markers.
  - calibration_output.png: any warm/cool colour cast should be reduced
                            compared to calibration_input.png.
"""

import sys
import os
import cv2
import numpy as np

# Allow 'app' to be found when running this script from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers
from app.cv.perspective import correct_perspective
from app.cv.calibration import (
    sample_grey_reference,
    apply_grey_correction,
    GREY_ROI_LEFT_FRAC,
    GREY_ROI_RIGHT_FRAC,
    GREY_ROI_TOP_FRAC,
    GREY_ROI_BOTTOM_FRAC,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
TESTS_DIR          = os.path.dirname(__file__)
INPUT_SAVE_PATH    = os.path.join(TESTS_DIR, "calibration_input.png")
ROI_SAVE_PATH      = os.path.join(TESTS_DIR, "calibration_roi.png")
OUTPUT_SAVE_PATH   = os.path.join(TESTS_DIR, "calibration_output.png")


def main():
    # ------------------------------------------------------------------
    # Parse argument
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print()
        print("Usage: python tests/manual_calibration_test.py <image_path>")
        print()
        print("Example:")
        print("  python tests/manual_calibration_test.py ../test_data/test_data2.png")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.isfile(image_path):
        print()
        print(f"ERROR: Image file not found: {image_path}")
        print("Make sure the file exists and the path is correct.")
        sys.exit(1)

    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: OpenCV could not read the image at: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"  Input size : {w} x {h} pixels")

    # ------------------------------------------------------------------
    # Phase 2: ArUco detection
    # ------------------------------------------------------------------
    print("\n--- Phase 2: ArUco Detection ---")
    detection = detect_markers(image)
    detected_ids = sorted(detection["markers"].keys())
    print(f"  Detected marker IDs : {detected_ids}")
    print(f"  Missing marker IDs  : {detection['missing_ids']}")

    if detection["missing_ids"]:
        print("\nERROR: Not all four markers detected.  Cannot proceed.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: Perspective correction
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Perspective Correction ---")
    p_result = correct_perspective(image, detection)
    if not p_result["success"]:
        print(f"  ERROR: Perspective correction failed: {p_result['reason']}")
        sys.exit(1)

    corrected = p_result["warped_image"]
    ph, pw = corrected.shape[:2]
    print(f"  Corrected image size : {pw} x {ph} pixels")
    cv2.imwrite(INPUT_SAVE_PATH, corrected)
    print(f"  Saved to : {INPUT_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Phase 4: Grey reference calibration
    # ------------------------------------------------------------------
    print("\n--- Phase 4: Grey Reference Calibration ---")
    cal_result = sample_grey_reference(corrected)

    if not cal_result["success"]:
        print(f"  ERROR: Calibration failed: {cal_result['reason']}")
        sys.exit(1)

    # Print detailed calibration report
    obs = cal_result["observed_bgr"]
    gains = cal_result["gains"]
    quality = cal_result["quality"]
    roi = cal_result["roi"]

    print(f"  Grey ROI (pixels)      : x={roi[0]}-{roi[2]}, y={roi[1]}-{roi[3]}")
    print(f"  Observed BGR medians   : B={obs[0]:.1f}  G={obs[1]:.1f}  R={obs[2]:.1f}")
    print(f"  Reference target       : {cal_result['reference_target']:.2f}")
    print(f"  Correction gains       : B={gains['B']:.4f}  G={gains['G']:.4f}  R={gains['R']:.4f}")
    print(f"  Channel imbalance      : {quality['channel_imbalance']:.2f}  (0 = perfect neutral)")
    print(f"  Patch std dev          : {quality['patch_std']:.2f}  (low = uniform surface)")
    print(f"  Patch mean intensity   : {quality['patch_mean']:.2f}")
    print(f"  ROI area               : {quality['roi_area_px']} px²")

    # ------------------------------------------------------------------
    # Save ROI visualisation (green rectangle on corrected image)
    # ------------------------------------------------------------------
    roi_vis = corrected.copy()
    x1, y1, x2, y2 = roi
    cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(
        roi_vis, "Grey reference ROI",
        (x1 + 5, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
    )
    cv2.imwrite(ROI_SAVE_PATH, roi_vis)
    print(f"\n  ROI visualisation saved to : {ROI_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Apply correction and save output
    # ------------------------------------------------------------------
    colour_corrected = apply_grey_correction(corrected, cal_result)
    cv2.imwrite(OUTPUT_SAVE_PATH, colour_corrected)
    print(f"  Colour-corrected image saved to : {OUTPUT_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Calibration Summary")
    print("=" * 60)
    print()
    print("  How to verify:")
    print("  1. Open calibration_input.png  -> perspective-corrected card")
    print("  2. Open calibration_roi.png    -> green box = grey sample region")
    print("     It should sit in the central grey area of the card,")
    print("     away from the ArUco markers at the corners.")
    print("  3. Open calibration_output.png -> colour-cast-corrected card")
    print("     Any warm/cool tint should appear reduced vs. input.")
    print()

    if quality["channel_imbalance"] < 5:
        print("  Result: Excellent - illumination is nearly neutral grey.")
    elif quality["channel_imbalance"] < 15:
        print("  Result: Good - mild colour cast corrected.")
    elif quality["channel_imbalance"] < 30:
        print("  Result: Moderate colour cast detected and corrected.")
    else:
        print("  Result: Strong colour cast detected - check lighting conditions.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
