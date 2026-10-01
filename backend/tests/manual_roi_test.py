"""
manual_roi_test.py
==================

Phase 5 - Manual Test Strip ROI Extraction Test

Run this script to visually verify that the ROI extraction works correctly
on a real phone photograph of the NarcScan card.

Runs the full pipeline up to and including Phase 5:
    ArUco detection
    -> Perspective correction
    -> Grey calibration
    -> Test strip ROI extraction

Usage
-----
From the backend/ folder with the virtual environment active:

    python tests/manual_roi_test.py ../test_data/test_data2.png

You can also pass the Phase-3 perspective-corrected image directly:

    python tests/manual_roi_test.py tests/calibration_input.png

Output
------
Saved to backend/tests/:
    roi_input.png   -- the grey-calibrated card (Phase 4 output)
    roi_output.png  -- same card with the test-strip ROI rectangle drawn on it

Open roi_output.png to verify:
  - The magenta rectangle sits in the CENTRAL area of the card.
  - It is well below the grey calibration region (upper-centre).
  - It is well above and away from the ArUco marker corners.

If the rectangle is in the wrong place, adjust the STRIP_ROI_*_FRAC constants
in backend/app/cv/roi.py.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers
from app.cv.perspective import correct_perspective
from app.cv.calibration import sample_grey_reference, apply_grey_correction
from app.cv.roi import (
    extract_strip_roi,
    draw_strip_roi,
    STRIP_ROI_LEFT_FRAC,
    STRIP_ROI_RIGHT_FRAC,
    STRIP_ROI_TOP_FRAC,
    STRIP_ROI_BOTTOM_FRAC,
)

TESTS_DIR        = os.path.dirname(__file__)
INPUT_SAVE_PATH  = os.path.join(TESTS_DIR, "roi_input.png")
OUTPUT_SAVE_PATH = os.path.join(TESTS_DIR, "roi_output.png")


def main():
    # ------------------------------------------------------------------
    # Argument
    # ------------------------------------------------------------------
    if len(sys.argv) < 2:
        print()
        print("Usage: python tests/manual_roi_test.py <image_path>")
        print()
        print("Examples:")
        print("  python tests/manual_roi_test.py ../test_data/test_data2.png")
        print("  python tests/manual_roi_test.py tests/calibration_input.png")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"\nERROR: File not found: {image_path}")
        sys.exit(1)

    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"ERROR: OpenCV could not read: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"  Input size : {w} x {h}")

    # ------------------------------------------------------------------
    # Decide whether the input already looks like a corrected card
    # (roughly 1000×700) or a raw phone photo that needs full pipeline.
    # We detect this by checking whether ArUco markers are present.
    # ------------------------------------------------------------------
    detection = detect_markers(image)

    if not detection["missing_ids"]:
        # Raw photo — run full pipeline
        print("\n--- Phase 2: ArUco Detection ---")
        print(f"  Detected IDs : {sorted(detection['markers'].keys())}")

        print("\n--- Phase 3: Perspective Correction ---")
        p_result = correct_perspective(image, detection)
        if not p_result["success"]:
            print(f"  ERROR: {p_result['reason']}")
            sys.exit(1)
        corrected = p_result["warped_image"]
        ph, pw = corrected.shape[:2]
        print(f"  Corrected size : {pw} x {ph}")

        print("\n--- Phase 4: Grey Calibration ---")
        cal_result = sample_grey_reference(corrected)
        if not cal_result["success"]:
            print(f"  WARNING: Calibration failed: {cal_result['reason']}")
            print("  Skipping correction — using uncorrected image.")
            calibrated = corrected
        else:
            obs = cal_result["observed_bgr"]
            gains = cal_result["gains"]
            print(f"  Observed BGR  : B={obs[0]:.1f}  G={obs[1]:.1f}  R={obs[2]:.1f}")
            print(f"  Gains         : B={gains['B']:.4f}  G={gains['G']:.4f}  R={gains['R']:.4f}")
            calibrated = apply_grey_correction(corrected, cal_result)
    else:
        # Already a corrected card (or partial detection) — skip to Phase 5
        print("\nNo (or partial) ArUco markers found — treating input as")
        print("a pre-corrected card image.  Skipping Phases 2–4.")
        calibrated = image

    # ------------------------------------------------------------------
    # Save Phase 4 output as roi_input.png
    # ------------------------------------------------------------------
    cv2.imwrite(INPUT_SAVE_PATH, calibrated)
    print(f"\n  Saved calibrated card : {INPUT_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Phase 5: Extract ROI
    # ------------------------------------------------------------------
    print("\n--- Phase 5: Test Strip ROI Extraction ---")
    roi_result = extract_strip_roi(calibrated)

    if not roi_result["success"]:
        print(f"  ERROR: ROI extraction failed: {roi_result['reason']}")
        sys.exit(1)

    x1, y1, x2, y2 = roi_result["pixel_coords"]
    rw, rh = roi_result["roi_width"], roi_result["roi_height"]
    left, top, right, bottom = roi_result["norm_coords"]

    print(f"  Strip ROI (pixels)   : x={x1}–{x2},  y={y1}–{y2}")
    print(f"  Strip ROI (norm)     : left={left:.2f}  top={top:.2f}  "
          f"right={right:.2f}  bottom={bottom:.2f}")
    print(f"  Extracted size       : {rw} x {rh} px")
    print(f"  Extracted area       : {rw * rh:,} px²")

    # ------------------------------------------------------------------
    # Draw ROI and save
    # ------------------------------------------------------------------
    visualised = draw_strip_roi(calibrated, roi_result=roi_result)
    cv2.imwrite(OUTPUT_SAVE_PATH, visualised)
    print(f"\n  ROI visualisation saved : {OUTPUT_SAVE_PATH}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  Phase 5 Summary")
    print("=" * 60)
    print()
    print(f"  ROI constants (in roi.py):")
    print(f"    STRIP_ROI_LEFT_FRAC   = {STRIP_ROI_LEFT_FRAC}")
    print(f"    STRIP_ROI_RIGHT_FRAC  = {STRIP_ROI_RIGHT_FRAC}")
    print(f"    STRIP_ROI_TOP_FRAC    = {STRIP_ROI_TOP_FRAC}")
    print(f"    STRIP_ROI_BOTTOM_FRAC = {STRIP_ROI_BOTTOM_FRAC}")
    print()
    print("  How to verify:")
    print("  1. Open roi_input.png  -> grey-calibrated card (Phase 4 output)")
    print("  2. Open roi_output.png -> card with magenta rectangle drawn")
    print("     The rectangle should sit in the CENTRAL area of the card,")
    print("     below the upper-centre grey calibration region,")
    print("     and above / away from the corner ArUco markers.")
    print()
    print("  To reposition the strip, edit STRIP_ROI_*_FRAC in app/cv/roi.py")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
