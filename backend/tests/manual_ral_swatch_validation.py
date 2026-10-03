"""
manual_ral_swatch_validation.py
===============================

Phase 8.1 - Physical Validation Hook

This standalone diagnostic script accepts an image path of a physically 
photographed RAL 4004 colour swatch (printed or paint-chip), captured 
through the existing real NarcScan pipeline, and compares the resulting 
measured Lab value against the literature-derived reference Lab for 
TEST_A (Opiates).

This does NOT feed into classify_reaction() or change any production behaviour.
"""

import sys
import os
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers
from app.cv.perspective import correct_perspective
from app.cv.calibration import sample_grey_reference, apply_grey_correction
from app.cv.roi import extract_strip_roi
from app.cv.reaction_analysis import extract_reaction_features
from app.cv.color_analysis import calculate_delta_e_ciede2000
from app.cv.reference_profiles import TEST_A

def main():
    parser = argparse.ArgumentParser(description="Validate pipeline against RAL 4004 swatch.")
    parser.add_argument("image_path", help="Path to the test image")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: File not found: {args.image_path}")
        return

    # 1. Load Image
    image = cv2.imread(args.image_path)
    if image is None:
        print("Error: Could not read image.")
        return

    print("==================================================")
    print("RAL SWATCH VALIDATION DIAGNOSTIC")
    print("==================================================")

    # 2. Pipeline: ArUco -> Perspective -> Calibration -> ROI -> Features
    markers_res = detect_markers(image)
    if not markers_res.get("success"):
        print("Error: Failed to detect ArUco markers.")
        return

    perspective_res = correct_perspective(image, markers_res["corners"], markers_res["ids"])
    if not perspective_res.get("success"):
        print("Error: Perspective correction failed.")
        return

    warped_image = perspective_res["warped_image"]

    calib_res = sample_grey_reference(warped_image)
    if not calib_res.get("success"):
        print("Error: Calibration sampling failed.")
        return

    grey_bgr = calib_res["mean_bgr"]
    corrected_image = apply_grey_correction(warped_image, grey_bgr)

    roi_res = extract_strip_roi(corrected_image)
    if not roi_res.get("success"):
        print("Error: ROI extraction failed.")
        return

    roi_image = roi_res["roi_image"]

    features_res = extract_reaction_features(roi_image)
    if not features_res.get("success"):
        print("Error: Feature extraction failed.")
        return

    observed_lab = features_res["median_lab"]
    
    # 3. Compare with TEST_A Opiates target
    target = TEST_A.target_reactions["Opiates"]
    ref_lab = target.reference_lab

    if ref_lab is None:
        print("Error: TEST_A Opiates has no reference Lab populated.")
        return

    dist = calculate_delta_e_ciede2000(observed_lab, ref_lab)

    print()
    print("--------------------------------------------------")
    print(f"Observed Lab (from image) : (L*={observed_lab[0]:.1f}, a*={observed_lab[1]:.1f}, b*={observed_lab[2]:.1f})")
    print(f"Reference Lab (RAL 4004)  : (L*={ref_lab[0]:.1f}, a*={ref_lab[1]:.1f}, b*={ref_lab[2]:.1f})")
    print(f"\u0394E (CIEDE2000) Distance   : {dist:.2f}")
    print("--------------------------------------------------")
    print()
    print("IMPORTANT DISCLAIMER:")
    print("Camera-measured Lab vs. literature-derived reference Lab \u2014 this is a")
    print("sanity check of our own measurement pipeline against a published standard,")
    print("NOT a validated accuracy claim.")
    print("==================================================")


if __name__ == "__main__":
    main()
