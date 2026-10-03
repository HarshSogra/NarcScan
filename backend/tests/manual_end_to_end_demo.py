"""
manual_end_to_end_demo.py
=========================

A temporary script to demonstrate the full pipeline from raw image 
to classification.

Pipeline:
  Image -> ArUco -> Perspective -> Calibration -> ROI -> Reaction Features -> Classification
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers
from app.cv.perspective import correct_perspective
from app.cv.calibration import sample_grey_reference, apply_grey_correction
from app.cv.roi import extract_strip_roi
from app.cv.reaction_analysis import extract_reaction_features
from app.cv.classifier import classify_reaction

def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_end_to_end_demo.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"\nLoading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not decode image.")
        sys.exit(1)

    print(f"  Input size: {image.shape[1]} x {image.shape[0]} px")

    # Phase 2: ArUco Detection
    print("\n--- Phase 2: ArUco Detection ---")
    aruco_res = detect_markers(image)
    if not aruco_res.get("detected") or aruco_res.get("missing_ids"):
        print(f"  Failed: Missing markers {aruco_res.get('missing_ids')}")
        sys.exit(1)
    
    # Phase 3: Perspective
    print("--- Phase 3: Perspective Correction ---")
    persp_res = correct_perspective(image, aruco_res)
    if not persp_res.get("success"):
        print(f"  Failed: {persp_res.get('reason')}")
        sys.exit(1)
    
    # Phase 4: Calibration
    print("--- Phase 4: Grey Calibration ---")
    calib_res = sample_grey_reference(persp_res["warped_image"])
    if not calib_res.get("success"):
        print(f"  Failed: {calib_res.get('reason')}")
        sys.exit(1)
    
    calibrated_image = apply_grey_correction(persp_res["warped_image"], calib_res)

    # Phase 5: ROI Extraction
    print("--- Phase 5: Test Strip ROI Extraction ---")
    roi_res = extract_strip_roi(calibrated_image)
    if not roi_res.get("success"):
        print(f"  Failed: {roi_res.get('reason')}")
        sys.exit(1)

    # Phase 7: Reaction Analysis
    print("--- Phase 7: Reaction Colour Analysis ---")
    # You can change the region fractions if the reaction only happens on one side
    react_res = extract_reaction_features(roi_res["roi_image"])
    if not react_res.success:
        print(f"  Failed: {react_res.reason}")
        sys.exit(1)
        
    print(f"  Extracted Center L*a*b*: ({react_res.regions.center.L_median:.2f}, {react_res.regions.center.a_median:.2f}, {react_res.regions.center.b_median:.2f})")

    # Phase 8: Classification
    print("\n--- Phase 8: Demo Classification ---")
    # We will use the center region's median Lab colour for classification
    center_lab = (
        react_res.regions.center.L_median,
        react_res.regions.center.a_median,
        react_res.regions.center.b_median
    )
    
    class_res = classify_reaction(observed_lab=center_lab)
    
    if not class_res.success:
        print(f"  Failed: {class_res.reason}")
        sys.exit(1)

    print(f"  Result: {class_res.result}")
    print(f"  Mode: {class_res.mode} (Validated: {class_res.validated})")
    print(f"  Positive \u0394E: {class_res.positive_distance:.2f}")
    print(f"  Negative \u0394E: {class_res.negative_distance:.2f}")

    print("\nDone.")

if __name__ == "__main__":
    main()
