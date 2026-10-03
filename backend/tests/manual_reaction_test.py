"""
manual_reaction_test.py
=======================

Manual test for Phase 7 reaction colour analysis.

Usage:
    python manual_reaction_test.py <phase5_roi_image_path>

Example:
    python manual_reaction_test.py roi_output.png

IMPORTANT:
This script expects the input image to be an ALREADY EXTRACTED test-strip ROI 
(from Phase 5). It does not perform ArUco detection, perspective correction, 
or grey calibration. Please use manual_roi_test.py to generate a valid ROI 
image if needed.
"""

import sys
import os
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.reaction_analysis import extract_reaction_features

def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_reaction_test.py <phase5_roi_image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"\nLoading ROI image: {image_path}")
    roi_image = cv2.imread(image_path)
    
    if roi_image is None:
        print("Error: Could not decode image.")
        sys.exit(1)
        
    print(f"  Input size: {roi_image.shape[1]} x {roi_image.shape[0]} px")
    print("\n--- Phase 7: Reaction Analysis ---")

    # We will use the default region boundaries (0.33 and 0.67)
    left_frac = 0.33
    right_frac = 0.67
    
    result = extract_reaction_features(
        roi_image, 
        region_left_frac=left_frac, 
        region_right_frac=right_frac
    )

    if not result.success:
        print(f"  Failed: {result.reason}")
        sys.exit(1)

    # Print Overall
    print("\n  Overall ROI:")
    print(f"    Valid pixels: {result.overall.valid_pixel_count}")
    print(f"    Median Lab: L*={result.overall.L_median:.2f}  a*={result.overall.a_median:.2f}  b*={result.overall.b_median:.2f}")
    print(f"    Mean Lab:   L*={result.overall.L_mean:.2f}  a*={result.overall.a_mean:.2f}  b*={result.overall.b_mean:.2f}")
    print(f"    Std Dev:    L*={result.overall.L_std:.2f}  a*={result.overall.a_std:.2f}  b*={result.overall.b_std:.2f}")
    
    # Print Regions
    print("\n  Regions (Median Lab):")
    print(f"    Left:   L*={result.regions.left.L_median:.2f}  a*={result.regions.left.a_median:.2f}  b*={result.regions.left.b_median:.2f}")
    print(f"    Center: L*={result.regions.center.L_median:.2f}  a*={result.regions.center.a_median:.2f}  b*={result.regions.center.b_median:.2f}")
    print(f"    Right:  L*={result.regions.right.L_median:.2f}  a*={result.regions.right.a_median:.2f}  b*={result.regions.right.b_median:.2f}")

    # Print Delta E
    print("\n  Delta-E Comparisons:")
    print(f"    Center vs Left:  {result.delta_e.center_vs_left:.2f}")
    print(f"    Center vs Right: {result.delta_e.center_vs_right:.2f}")
    print(f"    Left vs Right:   {result.delta_e.left_vs_right:.2f}")
    
    # Visualization
    h, w = roi_image.shape[:2]
    left_x = int(w * left_frac)
    right_x = int(w * right_frac)
    
    vis = roi_image.copy()
    
    # Draw boundaries
    cv2.line(vis, (left_x, 0), (left_x, h), (0, 0, 0), 2)
    cv2.line(vis, (right_x, 0), (right_x, h), (0, 0, 0), 2)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    cv2.putText(vis, "LEFT", (10, h // 2), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(vis, "LEFT", (10, h // 2), font, font_scale, (255, 255, 255), thickness)
    
    cv2.putText(vis, "CENTER", (left_x + 10, h // 2), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(vis, "CENTER", (left_x + 10, h // 2), font, font_scale, (255, 255, 255), thickness)
    
    cv2.putText(vis, "RIGHT", (right_x + 10, h // 2), font, font_scale, (0, 0, 0), thickness+1)
    cv2.putText(vis, "RIGHT", (right_x + 10, h // 2), font, font_scale, (255, 255, 255), thickness)
    
    out_path = "reaction_output.png"
    cv2.imwrite(out_path, vis)
    print(f"\n  Saved visualisation to: {out_path}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
