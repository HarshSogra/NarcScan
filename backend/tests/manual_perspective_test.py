"""
manual_perspective_test.py
==========================

Phase 3 - Manual Perspective Correction Test

Run this script to visually verify that the perspective correction module
works correctly.  It can operate in two modes:

MODE 1 - Synthetic image (no arguments):
    Generates a synthetic card image containing all four ArUco markers,
    applies a perspective warp to simulate a tilted camera, then runs
    the full pipeline:
        ArUco detection -> marker centers -> perspective correction -> output

MODE 2 - Real phone photograph (image path as argument):
    Loads a real image from disk, runs ArUco detection, then perspective
    correction.  Use this to test with actual photos of the physical card.

Usage
-----
From the backend/ folder with the virtual environment active:

    # Synthetic test (no image needed):
    python tests/manual_perspective_test.py

    # Real image test:
    python tests/manual_perspective_test.py ../test_data/aruco/real_photos/phone_test_01.jpg

Output
------
The perspective-corrected image is saved to:
    backend/tests/perspective_output.png

Open that file in any image viewer to visually verify the result.
"""

import sys
import os
import cv2
import numpy as np

# Allow 'app' to be found when running this script directly from anywhere.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers, ARUCO_DICT
from app.cv.perspective import correct_perspective, OUTPUT_WIDTH, OUTPUT_HEIGHT


# ---------------------------------------------------------------------------
# Helper - build a synthetic card image for the self-test
# ---------------------------------------------------------------------------

def make_synthetic_card(
    card_width: int = 1000,
    card_height: int = 700,
    marker_size: int = 100,
    border: int = 15,
) -> np.ndarray:
    """
    Build a flat synthetic card image with all four ArUco markers
    placed at the four corners.

    ID 0 = top-left
    ID 1 = top-right
    ID 2 = bottom-right
    ID 3 = bottom-left
    """
    canvas = np.ones((card_height, card_width, 3), dtype=np.uint8) * 240

    # Draw a light card border so the card boundary is visible
    cv2.rectangle(canvas, (0, 0), (card_width - 1, card_height - 1), (180, 180, 180), 3)

    def place_marker(marker_id: int, x: int, y: int):
        marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_size)
        bordered = cv2.copyMakeBorder(
            marker_img,
            top=border, bottom=border, left=border, right=border,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )
        h, w = bordered.shape
        bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        canvas[y:y + h, x:x + w] = bgr

    full = marker_size + 2 * border
    place_marker(0, 0,                       0)
    place_marker(1, card_width - full,       0)
    place_marker(2, card_width - full,       card_height - full)
    place_marker(3, 0,                       card_height - full)

    return canvas


def apply_perspective_distortion(image: np.ndarray, strength: int = 80) -> np.ndarray:
    """
    Apply a synthetic perspective distortion to simulate a tilted camera.

    We shift the top-right and bottom-right corners to simulate the card
    being viewed from a slight angle.
    """
    h, w = image.shape[:2]

    src = np.float32([
        [0,     0    ],
        [w - 1, 0    ],
        [w - 1, h - 1],
        [0,     h - 1],
    ])

    dst = np.float32([
        [0,              strength     ],   # top-left: shift down slightly
        [w - 1 - strength, 0         ],   # top-right: shift left
        [w - 1,          h - 1       ],   # bottom-right: unchanged
        [strength,       h - 1       ],   # bottom-left: shift right slightly
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (w, h))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(image: np.ndarray, label: str) -> None:
    """
    Run the full ArUco detection + perspective correction pipeline on
    the given image and print a detailed report.

    Parameters
    ----------
    image : the input image (BGR)
    label : a short description of the image source for printing
    """
    print()
    print("=" * 60)
    print(f"  {label}")
    print("=" * 60)

    h, w = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1
    print(f"  Input image size  : {w} x {h} pixels  ({channels} channel(s))")

    # --- Step 1: ArUco detection ---
    detection = detect_markers(image)
    detected_ids = sorted(detection["markers"].keys())
    missing_ids  = detection["missing_ids"]

    print(f"  Detected marker IDs : {detected_ids}")
    print(f"  Missing marker IDs  : {missing_ids}")

    if detected_ids:
        print()
        print("  Marker centers (source points):")
        for mid in sorted(detection["markers"].keys()):
            corners = detection["markers"][mid]
            cx = sum(p[0] for p in corners) / 4
            cy = sum(p[1] for p in corners) / 4
            print(f"    Marker ID {mid}: center = ({cx:.1f}, {cy:.1f})")

    # --- Step 2: Perspective correction ---
    result = correct_perspective(image, detection)

    print()
    if result["success"]:
        warped = result["warped_image"]
        wh, ww = warped.shape[:2]
        print(f"  Perspective correction : SUCCESS")
        print(f"  Output image size      : {ww} x {wh} pixels")
        print(f"  Source points used     : {result['src_points']}")
        return warped
    else:
        print(f"  Perspective correction : FAILED")
        print(f"  Reason                 : {result['reason']}")
        print(f"  Missing IDs            : {result['missing_ids']}")
        return None


def main():
    output_path = os.path.join(os.path.dirname(__file__), "perspective_output.png")

    if len(sys.argv) >= 2:
        # ---------------------------------------------------------------
        # MODE 2: Real phone photograph
        # ---------------------------------------------------------------
        image_path = sys.argv[1]

        if not os.path.isfile(image_path):
            print()
            print(f"ERROR: Image file not found: {image_path}")
            print()
            print("Make sure the file exists and the path is correct.")
            print("Example: python tests/manual_perspective_test.py ../test_data/aruco/real_photos/phone_test_01.jpg")
            sys.exit(1)

        print(f"\nLoading real image: {image_path}")
        image = cv2.imread(image_path)

        if image is None:
            print(f"ERROR: OpenCV could not read the image at: {image_path}")
            print("The file may be corrupt or in an unsupported format.")
            sys.exit(1)

        warped = run_pipeline(image, f"Real photo: {os.path.basename(image_path)}")

    else:
        # ---------------------------------------------------------------
        # MODE 1: Synthetic test
        # ---------------------------------------------------------------
        print("\nNo image path provided - running synthetic self-test.")
        print("(Pass an image path as an argument to test with a real photo.)")

        flat_card = make_synthetic_card()
        distorted = apply_perspective_distortion(flat_card, strength=80)

        # Save the distorted input so you can compare it with the output
        input_save_path = os.path.join(os.path.dirname(__file__), "perspective_input.png")
        cv2.imwrite(input_save_path, distorted)
        print(f"\n  Distorted input saved to : {input_save_path}")

        warped = run_pipeline(distorted, "Synthetic distorted card")

    # --- Save output ---
    if warped is not None:
        cv2.imwrite(output_path, warped)
        print()
        print(f"  Output saved to: {output_path}")
        print()
        print("  How to verify:")
        print("  - Open perspective_input.png  -> should look like a trapezoid")
        print("  - Open perspective_output.png -> should look like a flat rectangle")
        print("  - The four ArUco markers should appear at the corners in both images")
    else:
        print()
        print("  No output image saved (correction failed).")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
