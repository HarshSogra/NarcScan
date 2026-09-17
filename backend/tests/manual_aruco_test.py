"""
manual_aruco_test.py

Run this script manually to visually verify the ArUco detector.
It generates a synthetic image, runs detection, and prints results.

Usage (from the backend/ folder, with .venv active):
    python tests/manual_aruco_test.py
"""
import sys
import os
import cv2
import numpy as np

# Make sure Python can find the 'app' package when running this script directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers, ARUCO_DICT


def generate_test_image_with_all_markers() -> np.ndarray:
    """
    Build a simple image containing all four expected markers (IDs 0-3)
    arranged in a horizontal row with white borders between them.
    """
    marker_size = 200
    border = 30
    gap = 20

    images = []
    for marker_id in [0, 1, 2, 3]:
        marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_size)
        # Add white border around each marker
        bordered = cv2.copyMakeBorder(
            marker_img,
            top=border, bottom=border, left=border, right=border,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )
        images.append(bordered)

    # Stack all four markers side by side
    h = images[0].shape[0]
    total_w = sum(img.shape[1] for img in images) + gap * (len(images) - 1)
    canvas = np.ones((h, total_w), dtype=np.uint8) * 255

    x = 0
    for img in images:
        canvas[:, x:x + img.shape[1]] = img
        x += img.shape[1] + gap

    return canvas


if __name__ == "__main__":
    print("=" * 50)
    print("NarcScan — Manual ArUco Detector Test")
    print("=" * 50)

    # Generate the test image
    test_image = generate_test_image_with_all_markers()
    print(f"\nGenerated test image: {test_image.shape[1]}x{test_image.shape[0]} pixels (grayscale)")

    # Run the detector
    result = detect_markers(test_image)

    # Print results
    print(f"\nDetected any marker : {result['detected']}")
    print(f"Missing IDs         : {result['missing_ids']}")
    print(f"\nDetected markers:")

    if not result["markers"]:
        print("  (none)")
    else:
        for marker_id, corners in result["markers"].items():
            print(f"\n  Marker ID {marker_id}:")
            for i, point in enumerate(corners):
                label = ["top-left", "top-right", "bottom-right", "bottom-left"][i]
                print(f"    corner {i} ({label}): x={point[0]:.1f}, y={point[1]:.1f}")

    # Optionally save the image to disk so you can inspect it
    output_path = os.path.join(os.path.dirname(__file__), "manual_test_output.png")
    cv2.imwrite(output_path, test_image)
    print(f"\nTest image saved to: {output_path}")
    print("\nDone.")
