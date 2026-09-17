"""
scan_image.py

Loads any image file from disk and runs the ArUco detector on it.
Prints detected marker IDs, corners, and missing IDs.

Usage (from the backend/ folder, with .venv active):
    .venv\Scripts\python.exe tests/scan_image.py <path_to_image>

Example:
    .venv\Scripts\python.exe tests/scan_image.py ../test_data/my_photo.jpg
"""
import sys
import os
import cv2

# Allow Python to find the 'app' package when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers


def main():
    # --- Check that the user gave us a file path ---
    if len(sys.argv) < 2:
        print("Usage: python tests/scan_image.py <path_to_image>")
        print("Example: python tests/scan_image.py ../test_data/photo.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # --- Check the file actually exists ---
    if not os.path.exists(image_path):
        print(f"Error: file not found: {image_path}")
        sys.exit(1)

    print("=" * 50)
    print("NarcScan — Image Scanner")
    print("=" * 50)
    print(f"\nScanning: {image_path}")

    # --- Load the image ---
    # cv2.imread loads the image as a BGR (colour) NumPy array.
    # Our detector handles BGR images, so no conversion needed here.
    image = cv2.imread(image_path)

    if image is None:
        print("Error: OpenCV could not read this file.")
        print("Make sure it is a valid image (JPG, PNG, etc.)")
        sys.exit(1)

    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # --- Run the detector ---
    result = detect_markers(image)

    # --- Print results ---
    print(f"\nDetected any marker : {result['detected']}")
    print(f"Missing IDs         : {result['missing_ids']}")

    if not result["detected"]:
        print("\nNo ArUco markers found in this image.")
        print("This is expected if the image does not contain any ArUco markers.")
        return

    print(f"\nDetected markers ({len(result['markers'])} found):")
    for marker_id, corners in result["markers"].items():
        print(f"\n  Marker ID {marker_id}:")
        labels = ["top-left", "top-right", "bottom-right", "bottom-left"]
        for i, point in enumerate(corners):
            print(f"    corner {i} ({labels[i]}): x={point[0]:.1f}, y={point[1]:.1f}")

    if result["missing_ids"]:
        print(f"\nWarning: expected markers not found: {result['missing_ids']}")
        print("All 4 markers (IDs 0-3) are required for perspective correction in Phase 3.")
    else:
        print("\nAll 4 expected markers detected. Ready for Phase 3.")


if __name__ == "__main__":
    main()
