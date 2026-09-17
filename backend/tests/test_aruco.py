import cv2
import numpy as np
import sys
import os

# Make sure Python can find 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers, ARUCO_DICT


# ---------------------------------------------------------------------------
# Helper — generate a synthetic ArUco marker image
# ---------------------------------------------------------------------------

def make_marker_image(marker_id: int, size: int = 200) -> np.ndarray:
    """
    Generate a single ArUco marker as a grayscale NumPy image.

    OpenCV can draw any marker from the dictionary onto a blank image.
    This gives us a deterministic image we can use in tests — no camera needed.

    Parameters
    ----------
    marker_id : int   — which marker to draw (must be in the dictionary)
    size      : int   — side length in pixels of the generated image

    Returns
    -------
    A grayscale image (2-D NumPy array) containing one ArUco marker
    centred with a white border around it.
    """
    # generateImageMarker draws the marker onto a black/white image.
    # Arguments: dictionary, id, output_size_in_pixels, border_bits
    marker_image = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, size)

    # Add a white border so the detector can clearly see the marker edges.
    # Without a border the black outer ring of the marker blends into
    # the edge of the image and detection can fail.
    bordered = cv2.copyMakeBorder(
        marker_image,
        top=20, bottom=20, left=20, right=20,
        borderType=cv2.BORDER_CONSTANT,
        value=255,   # white border
    )
    return bordered


def make_multi_marker_image(marker_ids: list, marker_size: int = 150) -> np.ndarray:
    """
    Place multiple markers side-by-side in one image.

    We place them in a horizontal row with gaps between them.
    Each marker has a white border so they don't visually merge.
    """
    border = 30
    gap = 30
    individual_images = [make_marker_image(mid, marker_size) for mid in marker_ids]

    # All generated images are the same size, so we can read width/height
    # from the first one.
    h = individual_images[0].shape[0]
    total_width = len(individual_images) * individual_images[0].shape[1] + (len(individual_images) - 1) * gap

    # Create a blank white canvas and paste each marker onto it.
    canvas = np.ones((h, total_width), dtype=np.uint8) * 255
    x_offset = 0
    for img in individual_images:
        w = img.shape[1]
        canvas[:, x_offset:x_offset + w] = img
        x_offset += w + gap

    return canvas


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_marker_detected():
    """
    A grayscale image containing only marker ID 0 should be detected.
    """
    image = make_marker_image(marker_id=0)
    result = detect_markers(image)

    assert result["detected"] is True
    assert 0 in result["markers"], "Marker 0 should be in markers dict"
    # Each marker's corners should be a list of 4 points
    assert len(result["markers"][0]) == 4


def test_single_marker_corner_format():
    """
    Each corner should be a pair of [x, y] floats in pixel coordinates.
    """
    image = make_marker_image(marker_id=1)
    result = detect_markers(image)

    assert result["detected"] is True
    corners = result["markers"][1]   # list of 4 corners
    assert len(corners) == 4

    for point in corners:
        # Each point must be [x, y]
        assert len(point) == 2, f"Expected [x, y] but got {point}"


def test_multiple_markers_detected():
    """
    An image with markers 0, 1, 2, and 3 should detect all four.
    """
    image = make_multi_marker_image([0, 1, 2, 3])
    result = detect_markers(image)

    assert result["detected"] is True

    for expected_id in [0, 1, 2, 3]:
        assert expected_id in result["markers"], f"Marker {expected_id} should be detected"

    assert result["missing_ids"] == [], "No markers should be missing"


def test_no_markers_detected():
    """
    A plain white image should return detected=False and all IDs missing.
    """
    # Create a blank white image — contains no markers at all
    blank = np.ones((300, 300), dtype=np.uint8) * 255
    result = detect_markers(blank)

    assert result["detected"] is False
    assert result["markers"] == {}
    # All four expected IDs should appear in missing_ids
    assert sorted(result["missing_ids"]) == [0, 1, 2, 3]


def test_partial_markers_missing_ids():
    """
    An image with only markers 0 and 1 should report 2 and 3 as missing.
    """
    image = make_multi_marker_image([0, 1])
    result = detect_markers(image)

    assert result["detected"] is True
    assert 0 in result["markers"]
    assert 1 in result["markers"]

    # Markers 2 and 3 were never in the image
    assert 2 in result["missing_ids"]
    assert 3 in result["missing_ids"]


def test_colour_image_works():
    """
    The detector should also accept a BGR colour image, not just grayscale.
    OpenCV normally loads images in BGR format, so this is an important case.
    """
    gray_marker = make_marker_image(marker_id=2)

    # Convert the grayscale image to a 3-channel BGR image
    bgr_marker = cv2.cvtColor(gray_marker, cv2.COLOR_GRAY2BGR)

    result = detect_markers(bgr_marker)

    assert result["detected"] is True
    assert 2 in result["markers"]
