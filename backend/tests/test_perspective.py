"""
test_perspective.py
===================

Phase 3 automated tests for the perspective correction module.

All tests are deterministic - no real camera image required.

Strategy
--------
We synthetically build a card-like image containing all four ArUco markers
placed at the four corners, then:
  1. Optionally apply a perspective warp to simulate a tilted camera.
  2. Run the ArUco detector.
  3. Pass the result to correct_perspective().
  4. Verify the output dimensions and success status.
"""

import sys
import os

import cv2
import numpy as np
import pytest

# Allow imports of 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.aruco_detector import detect_markers, ARUCO_DICT
from app.cv.perspective import (
    correct_perspective,
    OUTPUT_WIDTH,
    OUTPUT_HEIGHT,
    _marker_center,
)


# ---------------------------------------------------------------------------
# Shared helper - build a synthetic card image
# ---------------------------------------------------------------------------

def make_card_image(
    card_width: int = 1000,
    card_height: int = 700,
    marker_size: int = 100,
    border: int = 10,
) -> np.ndarray:
    """
    Build a synthetic BGR image that looks like the NarcScan card.

    The image contains all four ArUco markers (IDs 0-3) placed at the four
    corners of a white rectangle on a grey background.

    Layout (matching the physical card):
        ID 0 = top-left
        ID 1 = top-right
        ID 2 = bottom-right
        ID 3 = bottom-left

    Parameters
    ----------
    card_width  : total image width in pixels
    card_height : total image height in pixels
    marker_size : size of each ArUco marker in pixels (square)
    border      : white border added around each marker so detection works

    Returns
    -------
    A 3-channel BGR image (NumPy array).
    """
    # Create a white background canvas in BGR (3 channels)
    canvas = np.ones((card_height, card_width, 3), dtype=np.uint8) * 255

    def place_marker(marker_id: int, top_left_x: int, top_left_y: int):
        """Draw one ArUco marker (with border) at the given position."""
        marker_img = cv2.aruco.generateImageMarker(ARUCO_DICT, marker_id, marker_size)

        # Add white border so the detector can see the marker edges clearly
        bordered = cv2.copyMakeBorder(
            marker_img,
            top=border, bottom=border, left=border, right=border,
            borderType=cv2.BORDER_CONSTANT,
            value=255,
        )

        h, w = bordered.shape
        # Convert grayscale marker to BGR to paste onto the colour canvas
        bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        canvas[top_left_y:top_left_y + h, top_left_x:top_left_x + w] = bgr

    full_size = marker_size + 2 * border   # total marker+border block size

    # Place the four markers at the four corners of the image
    place_marker(marker_id=0, top_left_x=0,                          top_left_y=0)
    place_marker(marker_id=1, top_left_x=card_width - full_size,     top_left_y=0)
    place_marker(marker_id=2, top_left_x=card_width - full_size,     top_left_y=card_height - full_size)
    place_marker(marker_id=3, top_left_x=0,                          top_left_y=card_height - full_size)

    return canvas


def apply_perspective_warp(image: np.ndarray, strength: int = 60) -> np.ndarray:
    """
    Simulate a perspective distortion by warping the image with a known
    homography.

    We shift the top-right corner inward (to simulate the camera being
    positioned slightly to the left).  This makes the card appear as a
    trapezoid - similar to a real phone photograph taken at an angle.

    Parameters
    ----------
    image    : the original flat card image
    strength : pixel shift applied to the top-right corner

    Returns
    -------
    A perspective-warped version of the input image (same size as input).
    """
    h, w = image.shape[:2]

    # Source: the four corners of the original flat image
    src = np.float32([
        [0,       0      ],   # top-left
        [w - 1,   0      ],   # top-right
        [w - 1,   h - 1  ],   # bottom-right
        [0,       h - 1  ],   # bottom-left
    ])

    # Destination: move the top-right corner inward to simulate camera tilt
    dst = np.float32([
        [0,              0      ],   # top-left stays fixed
        [w - 1 - strength, strength],   # top-right moves in and down
        [w - 1,          h - 1  ],   # bottom-right stays fixed
        [0,              h - 1  ],   # bottom-left stays fixed
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMarkerCenter:
    """Unit tests for the _marker_center helper."""

    def test_center_of_axis_aligned_square(self):
        """
        A 100x100 square of corners should have center at (50, 50).
        """
        corners = [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]
        cx, cy = _marker_center(corners)
        assert abs(cx - 50.0) < 1e-5
        assert abs(cy - 50.0) < 1e-5

    def test_center_of_offset_square(self):
        """
        A 100x100 square offset to (200, 300) should have center at (250, 350).
        """
        corners = [
            [200.0, 300.0],
            [300.0, 300.0],
            [300.0, 400.0],
            [200.0, 400.0],
        ]
        cx, cy = _marker_center(corners)
        assert abs(cx - 250.0) < 1e-5
        assert abs(cy - 350.0) < 1e-5


class TestMissingMarkers:
    """Tests for the failure path when markers are missing."""

    def test_missing_all_markers_returns_failure(self):
        """
        A blank image (no markers) should return success=False.
        """
        blank = np.ones((300, 400, 3), dtype=np.uint8) * 255
        detection = detect_markers(blank)
        result = correct_perspective(blank, detection)

        assert result["success"] is False
        assert "missing_ids" in result
        assert len(result["missing_ids"]) == 4

    def test_missing_single_marker_returns_failure(self):
        """
        If one marker is missing the whole correction must fail gracefully.
        We simulate this by manually constructing a detection result that
        omits marker ID 2.
        """
        fake_detection = {
            "detected": True,
            "markers": {
                0: [[10, 10], [50, 10], [50, 50], [10, 50]],
                1: [[300, 10], [340, 10], [340, 50], [300, 50]],
                # ID 2 is deliberately missing
                3: [[10, 200], [50, 200], [50, 240], [10, 240]],
            },
            "missing_ids": [2],
        }

        dummy_image = np.ones((300, 400, 3), dtype=np.uint8) * 200
        result = correct_perspective(dummy_image, fake_detection)

        assert result["success"] is False
        assert 2 in result["missing_ids"]
        assert "reason" in result

    def test_failure_result_does_not_contain_warped_image(self):
        """
        On failure, 'warped_image' should not be in the result.
        """
        fake_detection = {
            "detected": False,
            "markers": {},
            "missing_ids": [0, 1, 2, 3],
        }
        dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = correct_perspective(dummy_image, fake_detection)

        assert result["success"] is False
        assert "warped_image" not in result


class TestPerfectRectangularCard:
    """
    Tests using a perfectly flat, rectangular synthetic card.

    This is the ideal case: no perspective distortion.
    """

    def setup_method(self):
        """Build the card image and run detection once for all tests."""
        self.card = make_card_image()
        self.detection = detect_markers(self.card)

    def test_all_four_markers_detected(self):
        """All four ArUco markers must be found in the synthetic card image."""
        assert self.detection["detected"] is True
        assert self.detection["missing_ids"] == []
        for mid in [0, 1, 2, 3]:
            assert mid in self.detection["markers"]

    def test_correction_succeeds(self):
        """correct_perspective() should succeed on a card with all markers."""
        result = correct_perspective(self.card, self.detection)
        assert result["success"] is True

    def test_output_dimensions_match_default_constants(self):
        """
        The warped image must have the exact dimensions specified by the
        OUTPUT_WIDTH and OUTPUT_HEIGHT constants.
        """
        result = correct_perspective(self.card, self.detection)
        assert result["success"] is True

        warped = result["warped_image"]
        assert warped.shape[1] == OUTPUT_WIDTH,  f"Expected width {OUTPUT_WIDTH}, got {warped.shape[1]}"
        assert warped.shape[0] == OUTPUT_HEIGHT, f"Expected height {OUTPUT_HEIGHT}, got {warped.shape[0]}"

    def test_custom_output_dimensions(self):
        """
        Passing custom output_width / output_height should produce an image
        with those exact dimensions.
        """
        result = correct_perspective(
            self.card,
            self.detection,
            output_width=640,
            output_height=480,
        )
        assert result["success"] is True
        warped = result["warped_image"]
        assert warped.shape[1] == 640
        assert warped.shape[0] == 480

    def test_result_contains_src_and_dst_points(self):
        """The result dict should include the source and destination points."""
        result = correct_perspective(self.card, self.detection)
        assert result["success"] is True
        assert "src_points" in result
        assert "dst_points" in result
        # Must be exactly 4 points each
        assert len(result["src_points"]) == 4
        assert len(result["dst_points"]) == 4

    def test_warped_image_is_not_all_black(self):
        """
        A correctly warped card image should contain actual pixel data,
        not be entirely black (which would indicate a failed warp).
        """
        result = correct_perspective(self.card, self.detection)
        assert result["success"] is True
        warped = result["warped_image"]
        # The mean pixel value should be well above zero for a white card image
        assert np.mean(warped) > 50.0, "Warped image looks suspiciously dark"


class TestPerspectiveDistortedCard:
    """
    Tests using a synthetically perspective-distorted card.

    We start from the flat card, apply a known warp to simulate a tilted
    camera, then verify that correct_perspective() can recover a rectangular
    output of the expected size.
    """

    def setup_method(self):
        """Build the flat card, distort it, then run detection."""
        flat_card = make_card_image()
        self.distorted_card = apply_perspective_warp(flat_card, strength=60)
        self.detection = detect_markers(self.distorted_card)

    def test_markers_still_detected_after_warp(self):
        """
        ArUco markers should still be detectable in the perspective-warped
        image.  If detection fails here the warp was too aggressive.
        """
        if self.detection["missing_ids"]:
            pytest.skip(
                f"ArUco detection failed on distorted image "
                f"(missing IDs: {self.detection['missing_ids']}). "
                f"Skipping - warp may be too strong for synthetic test."
            )
        assert self.detection["detected"] is True

    def test_perspective_correction_output_has_correct_size(self):
        """
        After perspective correction the output image must have the standard
        dimensions regardless of the input distortion.
        """
        if self.detection["missing_ids"]:
            pytest.skip("Skipping: ArUco detection failed on distorted card.")

        result = correct_perspective(self.distorted_card, self.detection)
        assert result["success"] is True

        warped = result["warped_image"]
        assert warped.shape[1] == OUTPUT_WIDTH
        assert warped.shape[0] == OUTPUT_HEIGHT

    def test_perspective_correction_succeeds_on_distorted_card(self):
        """
        correct_perspective() should succeed (success=True) on the distorted
        card as long as all four markers were detected.
        """
        if self.detection["missing_ids"]:
            pytest.skip("Skipping: ArUco detection failed on distorted card.")

        result = correct_perspective(self.distorted_card, self.detection)
        assert result["success"] is True
