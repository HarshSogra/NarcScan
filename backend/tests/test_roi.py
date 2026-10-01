"""
test_roi.py
===========

Phase 5 automated tests for the test-strip ROI extraction module.

All tests are deterministic — no real camera image required.

Strategy
--------
We build known solid-colour or gradient BGR images and verify:
  - Normalised → pixel coordinate conversion
  - Correct ROI dimensions at standard and non-standard resolutions
  - Correct pixel content in the extracted crop
  - All validation / guard paths
  - The original image is never mutated
  - The visualisation helper works correctly
"""

import sys
import os

import cv2
import numpy as np
import pytest

# Allow imports of 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.roi import (
    extract_strip_roi,
    draw_strip_roi,
    _compute_strip_pixels,
    STRIP_ROI_LEFT_FRAC,
    STRIP_ROI_RIGHT_FRAC,
    STRIP_ROI_TOP_FRAC,
    STRIP_ROI_BOTTOM_FRAC,
    MIN_STRIP_AREA_PX,
    ROI_DRAW_COLOUR,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_solid_bgr(height: int, width: int, b: int, g: int, r: int) -> np.ndarray:
    """Create a solid-colour BGR uint8 image with known pixel values."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def make_standard_card() -> np.ndarray:
    """Return a 1000 × 700 neutral grey card image (Phase 3 standard size)."""
    return make_solid_bgr(700, 1000, 128, 128, 128)


# ---------------------------------------------------------------------------
# TestComputeStripPixels
# ---------------------------------------------------------------------------

class TestComputeStripPixels:
    """Unit tests for the _compute_strip_pixels helper."""

    def test_roi_inside_image_standard_size(self):
        """Default ROI must fit entirely within the standard 1000×700 image."""
        w, h = 1000, 700
        x1, y1, x2, y2 = _compute_strip_pixels(w, h)
        assert 0 <= x1 < x2 <= w, f"x bounds invalid: {x1} {x2} {w}"
        assert 0 <= y1 < y2 <= h, f"y bounds invalid: {y1} {y2} {h}"

    def test_fractions_applied_correctly(self):
        """Pixel coordinates must exactly match fractions × dimensions (rounded)."""
        w, h = 1000, 700
        x1, y1, x2, y2 = _compute_strip_pixels(w, h)
        assert x1 == round(STRIP_ROI_LEFT_FRAC   * w)
        assert x2 == round(STRIP_ROI_RIGHT_FRAC  * w)
        assert y1 == round(STRIP_ROI_TOP_FRAC    * h)
        assert y2 == round(STRIP_ROI_BOTTOM_FRAC * h)

    def test_positive_area(self):
        """The computed ROI must always have positive width and height."""
        x1, y1, x2, y2 = _compute_strip_pixels(800, 600)
        assert x2 > x1
        assert y2 > y1

    def test_different_resolution_still_inside(self):
        """ROI must fit inside non-standard resolutions too."""
        for w, h in [(640, 480), (1920, 1080), (500, 500), (1280, 720)]:
            x1, y1, x2, y2 = _compute_strip_pixels(w, h)
            assert 0 <= x1 < x2 <= w, f"x out of bounds for {w}×{h}"
            assert 0 <= y1 < y2 <= h, f"y out of bounds for {w}×{h}"

    def test_roi_does_not_overlap_calibration_region(self):
        """
        The strip ROI (top >= 0.50) must sit BELOW the calibration ROI
        (bottom = 0.45) with no overlap.
        """
        from app.cv.calibration import GREY_ROI_BOTTOM_FRAC
        assert STRIP_ROI_TOP_FRAC >= GREY_ROI_BOTTOM_FRAC, (
            "Strip ROI top must be at or below the calibration ROI bottom"
        )


# ---------------------------------------------------------------------------
# TestExtractStripRoiSuccess
# ---------------------------------------------------------------------------

class TestExtractStripRoiSuccess:
    """Tests for the happy-path of extract_strip_roi."""

    def test_success_flag_is_true(self):
        """extract_strip_roi must return success=True on a valid image."""
        image = make_standard_card()
        result = extract_strip_roi(image)
        assert result["success"] is True

    def test_result_contains_required_keys(self):
        """All documented keys must be present in a success result."""
        image = make_standard_card()
        result = extract_strip_roi(image)
        for key in ("success", "roi_image", "pixel_coords",
                    "norm_coords", "roi_width", "roi_height"):
            assert key in result, f"Missing key: {key}"

    def test_roi_image_is_numpy_array(self):
        """The extracted ROI must be a NumPy ndarray."""
        result = extract_strip_roi(make_standard_card())
        assert isinstance(result["roi_image"], np.ndarray)

    def test_roi_image_dtype_is_uint8(self):
        """The extracted ROI must be uint8."""
        result = extract_strip_roi(make_standard_card())
        assert result["roi_image"].dtype == np.uint8

    def test_roi_image_is_three_channel(self):
        """The extracted ROI must have 3 channels (BGR)."""
        result = extract_strip_roi(make_standard_card())
        assert len(result["roi_image"].shape) == 3
        assert result["roi_image"].shape[2] == 3

    def test_roi_width_and_height_match_image_shape(self):
        """
        roi_width and roi_height must equal the actual dimensions of
        the extracted roi_image array.
        """
        result = extract_strip_roi(make_standard_card())
        assert result["success"] is True
        roi_img = result["roi_image"]
        assert result["roi_height"] == roi_img.shape[0]
        assert result["roi_width"]  == roi_img.shape[1]

    def test_roi_dimensions_match_standard_card(self):
        """
        At 1000 × 700 the default ROI must produce the expected pixel size.
        """
        w, h = 1000, 700
        x1, y1, x2, y2 = _compute_strip_pixels(w, h)
        expected_w = x2 - x1
        expected_h = y2 - y1

        result = extract_strip_roi(make_standard_card())
        assert result["roi_width"]  == expected_w
        assert result["roi_height"] == expected_h

    def test_pixel_coords_tuple_has_four_elements(self):
        """pixel_coords must be a 4-element tuple (x1, y1, x2, y2)."""
        result = extract_strip_roi(make_standard_card())
        assert len(result["pixel_coords"]) == 4

    def test_pixel_coords_inside_image(self):
        """All pixel coordinates must be within the image bounds."""
        w, h = 1000, 700
        result = extract_strip_roi(make_standard_card())
        x1, y1, x2, y2 = result["pixel_coords"]
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h

    def test_norm_coords_tuple_has_four_elements(self):
        """norm_coords must be a 4-element tuple."""
        result = extract_strip_roi(make_standard_card())
        assert len(result["norm_coords"]) == 4

    def test_norm_coords_match_constants(self):
        """
        When no override roi is given, norm_coords must equal the module
        constants exactly.
        """
        result = extract_strip_roi(make_standard_card())
        left, top, right, bottom = result["norm_coords"]
        assert abs(left   - STRIP_ROI_LEFT_FRAC)   < 1e-9
        assert abs(top    - STRIP_ROI_TOP_FRAC)    < 1e-9
        assert abs(right  - STRIP_ROI_RIGHT_FRAC)  < 1e-9
        assert abs(bottom - STRIP_ROI_BOTTOM_FRAC) < 1e-9

    def test_extracted_pixel_values_match_source(self):
        """
        Pixel values in the extracted crop must exactly match the
        corresponding region of the source image.
        """
        # Use a unique colour so we can distinguish the crop from any border.
        image = make_solid_bgr(700, 1000, b=42, g=84, r=168)
        result = extract_strip_roi(image)
        assert result["success"] is True

        x1, y1, x2, y2 = result["pixel_coords"]
        expected = image[y1:y2, x1:x2]
        np.testing.assert_array_equal(result["roi_image"], expected)

    def test_roi_is_a_copy_not_a_view(self):
        """
        Modifying the returned roi_image must NOT affect the original image.
        """
        image = make_solid_bgr(700, 1000, b=100, g=100, r=100)
        result = extract_strip_roi(image)
        assert result["success"] is True

        roi_img = result["roi_image"]
        original_value = int(image[result["pixel_coords"][1],
                                    result["pixel_coords"][0], 0])

        # Overwrite the entire extracted crop with zeros.
        roi_img[:] = 0

        # The source image must remain unchanged.
        assert int(image[result["pixel_coords"][1],
                          result["pixel_coords"][0], 0]) == original_value

    def test_original_image_is_unchanged_after_extract(self):
        """
        extract_strip_roi must never modify the original image.
        We take a full copy before calling and compare after.
        """
        image = make_standard_card()
        image_before = image.copy()
        extract_strip_roi(image)
        np.testing.assert_array_equal(image, image_before)

    # --- Custom ROI override ---

    def test_custom_roi_override_uses_supplied_coords(self):
        """
        Passing an explicit roi tuple must use those coordinates, not the
        default fractional ones.
        """
        image = make_solid_bgr(700, 1000, b=0, g=0, r=0)
        # Paint a unique colour in a known region
        image[100:200, 200:400] = [55, 66, 77]

        result = extract_strip_roi(image, roi=(200, 100, 400, 200))
        assert result["success"] is True
        assert result["pixel_coords"] == (200, 100, 400, 200)
        assert result["roi_width"]  == 200
        assert result["roi_height"] == 100

        # Pixel content should all be [55, 66, 77]
        assert result["roi_image"][0, 0, 0] == 55
        assert result["roi_image"][0, 0, 1] == 66
        assert result["roi_image"][0, 0, 2] == 77

    # --- Different resolutions ---

    def test_works_at_640x480(self):
        """extract_strip_roi must succeed at 640×480."""
        image = make_solid_bgr(480, 640, 128, 128, 128)
        result = extract_strip_roi(image)
        assert result["success"] is True
        x1, y1, x2, y2 = result["pixel_coords"]
        assert 0 <= x1 < x2 <= 640
        assert 0 <= y1 < y2 <= 480

    def test_works_at_1920x1080(self):
        """extract_strip_roi must succeed at 1920×1080."""
        image = make_solid_bgr(1080, 1920, 128, 128, 128)
        result = extract_strip_roi(image)
        assert result["success"] is True
        x1, y1, x2, y2 = result["pixel_coords"]
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080

    def test_roi_touches_image_boundary_is_valid(self):
        """
        A custom ROI that exactly touches the image boundary (x2==w, y2==h)
        must be accepted (boundary is inclusive for this validation).
        """
        image = make_solid_bgr(200, 200, 100, 100, 100)
        result = extract_strip_roi(image, roi=(0, 0, 200, 200))
        assert result["success"] is True
        assert result["roi_width"]  == 200
        assert result["roi_height"] == 200


# ---------------------------------------------------------------------------
# TestExtractStripRoiFailures
# ---------------------------------------------------------------------------

class TestExtractStripRoiFailures:
    """Tests for all failure / guard paths."""

    def test_none_image_returns_failure(self):
        """None input must fail gracefully without raising."""
        result = extract_strip_roi(None)
        assert result["success"] is False
        assert "reason" in result

    def test_empty_array_returns_failure(self):
        """An empty numpy array must fail gracefully."""
        result = extract_strip_roi(np.array([]))
        assert result["success"] is False
        assert "reason" in result

    def test_grayscale_image_returns_failure(self):
        """A 2-D grayscale image must be rejected."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = extract_strip_roi(gray)
        assert result["success"] is False
        assert "reason" in result

    def test_roi_out_of_bounds_right_fails(self):
        """ROI extending past the right edge must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(100, 50, 250, 150))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_out_of_bounds_bottom_fails(self):
        """ROI extending past the bottom edge must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(50, 100, 150, 250))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_negative_x1_fails(self):
        """ROI with negative x1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(-10, 0, 100, 100))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_negative_y1_fails(self):
        """ROI with negative y1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(0, -10, 100, 100))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_zero_width_fails(self):
        """ROI where x2 == x1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(50, 50, 50, 100))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_zero_height_fails(self):
        """ROI where y2 == y1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(50, 50, 100, 50))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_inverted_x_fails(self):
        """ROI where x2 < x1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(100, 50, 50, 150))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_inverted_y_fails(self):
        """ROI where y2 < y1 must fail."""
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(50, 100, 150, 50))
        assert result["success"] is False
        assert "reason" in result

    def test_tiny_roi_below_min_area_fails(self):
        """
        A 3×3 ROI (9 px², well below MIN_STRIP_AREA_PX=100) must fail.
        """
        image = make_solid_bgr(200, 200, 128, 128, 128)
        result = extract_strip_roi(image, roi=(50, 50, 53, 53))
        assert result["success"] is False
        assert "reason" in result

    def test_failure_has_no_roi_image_key(self):
        """On failure, 'roi_image' must not appear in the result."""
        result = extract_strip_roi(None)
        assert "roi_image" not in result

    def test_failure_reason_is_non_empty_string(self):
        """Every failure must include a non-empty string reason."""
        result = extract_strip_roi(None)
        assert isinstance(result.get("reason"), str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# TestDrawStripRoi
# ---------------------------------------------------------------------------

class TestDrawStripRoi:
    """Tests for the draw_strip_roi visualisation helper."""

    def test_returns_numpy_array(self):
        """draw_strip_roi must return a NumPy ndarray."""
        image = make_standard_card()
        vis = draw_strip_roi(image)
        assert isinstance(vis, np.ndarray)

    def test_output_dtype_is_uint8(self):
        """Output must be uint8."""
        image = make_standard_card()
        vis = draw_strip_roi(image)
        assert vis.dtype == np.uint8

    def test_output_shape_matches_input(self):
        """Output image must have the same shape as the input."""
        image = make_standard_card()
        vis = draw_strip_roi(image)
        assert vis.shape == image.shape

    def test_original_image_not_modified(self):
        """draw_strip_roi must not mutate the original image."""
        image = make_standard_card()
        before = image.copy()
        draw_strip_roi(image)
        np.testing.assert_array_equal(image, before)

    def test_rectangle_pixels_differ_from_original(self):
        """
        The drawn rectangle must change at least some pixels compared to the
        original solid-grey image (i.e. the rectangle was actually drawn).
        """
        image = make_standard_card()
        vis = draw_strip_roi(image)
        # At least one pixel must differ after drawing the coloured rectangle.
        assert not np.array_equal(vis, image), (
            "draw_strip_roi did not change any pixels — rectangle not drawn"
        )

    def test_works_with_roi_result(self):
        """draw_strip_roi must accept a roi_result dict from extract_strip_roi."""
        image = make_standard_card()
        result = extract_strip_roi(image)
        assert result["success"] is True
        vis = draw_strip_roi(image, roi_result=result)
        assert vis.shape == image.shape

    def test_works_with_explicit_roi_tuple(self):
        """draw_strip_roi must accept an explicit (x1, y1, x2, y2) tuple."""
        image = make_standard_card()
        vis = draw_strip_roi(image, roi=(200, 100, 600, 400))
        assert vis.shape == image.shape

    def test_none_image_raises_valueerror(self):
        """Passing None as image must raise ValueError."""
        with pytest.raises(ValueError):
            draw_strip_roi(None)

    def test_grayscale_image_raises_valueerror(self):
        """Passing a 2-D grayscale image must raise ValueError."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        with pytest.raises(ValueError):
            draw_strip_roi(gray)
