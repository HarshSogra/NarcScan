"""
test_calibration.py
===================

Phase 4 automated tests for the grey-reference calibration module.

All tests are deterministic - no real camera image required.

Strategy
--------
We build synthetic solid-colour BGR images whose pixel values are fully
known.  This lets us verify:

  - Correct median sampling
  - Mathematically exact gain calculation
  - Correct application of gains
  - All quality guards and failure paths
"""

import sys
import os

import cv2
import numpy as np
import pytest

# Allow imports of 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.calibration import (
    sample_grey_reference,
    apply_grey_correction,
    _compute_roi_pixels,
    GREY_ROI_LEFT_FRAC,
    GREY_ROI_RIGHT_FRAC,
    GREY_ROI_TOP_FRAC,
    GREY_ROI_BOTTOM_FRAC,
    MIN_ROI_AREA_PX,
    GAIN_MIN,
    GAIN_MAX,
    MIN_VALID_MEDIAN,
    MAX_VALID_MEDIAN,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_solid_bgr(
    height: int,
    width: int,
    b: int,
    g: int,
    r: int,
) -> np.ndarray:
    """
    Create a solid-colour BGR uint8 image.

    Every pixel has exactly the specified B, G, R channel values.
    This makes median / mean calculations fully predictable in tests.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 0] = b
    image[:, :, 1] = g
    image[:, :, 2] = r
    return image


def make_neutral_grey_image(
    height: int = 700,
    width: int = 1000,
    grey_value: int = 128,
) -> np.ndarray:
    """
    Create a perfectly neutral grey BGR image.

    All three channels have the same value, so calibration should produce
    gains of exactly 1.0 for all channels.
    """
    return make_solid_bgr(height, width, grey_value, grey_value, grey_value)


# ---------------------------------------------------------------------------
# TestComputeRoiPixels
# ---------------------------------------------------------------------------

class TestComputeRoiPixels:
    """Unit tests for the internal _compute_roi_pixels helper."""

    def test_roi_inside_image(self):
        """The computed ROI must fit within the image dimensions."""
        w, h = 1000, 700
        x1, y1, x2, y2 = _compute_roi_pixels(w, h)
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h

    def test_roi_fractions_applied_correctly(self):
        """Pixel ROI must match the fractional constants * image size."""
        w, h = 1000, 700
        x1, y1, x2, y2 = _compute_roi_pixels(w, h)
        assert x1 == round(GREY_ROI_LEFT_FRAC   * w)
        assert x2 == round(GREY_ROI_RIGHT_FRAC  * w)
        assert y1 == round(GREY_ROI_TOP_FRAC    * h)
        assert y2 == round(GREY_ROI_BOTTOM_FRAC * h)

    def test_roi_has_positive_area(self):
        """The ROI must always have positive width and height."""
        x1, y1, x2, y2 = _compute_roi_pixels(800, 600)
        assert x2 > x1
        assert y2 > y1


# ---------------------------------------------------------------------------
# TestSampleGreyReferenceSuccess
# ---------------------------------------------------------------------------

class TestSampleGreyReferenceSuccess:
    """Tests for the happy-path of sample_grey_reference."""

    def test_neutral_grey_returns_gains_near_one(self):
        """
        A perfectly neutral grey image (equal B, G, R) must produce
        gains of exactly 1.0 for all channels.
        """
        image = make_neutral_grey_image(grey_value=128)
        result = sample_grey_reference(image)

        assert result["success"] is True
        gains = result["gains"]
        assert abs(gains["B"] - 1.0) < 1e-6
        assert abs(gains["G"] - 1.0) < 1e-6
        assert abs(gains["R"] - 1.0) < 1e-6

    def test_reference_target_equals_grey_value(self):
        """
        For a neutral grey image the reference_target must equal the grey
        channel value itself (because all three channel medians are equal).
        """
        grey_value = 150
        image = make_neutral_grey_image(grey_value=grey_value)
        result = sample_grey_reference(image)

        assert result["success"] is True
        assert abs(result["reference_target"] - grey_value) < 1e-4

    def test_observed_bgr_matches_image_values(self):
        """
        For a solid-colour image the observed BGR medians must equal the
        exact pixel values used to create the image.
        """
        b, g, r = 100, 120, 140
        image = make_solid_bgr(700, 1000, b, g, r)
        result = sample_grey_reference(image)

        assert result["success"] is True
        obs = result["observed_bgr"]
        assert abs(obs[0] - b) < 1e-4, f"Expected B={b}, got {obs[0]}"
        assert abs(obs[1] - g) < 1e-4, f"Expected G={g}, got {obs[1]}"
        assert abs(obs[2] - r) < 1e-4, f"Expected R={r}, got {obs[2]}"

    def test_gains_computed_correctly_for_colour_cast(self):
        """
        Verify the exact gain arithmetic for a known colour-cast image.

        B=80, G=100, R=120  ->  target = (80+100+120)/3 = 100
        gain_B = 100/80 = 1.25
        gain_G = 100/100 = 1.0
        gain_R = 100/120 ≈ 0.8333
        """
        image = make_solid_bgr(700, 1000, b=80, g=100, r=120)
        result = sample_grey_reference(image)

        assert result["success"] is True
        gains = result["gains"]
        expected_target = (80 + 100 + 120) / 3.0
        assert abs(result["reference_target"] - expected_target) < 1e-4
        assert abs(gains["B"] - expected_target / 80)  < 1e-4
        assert abs(gains["G"] - expected_target / 100) < 1e-4
        assert abs(gains["R"] - expected_target / 120) < 1e-4

    def test_result_contains_required_keys_on_success(self):
        """The success result must contain all documented keys."""
        image = make_neutral_grey_image()
        result = sample_grey_reference(image)

        assert result["success"] is True
        for key in ("roi", "observed_bgr", "reference_target", "gains", "quality"):
            assert key in result, f"Missing key: {key}"

    def test_roi_returned_in_result(self):
        """The ROI tuple returned must be (x1, y1, x2, y2) with positive area."""
        image = make_neutral_grey_image(height=700, width=1000)
        result = sample_grey_reference(image)

        assert result["success"] is True
        x1, y1, x2, y2 = result["roi"]
        assert x2 > x1
        assert y2 > y1

    def test_quality_dict_present_and_complete(self):
        """The quality dict must contain the expected diagnostic keys."""
        image = make_neutral_grey_image()
        result = sample_grey_reference(image)

        assert result["success"] is True
        quality = result["quality"]
        for key in ("channel_imbalance", "patch_std", "patch_mean", "roi_area_px"):
            assert key in quality, f"Missing quality key: {key}"

    def test_neutral_grey_has_zero_channel_imbalance(self):
        """
        A perfectly neutral grey image must report channel_imbalance == 0.0.
        """
        image = make_neutral_grey_image(grey_value=128)
        result = sample_grey_reference(image)

        assert result["success"] is True
        assert result["quality"]["channel_imbalance"] == pytest.approx(0.0, abs=1e-4)

    def test_darker_grey_reference(self):
        """
        A darker grey (value=50) must still succeed because 50 > MIN_VALID_MEDIAN.
        Gains should all be 1.0 (neutral grey).
        """
        image = make_neutral_grey_image(grey_value=50)
        result = sample_grey_reference(image)

        assert result["success"] is True
        gains = result["gains"]
        assert abs(gains["B"] - 1.0) < 1e-6
        assert abs(gains["G"] - 1.0) < 1e-6
        assert abs(gains["R"] - 1.0) < 1e-6

    def test_lighter_grey_reference(self):
        """
        A lighter grey (value=200) must succeed (200 < MAX_VALID_MEDIAN=250).
        """
        image = make_neutral_grey_image(grey_value=200)
        result = sample_grey_reference(image)
        assert result["success"] is True

    def test_custom_roi_override(self):
        """
        Passing an explicit ROI tuple must use that region instead of the
        default fractional constants.
        """
        # Build an image where only the top-left 50x50 block differs.
        image = make_neutral_grey_image(grey_value=128)
        # Paint a different grey in a known area (y=10..60, x=10..60)
        # and sample exactly that area.
        image[10:60, 10:60] = 200   # all channels = 200

        result = sample_grey_reference(image, roi=(10, 10, 60, 60))
        assert result["success"] is True
        obs = result["observed_bgr"]
        assert abs(obs[0] - 200) < 1e-4
        assert abs(obs[1] - 200) < 1e-4
        assert abs(obs[2] - 200) < 1e-4


# ---------------------------------------------------------------------------
# TestSampleGreyReferenceFailures
# ---------------------------------------------------------------------------

class TestSampleGreyReferenceFailures:
    """Tests for all failure / quality-guard paths."""

    def test_none_image_returns_failure(self):
        """None input must fail gracefully, not raise."""
        result = sample_grey_reference(None)
        assert result["success"] is False
        assert "reason" in result

    def test_empty_image_returns_failure(self):
        """An empty array must fail gracefully."""
        result = sample_grey_reference(np.array([]))
        assert result["success"] is False
        assert "reason" in result

    def test_grayscale_image_returns_failure(self):
        """A 2-channel or 1-channel image must be rejected."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = sample_grey_reference(gray)
        assert result["success"] is False
        assert "reason" in result

    def test_roi_out_of_bounds_x_returns_failure(self):
        """ROI that extends past the right edge of the image must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(150, 50, 300, 150))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_out_of_bounds_y_returns_failure(self):
        """ROI that extends past the bottom edge of the image must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(50, 150, 150, 300))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_negative_origin_returns_failure(self):
        """ROI with negative x1 or y1 must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(-10, 0, 100, 100))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_zero_area_x_returns_failure(self):
        """ROI where x2 == x1 (zero width) must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(50, 50, 50, 100))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_zero_area_y_returns_failure(self):
        """ROI where y2 == y1 (zero height) must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(50, 50, 100, 50))
        assert result["success"] is False
        assert "reason" in result

    def test_roi_inverted_returns_failure(self):
        """ROI where x2 < x1 must fail."""
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(100, 50, 50, 150))
        assert result["success"] is False
        assert "reason" in result

    def test_tiny_roi_below_min_area_returns_failure(self):
        """
        A ROI whose area is below MIN_ROI_AREA_PX must fail.
        We use a 3x3 = 9 px² region (well below MIN_ROI_AREA_PX=100).
        """
        image = make_neutral_grey_image(height=200, width=200)
        result = sample_grey_reference(image, roi=(50, 50, 53, 53))
        assert result["success"] is False
        assert "reason" in result

    def test_too_dark_image_returns_failure(self):
        """
        An image whose grey patch is below MIN_VALID_MEDIAN must fail.
        We use pixel value 1, well below the threshold of 5.
        """
        image = make_solid_bgr(700, 1000, b=1, g=1, r=1)
        result = sample_grey_reference(image)
        assert result["success"] is False
        assert "reason" in result

    def test_saturated_image_returns_failure(self):
        """
        An image with all channels at 255 (> MAX_VALID_MEDIAN=250) must fail.
        """
        image = make_solid_bgr(700, 1000, b=255, g=255, r=255)
        result = sample_grey_reference(image)
        assert result["success"] is False
        assert "reason" in result

    def test_failure_result_has_no_gains_key(self):
        """On failure, 'gains' must not be present in the result."""
        result = sample_grey_reference(None)
        assert result["success"] is False
        assert "gains" not in result

    def test_failure_result_has_reason(self):
        """Every failure path must include a human-readable 'reason'."""
        result = sample_grey_reference(None)
        assert "reason" in result
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# TestApplyGreyCorrection
# ---------------------------------------------------------------------------

class TestApplyGreyCorrection:
    """Tests for apply_grey_correction."""

    def _make_calibration_result(self, gain_b=1.0, gain_g=1.0, gain_r=1.0):
        """Build a minimal valid calibration result for testing."""
        return {
            "success": True,
            "gains": {"B": gain_b, "G": gain_g, "R": gain_r},
        }

    def test_unity_gains_preserve_image(self):
        """
        Applying gains of 1.0 to all channels must return an image that is
        identical (or within rounding) to the original.
        """
        image = make_solid_bgr(100, 100, b=100, g=150, r=200)
        cal = self._make_calibration_result(1.0, 1.0, 1.0)
        corrected = apply_grey_correction(image, cal)

        assert corrected.shape == image.shape
        assert corrected.dtype == np.uint8
        np.testing.assert_array_equal(corrected, image)

    def test_gains_applied_per_channel(self):
        """
        With known gains, the corrected pixel values must match the expected
        calculation: corrected_c = clip(original_c * gain_c, 0, 255).
        """
        # Solid image: B=80, G=100, R=120
        image = make_solid_bgr(100, 100, b=80, g=100, r=120)
        # Gains chosen to produce exact round numbers after multiplication
        cal = self._make_calibration_result(gain_b=1.25, gain_g=1.0, gain_r=0.5)
        corrected = apply_grey_correction(image, cal)

        expected_b = min(int(80 * 1.25), 255)    # 100
        expected_g = min(int(100 * 1.0), 255)    # 100
        expected_r = min(int(120 * 0.5), 255)    # 60

        assert int(corrected[0, 0, 0]) == expected_b
        assert int(corrected[0, 0, 1]) == expected_g
        assert int(corrected[0, 0, 2]) == expected_r

    def test_overflow_is_clipped_to_255(self):
        """
        A gain that would push a channel above 255 must be clipped to 255,
        not wrap around or raise an exception.
        """
        image = make_solid_bgr(100, 100, b=200, g=200, r=200)
        cal = self._make_calibration_result(gain_b=5.0, gain_g=1.0, gain_r=1.0)
        corrected = apply_grey_correction(image, cal)

        # 200 * 5.0 = 1000 -> clipped to 255
        assert corrected[0, 0, 0] == 255

    def test_output_is_uint8(self):
        """The output dtype must always be uint8."""
        image = make_neutral_grey_image()
        cal = self._make_calibration_result()
        corrected = apply_grey_correction(image, cal)
        assert corrected.dtype == np.uint8

    def test_output_shape_matches_input(self):
        """Output shape must match the input shape exactly."""
        image = make_neutral_grey_image(height=300, width=400)
        cal = self._make_calibration_result()
        corrected = apply_grey_correction(image, cal)
        assert corrected.shape == image.shape

    def test_failed_calibration_raises_valueerror(self):
        """
        Passing a calibration_result with success=False must raise ValueError.
        """
        image = make_neutral_grey_image()
        bad_cal = {"success": False, "reason": "test failure"}
        with pytest.raises(ValueError):
            apply_grey_correction(image, bad_cal)

    def test_missing_gains_key_raises_valueerror(self):
        """
        A calibration_result that is missing the 'gains' key must raise
        ValueError, not KeyError.
        """
        image = make_neutral_grey_image()
        bad_cal = {"success": True}  # no 'gains' key
        with pytest.raises(ValueError):
            apply_grey_correction(image, bad_cal)

    def test_none_image_raises_valueerror(self):
        """Passing None as the image must raise ValueError."""
        cal = self._make_calibration_result()
        with pytest.raises(ValueError):
            apply_grey_correction(None, cal)

    def test_grayscale_image_raises_valueerror(self):
        """Passing a 2D grayscale image must raise ValueError."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        cal = self._make_calibration_result()
        with pytest.raises(ValueError):
            apply_grey_correction(gray, cal)

    def test_roundtrip_correction_neutralises_cast(self):
        """
        If we calibrate from a colour-cast image and then apply the correction
        to the SAME image, the result must be close to a neutral grey
        (all channels approximately equal).
        """
        # Colour-cast: B=80, G=100, R=120
        image = make_solid_bgr(700, 1000, b=80, g=100, r=120)
        cal_result = sample_grey_reference(image)
        assert cal_result["success"] is True

        corrected = apply_grey_correction(image, cal_result)

        # After correction all channels should be approximately equal
        median_b = float(np.median(corrected[:, :, 0]))
        median_g = float(np.median(corrected[:, :, 1]))
        median_r = float(np.median(corrected[:, :, 2]))

        # Allow a tolerance of 2 counts for rounding
        assert abs(median_b - median_g) < 3, (
            f"After correction B={median_b} and G={median_g} should be close"
        )
        assert abs(median_g - median_r) < 3, (
            f"After correction G={median_g} and R={median_r} should be close"
        )
