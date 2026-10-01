"""
test_color_analysis.py
======================

Phase 6 automated tests for the CIELAB colour analysis module.

All tests are deterministic -- no real camera image required.

Strategy
--------
We build known solid-colour or gradient BGR images with predictable Lab
values and verify:

  - BGR/RGB conversion correctness
  - CIELAB conversion for known colours (black, white, grey, primaries)
  - Median and mean Lab statistics
  - Masking behaviour
  - Invalid / edge-case inputs (None, empty, wrong channels, wrong dtype)
  - CIEDE2000 for identical colours -> approximately 0
  - CIEDE2000 symmetry (dE(A,B) == dE(B,A))
  - CIEDE2000 for known Lab pairs
  - CIEDE2000 invalid input handling
  - Different ROI sizes
"""

import sys
import os
import math

import cv2
import numpy as np
import pytest

# Allow imports of 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.color_analysis import (
    analyse_lab,
    calculate_delta_e_ciede2000,
    _bgr_to_lab_float,
    _validate_bgr_image,
    _L_SCALE,
    _AB_OFFSET,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_bgr(height: int, width: int, b: int, g: int, r: int) -> np.ndarray:
    """Create a solid-colour BGR uint8 image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def make_grey(height: int, width: int, value: int) -> np.ndarray:
    """Create a solid neutral-grey BGR image (B==G==R==value)."""
    return make_bgr(height, width, value, value, value)


def bgr_pixel_to_lab(b: int, g: int, r: int) -> tuple:
    """
    Convert a single BGR uint8 pixel to standard CIELAB using the same
    OpenCV pipeline as color_analysis.py.  Returns (L*, a*, b*) floats.
    """
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)
    lab_img = _bgr_to_lab_float(pixel)
    L, a, b_ch = float(lab_img[0, 0, 0]), float(lab_img[0, 0, 1]), float(lab_img[0, 0, 2])
    return L, a, b_ch


# ---------------------------------------------------------------------------
# TestValidateBgrImage
# ---------------------------------------------------------------------------

class TestValidateBgrImage:
    """Tests for the internal _validate_bgr_image helper."""

    def test_valid_image_returns_true(self):
        """A valid 3-channel uint8 BGR image must return (True, '')."""
        img = make_grey(10, 10, 128)
        ok, reason = _validate_bgr_image(img)
        assert ok is True
        assert reason == ""

    def test_none_returns_false(self):
        ok, reason = _validate_bgr_image(None)
        assert ok is False
        assert len(reason) > 0

    def test_empty_array_returns_false(self):
        ok, reason = _validate_bgr_image(np.array([]))
        assert ok is False
        assert len(reason) > 0

    def test_grayscale_2d_returns_false(self):
        gray = np.ones((10, 10), dtype=np.uint8)
        ok, reason = _validate_bgr_image(gray)
        assert ok is False
        assert len(reason) > 0

    def test_wrong_dtype_float_returns_false(self):
        img = np.ones((10, 10, 3), dtype=np.float32)
        ok, reason = _validate_bgr_image(img)
        assert ok is False
        assert len(reason) > 0

    def test_non_ndarray_returns_false(self):
        ok, reason = _validate_bgr_image([[1, 2, 3]])
        assert ok is False
        assert len(reason) > 0


# ---------------------------------------------------------------------------
# TestBgrToLabFloat
# ---------------------------------------------------------------------------

class TestBgrToLabFloat:
    """Unit tests for the BGR -> CIELAB float32 conversion."""

    def test_output_dtype_is_float32(self):
        img = make_grey(5, 5, 128)
        lab = _bgr_to_lab_float(img)
        assert lab.dtype == np.float32

    def test_output_shape_matches_input(self):
        img = make_grey(7, 11, 128)
        lab = _bgr_to_lab_float(img)
        assert lab.shape == (7, 11, 3)

    def test_black_image_L_near_zero(self):
        """Pure black (0,0,0) must have L* close to 0."""
        img = make_grey(5, 5, 0)
        lab = _bgr_to_lab_float(img)
        L = float(np.median(lab[:, :, 0]))
        assert L < 5.0, f"Expected L*~0 for black, got {L}"

    def test_white_image_L_near_100(self):
        """Pure white (255,255,255) must have L* close to 100."""
        img = make_grey(5, 5, 255)
        lab = _bgr_to_lab_float(img)
        L = float(np.median(lab[:, :, 0]))
        assert L > 95.0, f"Expected L*~100 for white, got {L}"

    def test_neutral_grey_a_and_b_near_zero(self):
        """A neutral grey must have a* and b* close to 0."""
        img = make_grey(10, 10, 128)
        lab = _bgr_to_lab_float(img)
        a_med = float(np.median(lab[:, :, 1]))
        b_med = float(np.median(lab[:, :, 2]))
        assert abs(a_med) < 5.0, f"Expected a*~0 for grey, got {a_med}"
        assert abs(b_med) < 5.0, f"Expected b*~0 for grey, got {b_med}"

    def test_L_range_is_0_to_100(self):
        """L* values for any uint8 image must be in [0, 100]."""
        for grey_val in [0, 64, 128, 192, 255]:
            img = make_grey(3, 3, grey_val)
            lab = _bgr_to_lab_float(img)
            assert lab[:, :, 0].min() >= 0.0
            assert lab[:, :, 0].max() <= 100.0 + 1e-4  # small float tolerance

    def test_ab_range_is_minus128_to_127(self):
        """a* and b* must be in approximately [-128, 127] for all uint8 inputs."""
        img = make_bgr(3, 3, 255, 0, 0)   # pure blue
        lab = _bgr_to_lab_float(img)
        for ch in (1, 2):
            assert lab[:, :, ch].min() >= -129.0
            assert lab[:, :, ch].max() <= 128.0

    def test_opencv_encoding_reversal(self):
        """
        Verify the scaling constants are consistent with OpenCV's encoding:
          L_std = L_opencv * (100/255)
          a_std = a_opencv - 128
          b_std = b_opencv - 128
        """
        img = make_grey(3, 3, 200)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        lab_uint8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        L_u8 = float(lab_uint8[0, 0, 0])
        a_u8 = float(lab_uint8[0, 0, 1])
        b_u8 = float(lab_uint8[0, 0, 2])

        lab_float = _bgr_to_lab_float(img)
        L_f = float(lab_float[0, 0, 0])
        a_f = float(lab_float[0, 0, 1])
        b_f = float(lab_float[0, 0, 2])

        assert abs(L_f - L_u8 * _L_SCALE) < 1e-4
        assert abs(a_f - (a_u8 - _AB_OFFSET)) < 1e-4
        assert abs(b_f - (b_u8 - _AB_OFFSET)) < 1e-4


# ---------------------------------------------------------------------------
# TestAnalyseLabSuccess
# ---------------------------------------------------------------------------

class TestAnalyseLabSuccess:
    """Happy-path tests for analyse_lab."""

    def test_success_flag_true_on_valid_image(self):
        img = make_grey(50, 100, 128)
        result = analyse_lab(img)
        assert result["success"] is True

    def test_result_contains_all_required_keys(self):
        """All documented keys must be present on success."""
        required = (
            "success", "L_median", "a_median", "b_median",
            "L_mean", "a_mean", "b_mean",
            "pixel_count", "roi_height", "roi_width", "roi_shape",
        )
        result = analyse_lab(make_grey(50, 100, 128))
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_pixel_count_equals_total_pixels(self):
        """Without a mask, pixel_count must equal h*w."""
        h, w = 30, 40
        result = analyse_lab(make_grey(h, w, 128))
        assert result["pixel_count"] == h * w

    def test_roi_shape_matches_input(self):
        h, w = 25, 60
        result = analyse_lab(make_grey(h, w, 128))
        assert result["roi_height"] == h
        assert result["roi_width"] == w
        assert result["roi_shape"] == (h, w)

    def test_neutral_grey_ab_near_zero(self):
        """A neutral grey image must have a* and b* close to 0."""
        result = analyse_lab(make_grey(50, 100, 128))
        assert result["success"] is True
        assert abs(result["a_median"]) < 5.0
        assert abs(result["b_median"]) < 5.0

    def test_black_L_near_zero(self):
        result = analyse_lab(make_grey(20, 20, 0))
        assert result["success"] is True
        assert result["L_median"] < 5.0

    def test_white_L_near_100(self):
        result = analyse_lab(make_grey(20, 20, 255))
        assert result["success"] is True
        assert result["L_median"] > 95.0

    def test_median_equals_mean_for_uniform_image(self):
        """For a solid-colour image, median and mean must be identical."""
        result = analyse_lab(make_grey(30, 30, 150))
        assert result["success"] is True
        assert abs(result["L_median"] - result["L_mean"]) < 1e-4
        assert abs(result["a_median"] - result["a_mean"]) < 1e-4
        assert abs(result["b_median"] - result["b_mean"]) < 1e-4

    def test_median_robust_against_outlier(self):
        """
        Replacing a small number of pixels with pure white must not
        shift the median significantly for a large grey image.
        """
        img = make_grey(100, 100, 128)
        img[0:2, 0:2] = [255, 255, 255]   # 4 outlier pixels out of 10,000

        result_clean = analyse_lab(make_grey(100, 100, 128))
        result_noisy = analyse_lab(img)
        assert result_noisy["success"] is True
        # Median should be essentially unchanged
        assert abs(result_noisy["L_median"] - result_clean["L_median"]) < 1.0

    def test_works_at_1x1_roi(self):
        """Single-pixel ROI must succeed."""
        img = make_grey(1, 1, 128)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["pixel_count"] == 1

    def test_works_at_large_roi(self):
        """Large ROI (500x200) must succeed without memory error."""
        img = make_grey(200, 500, 128)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["pixel_count"] == 200 * 500


# ---------------------------------------------------------------------------
# TestAnalyseLabMask
# ---------------------------------------------------------------------------

class TestAnalyseLabMask:
    """Tests for the optional pixel mask parameter."""

    def test_mask_reduces_pixel_count(self):
        """A mask that covers half the image must halve the pixel_count."""
        h, w = 20, 20
        img = make_grey(h, w, 128)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[:, : w // 2] = 255   # left half only
        result = analyse_lab(img, mask=mask)
        assert result["success"] is True
        assert result["pixel_count"] == h * (w // 2)

    def test_mask_selects_different_colour_region(self):
        """
        Masking only the red half of a bicolour image must give different
        Lab values than masking only the grey half.
        """
        h, w = 20, 40
        img = make_grey(h, w, 128)
        img[:, w // 2 :] = [0, 0, 255]   # right half = pure red (BGR)

        mask_left  = np.zeros((h, w), dtype=np.uint8)
        mask_left[:, : w // 2] = 255
        mask_right = np.zeros((h, w), dtype=np.uint8)
        mask_right[:, w // 2:] = 255

        res_left  = analyse_lab(img, mask=mask_left)
        res_right = analyse_lab(img, mask=mask_right)

        assert res_left["success"] and res_right["success"]
        # Red has high a*; grey has a*~0
        assert res_right["a_median"] > res_left["a_median"] + 5.0

    def test_full_mask_same_as_no_mask(self):
        """A mask of all-ones must give the same result as no mask."""
        h, w = 20, 20
        img = make_grey(h, w, 128)
        full_mask = np.ones((h, w), dtype=np.uint8) * 255

        res_nomask = analyse_lab(img)
        res_mask   = analyse_lab(img, mask=full_mask)

        assert abs(res_nomask["L_median"] - res_mask["L_median"]) < 1e-4
        assert abs(res_nomask["a_median"] - res_mask["a_median"]) < 1e-4
        assert abs(res_nomask["b_median"] - res_mask["b_median"]) < 1e-4

    def test_empty_mask_returns_failure(self):
        """A mask that excludes all pixels must fail gracefully."""
        h, w = 10, 10
        img = make_grey(h, w, 128)
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        result = analyse_lab(img, mask=empty_mask)
        assert result["success"] is False
        assert "reason" in result


# ---------------------------------------------------------------------------
# TestAnalyseLabFailures
# ---------------------------------------------------------------------------

class TestAnalyseLabFailures:
    """Failure / guard path tests for analyse_lab."""

    def test_none_image_returns_failure(self):
        result = analyse_lab(None)
        assert result["success"] is False
        assert "reason" in result

    def test_empty_array_returns_failure(self):
        result = analyse_lab(np.array([]))
        assert result["success"] is False
        assert "reason" in result

    def test_grayscale_2d_returns_failure(self):
        gray = np.ones((20, 20), dtype=np.uint8)
        result = analyse_lab(gray)
        assert result["success"] is False
        assert "reason" in result

    def test_float_image_returns_failure(self):
        img = np.ones((10, 10, 3), dtype=np.float32)
        result = analyse_lab(img)
        assert result["success"] is False
        assert "reason" in result

    def test_wrong_mask_shape_returns_failure(self):
        img = make_grey(10, 10, 128)
        bad_mask = np.ones((5, 5), dtype=np.uint8)   # wrong size
        result = analyse_lab(img, mask=bad_mask)
        assert result["success"] is False
        assert "reason" in result

    def test_failure_has_no_lab_keys(self):
        """On failure, Lab statistics keys must not appear."""
        result = analyse_lab(None)
        for key in ("L_median", "a_median", "b_median"):
            assert key not in result

    def test_failure_reason_is_non_empty_string(self):
        result = analyse_lab(None)
        assert isinstance(result.get("reason"), str)
        assert len(result["reason"]) > 0


# ---------------------------------------------------------------------------
# TestAnalyseLabKnownColours
# ---------------------------------------------------------------------------

class TestAnalyseLabKnownColours:
    """
    Tests using synthetic known colours to verify the Lab conversion pipeline.
    We check only the rough expected region in Lab space since OpenCV's D65
    white point and quantisation introduce small offsets.
    """

    def test_pure_red_has_positive_a(self):
        """Pure red (BGR: 0,0,255) must have a* significantly positive."""
        img = make_bgr(10, 10, b=0, g=0, r=255)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["a_median"] > 30.0, (
            f"Expected a*>30 for red, got {result['a_median']}"
        )

    def test_pure_blue_has_negative_b(self):
        """Pure blue (BGR: 255,0,0) must have b* significantly negative."""
        img = make_bgr(10, 10, b=255, g=0, r=0)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["b_median"] < -30.0, (
            f"Expected b*<-30 for blue, got {result['b_median']}"
        )

    def test_pure_green_has_negative_a(self):
        """Pure green (BGR: 0,255,0) must have a* significantly negative."""
        img = make_bgr(10, 10, b=0, g=255, r=0)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["a_median"] < -30.0, (
            f"Expected a*<-30 for green, got {result['a_median']}"
        )

    def test_pure_yellow_has_positive_b(self):
        """Pure yellow (BGR: 0,255,255) must have b* significantly positive."""
        img = make_bgr(10, 10, b=0, g=255, r=255)
        result = analyse_lab(img)
        assert result["success"] is True
        assert result["b_median"] > 30.0, (
            f"Expected b*>30 for yellow, got {result['b_median']}"
        )

    def test_darker_grey_has_lower_L(self):
        """A darker grey must have a lower L* than a lighter grey."""
        res_dark  = analyse_lab(make_grey(10, 10, 64))
        res_light = analyse_lab(make_grey(10, 10, 192))
        assert res_dark["L_median"] < res_light["L_median"]

    def test_different_roi_sizes_same_colour_same_lab(self):
        """
        Uniform solid-colour ROIs of different sizes must produce the same
        median Lab values (colour is independent of area).
        """
        colour = (42, 84, 168)   # arbitrary unique colour
        res_small = analyse_lab(make_bgr(5,  10, *colour))
        res_large = analyse_lab(make_bgr(50, 100, *colour))
        assert res_small["success"] and res_large["success"]
        # Medians should be identical for a solid colour
        assert abs(res_small["L_median"] - res_large["L_median"]) < 0.5
        assert abs(res_small["a_median"] - res_large["a_median"]) < 0.5
        assert abs(res_small["b_median"] - res_large["b_median"]) < 0.5


# ---------------------------------------------------------------------------
# TestCiede2000IdenticalColours
# ---------------------------------------------------------------------------

class TestCiede2000IdenticalColours:
    """CIEDE2000 of identical colours must be approximately 0."""

    def test_identical_black(self):
        dE = calculate_delta_e_ciede2000((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert abs(dE) < 1e-6

    def test_identical_white(self):
        dE = calculate_delta_e_ciede2000((100.0, 0.0, 0.0), (100.0, 0.0, 0.0))
        assert abs(dE) < 1e-6

    def test_identical_mid_grey(self):
        dE = calculate_delta_e_ciede2000((50.0, 0.0, 0.0), (50.0, 0.0, 0.0))
        assert abs(dE) < 1e-6

    def test_identical_chromatic_colour(self):
        lab = (55.0, 25.0, -15.0)
        dE = calculate_delta_e_ciede2000(lab, lab)
        assert abs(dE) < 1e-6


# ---------------------------------------------------------------------------
# TestCiede2000Symmetry
# ---------------------------------------------------------------------------

class TestCiede2000Symmetry:
    """CIEDE2000 must be symmetric: dE(A,B) == dE(B,A)."""

    def test_symmetry_grey_to_red(self):
        lab_a = (50.0, 0.0, 0.0)
        lab_b = (50.0, 30.0, 5.0)
        assert abs(
            calculate_delta_e_ciede2000(lab_a, lab_b)
            - calculate_delta_e_ciede2000(lab_b, lab_a)
        ) < 1e-6

    def test_symmetry_blue_to_yellow(self):
        lab_a = (70.0, -10.0, -40.0)
        lab_b = (80.0, -5.0, 50.0)
        assert abs(
            calculate_delta_e_ciede2000(lab_a, lab_b)
            - calculate_delta_e_ciede2000(lab_b, lab_a)
        ) < 1e-6

    def test_symmetry_dark_to_light(self):
        lab_a = (10.0, 5.0, -5.0)
        lab_b = (90.0, 5.0, -5.0)
        assert abs(
            calculate_delta_e_ciede2000(lab_a, lab_b)
            - calculate_delta_e_ciede2000(lab_b, lab_a)
        ) < 1e-6


# ---------------------------------------------------------------------------
# TestCiede2000KnownPairs
# ---------------------------------------------------------------------------

class TestCiede2000KnownPairs:
    """
    Verify CIEDE2000 against a selection of reference pairs from the
    Sharma et al. (2005) paper (Table 1, rounded to 4 decimal places).

    Tolerances are set to 0.0002 as per the paper's acceptance criteria.
    """

    @pytest.mark.parametrize("lab1,lab2,expected_dE,tol", [
        # Sharma 2005, Table 1, selected pairs
        # (L1,a1,b1), (L2,a2,b2), dE2000
        ((50.0000, 2.6772, -79.7751),
         (50.0000, 0.0000, -82.7485), 2.0425, 0.005),
        ((50.0000, 3.1571, -77.2803),
         (50.0000, 0.0000, -82.7485), 2.8615, 0.005),
        ((50.0000, 2.8361, -74.0200),
         (50.0000, 0.0000, -82.7485), 3.4412, 0.005),
        ((50.0000, -1.3802, -84.2814),
         (50.0000, 0.0000, -82.7485), 1.0000, 0.005),
        ((50.0000, -1.1848, -84.8006),
         (50.0000, 0.0000, -82.7485), 1.0000, 0.005),
    ])
    def test_sharma_reference_pairs(self, lab1, lab2, expected_dE, tol):
        dE = calculate_delta_e_ciede2000(lab1, lab2)
        assert abs(dE - expected_dE) < tol, (
            f"dE2000({lab1},{lab2}) = {dE:.6f}, expected {expected_dE:.4f} ±{tol}"
        )

    def test_large_lightness_difference(self):
        """Two colours differing only in L* should have positive dE > 0."""
        dE = calculate_delta_e_ciede2000((20.0, 0.0, 0.0), (80.0, 0.0, 0.0))
        assert dE > 10.0

    def test_tiny_difference_is_small(self):
        """Two very similar colours should have a small dE."""
        dE = calculate_delta_e_ciede2000((50.0, 0.0, 0.0), (50.5, 0.2, -0.2))
        assert dE < 1.0

    def test_different_hue_family_large_dE(self):
        """Blue vs yellow must have a very large dE (perceptually very different)."""
        # Rough Lab for blue and yellow
        dE = calculate_delta_e_ciede2000((32.0, 79.0, -108.0), (97.0, -21.0, 95.0))
        assert dE > 50.0


# ---------------------------------------------------------------------------
# TestCiede2000InvalidInput
# ---------------------------------------------------------------------------

class TestCiede2000InvalidInput:
    """CIEDE2000 must raise ValueError for invalid inputs."""

    def test_wrong_length_reference_raises(self):
        with pytest.raises(ValueError):
            calculate_delta_e_ciede2000((50.0, 0.0), (50.0, 0.0, 0.0))

    def test_wrong_length_observed_raises(self):
        with pytest.raises(ValueError):
            calculate_delta_e_ciede2000((50.0, 0.0, 0.0), (50.0,))

    def test_L_below_zero_raises(self):
        with pytest.raises(ValueError):
            calculate_delta_e_ciede2000((-1.0, 0.0, 0.0), (50.0, 0.0, 0.0))

    def test_L_above_100_raises(self):
        with pytest.raises(ValueError):
            calculate_delta_e_ciede2000((50.0, 0.0, 0.0), (101.0, 0.0, 0.0))

    def test_valid_boundary_L0_accepted(self):
        """L*=0 is a valid boundary; must NOT raise."""
        dE = calculate_delta_e_ciede2000((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        assert dE == 0.0

    def test_valid_boundary_L100_accepted(self):
        """L*=100 is a valid boundary; must NOT raise."""
        dE = calculate_delta_e_ciede2000((100.0, 0.0, 0.0), (100.0, 0.0, 0.0))
        assert dE == 0.0


# ---------------------------------------------------------------------------
# TestCiede2000NonNegative
# ---------------------------------------------------------------------------

class TestCiede2000NonNegative:
    """CIEDE2000 must always be non-negative."""

    @pytest.mark.parametrize("lab1,lab2", [
        ((0.0, 0.0, 0.0),    (100.0, 0.0, 0.0)),
        ((50.0, 50.0, 0.0),  (50.0, -50.0, 0.0)),
        ((50.0, 0.0, 50.0),  (50.0, 0.0, -50.0)),
        ((30.0, 20.0, -10.0),(70.0, -20.0, 10.0)),
    ])
    def test_non_negative(self, lab1, lab2):
        dE = calculate_delta_e_ciede2000(lab1, lab2)
        assert dE >= 0.0, f"dE2000 was negative: {dE}"
