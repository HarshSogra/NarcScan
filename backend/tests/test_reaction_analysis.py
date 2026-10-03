"""
test_reaction_analysis.py
=========================

Phase 7 automated tests for the reaction colour analysis layer.

All tests are deterministic and use synthetic images to verify
robust statistics, region splitting, and Delta-E comparisons.
"""

import sys
import os
import copy
import pytest
import numpy as np
import cv2

# Allow imports of 'app' when running pytest from the backend/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.reaction_analysis import (
    extract_reaction_features,
    ReactionResult,
    LabStats,
    RegionStats,
    DeltaEStats
)

def make_bgr(height: int, width: int, b: int, g: int, r: int) -> np.ndarray:
    """Create a solid-colour BGR uint8 image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

class TestExtractReactionFeaturesValidation:
    """Tests for edge cases and invalid input handling."""
    
    def test_none_image(self):
        """None image should fail gracefully."""
        res = extract_reaction_features(None)
        assert res.success is False
        assert "None" in res.reason

    def test_empty_image(self):
        """Empty image should fail gracefully."""
        res = extract_reaction_features(np.array([]))
        assert res.success is False
        assert "empty" in res.reason

    def test_wrong_channel_count(self):
        """Image with wrong channel count should fail gracefully."""
        gray = np.ones((10, 10), dtype=np.uint8)
        res = extract_reaction_features(gray)
        assert res.success is False
        assert "3-channel" in res.reason
        
    def test_completely_invalid_image(self):
        """Image containing only invalid (black/white) pixels should fail gracefully."""
        img = make_bgr(10, 10, 0, 0, 0)
        res = extract_reaction_features(img)
        assert res.success is False
        assert "No valid pixels" in res.reason

    def test_invalid_region_fractions(self):
        """Invalid region fractions should fail gracefully."""
        img = make_bgr(10, 30, 128, 128, 128)
        res = extract_reaction_features(img, region_left_frac=0.8, region_right_frac=0.2)
        assert res.success is False
        assert "Invalid region fractions" in res.reason
        
    def test_original_roi_not_modified(self):
        """Ensure the original image is not modified by the function."""
        img = make_bgr(10, 30, 128, 128, 128)
        img_copy = copy.deepcopy(img)
        extract_reaction_features(img)
        np.testing.assert_array_equal(img, img_copy)

class TestExtractReactionFeaturesStats:
    """Tests for statistical calculations and valid pixel selection."""
    
    def test_valid_colour_image(self):
        """A valid solid colour image should succeed."""
        img = make_bgr(10, 30, 100, 150, 200)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.valid_pixel_count == 300

    def test_completely_uniform_image(self):
        """For a uniform image, std dev should be 0, and mean == median."""
        img = make_bgr(10, 30, 128, 128, 128)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.L_std == 0.0
        assert res.overall.a_std == 0.0
        assert res.overall.b_std == 0.0
        assert res.overall.L_mean == res.overall.L_median
        
    def test_correct_valid_pixel_count(self):
        """Only valid pixels (not purely black/white) should be counted."""
        img = make_bgr(10, 30, 128, 128, 128)
        img[0, 0] = [0, 0, 0] # black outlier
        img[1, 1] = [255, 255, 255] # white outlier
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.valid_pixel_count == 300 - 2

    def test_median_and_mean_calculation(self):
        """Test median and mean robust calculations."""
        img = make_bgr(10, 30, 128, 128, 128)
        res1 = extract_reaction_features(img)
        
        # Introduce a few valid outliers (not pure black or white)
        img[0:2, 0:2] = [50, 50, 50]
        res2 = extract_reaction_features(img)
        
        # Median should be very stable
        assert abs(res1.overall.L_median - res2.overall.L_median) < 0.1
        # Mean should shift slightly
        assert res1.overall.L_mean != res2.overall.L_mean

    def test_percentile_calculation(self):
        """Test P10 and P90 calculations."""
        img = make_bgr(10, 30, 128, 128, 128)
        img[:, :15] = [100, 100, 100]
        img[:, 15:] = [200, 200, 200]
        res = extract_reaction_features(img)
        assert res.success is True
        
        # We expect two distinct populations of lightness.
        # P10 should be near the darker L*, P90 near the lighter L*
        assert res.overall.L_p10 < res.overall.L_median
        assert res.overall.L_p90 > res.overall.L_median


class TestExtractReactionFeaturesRegionsAndColors:
    """Tests for spatial regions, synthetic colors, and delta E."""

    def test_synthetic_red_region(self):
        """Verify red region produces positive a*."""
        img = make_bgr(10, 30, 0, 0, 200)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.a_median > 20.0

    def test_synthetic_green_region(self):
        """Verify green region produces negative a*."""
        img = make_bgr(10, 30, 0, 200, 0)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.a_median < -20.0

    def test_synthetic_blue_region(self):
        """Verify blue region produces negative b*."""
        img = make_bgr(10, 30, 200, 0, 0)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.overall.b_median < -20.0

    def test_different_colours_in_regions(self):
        """Ensure region splitting correctly isolates colors."""
        img = np.zeros((10, 30, 3), dtype=np.uint8)
        img[:, :10] = [200, 0, 0] # Left = blue
        img[:, 10:20] = [0, 200, 0] # Center = green
        img[:, 20:] = [0, 0, 200] # Right = red
        
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.regions.left.b_median < -20.0 # Blue has negative b*
        assert res.regions.center.a_median < -20.0 # Green has negative a*
        assert res.regions.right.a_median > 20.0 # Red has positive a*
        
    def test_region_splitting(self):
        """Test configuring region boundaries."""
        img = np.zeros((10, 30, 3), dtype=np.uint8)
        img[:, :5] = [200, 0, 0] # Left (5 pixels wide)
        img[:, 5:25] = [0, 200, 0] # Center (20 pixels wide)
        img[:, 25:] = [0, 0, 200] # Right (5 pixels wide)
        
        res = extract_reaction_features(img, region_left_frac=5/30, region_right_frac=25/30)
        assert res.success is True
        assert res.regions.left.valid_pixel_count == 5 * 10
        assert res.regions.center.valid_pixel_count == 20 * 10
        assert res.regions.right.valid_pixel_count == 5 * 10

    def test_identical_regions_delta_e_zero(self):
        """Identical regions should have a dE00 of approximately 0."""
        img = make_bgr(10, 30, 128, 128, 128)
        res = extract_reaction_features(img)
        assert res.success is True
        assert abs(res.delta_e.center_vs_left) < 1e-6
        assert abs(res.delta_e.center_vs_right) < 1e-6
        assert abs(res.delta_e.left_vs_right) < 1e-6

    def test_delta_e_between_different_regions(self):
        """Regions with different colors should yield positive dE00 > 0."""
        img = np.zeros((10, 30, 3), dtype=np.uint8)
        img[:, :10] = [200, 0, 0]
        img[:, 10:20] = [0, 200, 0]
        img[:, 20:] = [0, 0, 200]
        
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.delta_e.center_vs_left > 10.0
        assert res.delta_e.center_vs_right > 10.0
        assert res.delta_e.left_vs_right > 10.0


class TestExtractReactionFeaturesReference:
    """Tests for optional reference color functionality."""

    def test_no_reference_colour(self):
        """If no reference is provided, reference_comparison should be None."""
        img = make_bgr(10, 30, 128, 128, 128)
        res = extract_reaction_features(img)
        assert res.success is True
        assert res.reference_comparison is None

    def test_optional_reference_colour(self):
        """If reference is provided, reference_comparison should be populated."""
        img = make_bgr(10, 30, 128, 128, 128)
        # Assuming the center will be some gray, comparing with white:
        res = extract_reaction_features(img, reference_lab=(100.0, 0.0, 0.0))
        assert res.success is True
        assert res.reference_comparison is not None
        assert res.reference_comparison > 0.0

    def test_invalid_reference_colour(self):
        """An invalid reference colour should result in graceful failure."""
        img = make_bgr(10, 30, 128, 128, 128)
        res = extract_reaction_features(img, reference_lab=(150.0, 0.0, 0.0)) # L* > 100 is invalid
        assert res.success is False
        assert "Invalid reference colour" in res.reason
