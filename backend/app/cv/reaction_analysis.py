"""
reaction_analysis.py
====================

Phase 7 - Test-Reaction Colour Analysis & Feature Extraction
------------------------------------------------------------

This module extracts numerical colour features from the test-strip ROI 
produced in Phase 5. It uses the CIELAB colour space and CIEDE2000 
colour difference formula (from Phase 6) to quantify the reaction.

IMPORTANT SCIENTIFIC LIMITATION
-------------------------------
CIELAB (L*a*b*) and CIEDE2000 (ΔE₀₀) quantify observed colour differences.
They do NOT by themselves establish the identity or presence of a drug.
Actual classification (Positive/Negative/Inconclusive) requires experimentally 
collected and labelled test-strip samples under controlled conditions.
This module extracts FEATURES ONLY and makes NO classification decisions.

Pipeline position
-----------------
  Image
  -> ArUco Detection       (Phase 2)
  -> Perspective Correction (Phase 3)
  -> Grey Calibration      (Phase 4)
  -> Test Strip Extraction  (Phase 5)
  -> CIELAB Analysis        (Phase 6)
  -> Reaction Analysis      (Phase 7)  <- this module
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import cv2

from app.cv.color_analysis import (
    _bgr_to_lab_float,
    _validate_bgr_image,
    calculate_delta_e_ciede2000,
)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class LabStats:
    """Statistical summary of CIELAB values for a specific region."""
    L_median: float
    a_median: float
    b_median: float
    L_mean: float
    a_mean: float
    b_mean: float
    L_std: float
    a_std: float
    b_std: float
    L_p10: float
    L_p90: float
    a_p10: float
    a_p90: float
    b_p10: float
    b_p90: float
    valid_pixel_count: int

@dataclass
class RegionStats:
    """Statistics for the spatial subdivisions of the test strip."""
    left: LabStats
    center: LabStats
    right: LabStats

@dataclass
class DeltaEStats:
    """CIEDE2000 colour difference between regions."""
    center_vs_left: float
    center_vs_right: float
    left_vs_right: float

@dataclass
class ReactionResult:
    """Structured result containing all extracted reaction features."""
    success: bool
    reason: str
    overall: Optional[LabStats] = None
    regions: Optional[RegionStats] = None
    delta_e: Optional[DeltaEStats] = None
    reference_comparison: Optional[float] = None
    
@dataclass
class TemporalObservation:
    """Data structure for future kinetic analysis over time."""
    timestamp: float
    L: float
    a: float
    b: float


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _get_valid_mask(bgr_image: np.ndarray) -> np.ndarray:
    """
    Select valid pixels from the ROI.
    
    Filtering logic:
    We remove purely black (0,0,0) and purely white (255,255,255) pixels.
    Reason: These extreme values are typically caused by sensor clipping, 
    glare, or out-of-bounds padding from prior transformations. They contain 
    no useful chromatic information and could skew the mean/std calculations.
    
    Parameters
    ----------
    bgr_image : np.ndarray
        H x W x 3 uint8 BGR image.
        
    Returns
    -------
    np.ndarray
        H x W boolean mask (True for valid pixels).
    """
    sum_rgb = bgr_image.astype(np.int32).sum(axis=2)
    # Valid pixels are those that are not completely black and not completely white.
    return (sum_rgb > 0) & (sum_rgb < 255 * 3)


def _compute_lab_stats(lab_pixels: np.ndarray) -> Optional[LabStats]:
    """
    Compute robust statistics for an array of valid CIELAB pixels.
    
    Parameters
    ----------
    lab_pixels : np.ndarray
        N x 3 float32 array of CIELAB pixels.
        
    Returns
    -------
    LabStats or None if there are no pixels.
    """
    count = len(lab_pixels)
    if count == 0:
        return None
        
    L_vals = lab_pixels[:, 0]
    a_vals = lab_pixels[:, 1]
    b_vals = lab_pixels[:, 2]

    return LabStats(
        L_median=float(np.median(L_vals)),
        a_median=float(np.median(a_vals)),
        b_median=float(np.median(b_vals)),
        L_mean=float(np.mean(L_vals)),
        a_mean=float(np.mean(a_vals)),
        b_mean=float(np.mean(b_vals)),
        L_std=float(np.std(L_vals)),
        a_std=float(np.std(a_vals)),
        b_std=float(np.std(b_vals)),
        L_p10=float(np.percentile(L_vals, 10)),
        L_p90=float(np.percentile(L_vals, 90)),
        a_p10=float(np.percentile(a_vals, 10)),
        a_p90=float(np.percentile(a_vals, 90)),
        b_p10=float(np.percentile(b_vals, 10)),
        b_p90=float(np.percentile(b_vals, 90)),
        valid_pixel_count=count
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_reaction_features(
    roi_image: np.ndarray,
    region_left_frac: float = 0.33,
    region_right_frac: float = 0.67,
    reference_lab: Optional[Tuple[float, float, float]] = None
) -> ReactionResult:
    """
    Extract colour and regional features from the test-strip ROI.
    
    This function divides the strip horizontally into three regions (left, center, right),
    calculates robust CIELAB statistics for the overall ROI and each region,
    and computes the CIEDE2000 differences between the regions.
    
    Parameters
    ----------
    roi_image : np.ndarray
        A BGR uint8 image of the extracted test-strip region.
    region_left_frac : float
        Fraction of the width where the left region ends and center begins.
    region_right_frac : float
        Fraction of the width where the center region ends and right begins.
    reference_lab : tuple of floats, optional
        A reference (L*, a*, b*) colour to compare the center region against.
        If provided, the CIEDE2000 difference is included in the result.
        
    Returns
    -------
    ReactionResult
        A structured dataclass containing the extracted features.
    """
    # 1. Validate Input
    valid, reason = _validate_bgr_image(roi_image)
    if not valid:
        return ReactionResult(success=False, reason=reason)
        
    h, w = roi_image.shape[:2]
    if region_left_frac >= region_right_frac or region_left_frac < 0.0 or region_right_frac > 1.0:
        return ReactionResult(success=False, reason="Invalid region fractions.")

    # 2. Get Valid Pixels & Convert to CIELAB
    valid_mask = _get_valid_mask(roi_image)
    lab_float = _bgr_to_lab_float(roi_image)
    
    # 3. Overall Statistics
    overall_pixels = lab_float[valid_mask]
    overall_stats = _compute_lab_stats(overall_pixels)
    if overall_stats is None:
        return ReactionResult(success=False, reason="No valid pixels found in the ROI.")
        
    # 4. Regional Statistics
    left_bound = int(round(w * region_left_frac))
    right_bound = int(round(w * region_right_frac))
    
    # Create masks for regions
    left_mask = np.zeros((h, w), dtype=bool)
    left_mask[:, :left_bound] = True
    left_mask &= valid_mask
    
    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[:, left_bound:right_bound] = True
    center_mask &= valid_mask
    
    right_mask = np.zeros((h, w), dtype=bool)
    right_mask[:, right_bound:] = True
    right_mask &= valid_mask
    
    left_stats = _compute_lab_stats(lab_float[left_mask])
    center_stats = _compute_lab_stats(lab_float[center_mask])
    right_stats = _compute_lab_stats(lab_float[right_mask])
    
    if left_stats is None or center_stats is None or right_stats is None:
        return ReactionResult(success=False, reason="One or more regions contain no valid pixels.")
        
    regions = RegionStats(left=left_stats, center=center_stats, right=right_stats)
    
    # 5. Delta-E Comparisons (using median Lab values)
    center_lab = (center_stats.L_median, center_stats.a_median, center_stats.b_median)
    left_lab = (left_stats.L_median, left_stats.a_median, left_stats.b_median)
    right_lab = (right_stats.L_median, right_stats.a_median, right_stats.b_median)
    
    delta_e = DeltaEStats(
        center_vs_left=calculate_delta_e_ciede2000(center_lab, left_lab),
        center_vs_right=calculate_delta_e_ciede2000(center_lab, right_lab),
        left_vs_right=calculate_delta_e_ciede2000(left_lab, right_lab)
    )
    
    # 6. Optional Reference Comparison
    ref_comparison = None
    if reference_lab is not None:
        try:
            ref_comparison = calculate_delta_e_ciede2000(reference_lab, center_lab)
        except ValueError as e:
            return ReactionResult(success=False, reason=f"Invalid reference colour: {e}")
            
    return ReactionResult(
        success=True,
        reason="",
        overall=overall_stats,
        regions=regions,
        delta_e=delta_e,
        reference_comparison=ref_comparison
    )
