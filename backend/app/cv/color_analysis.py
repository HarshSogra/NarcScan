"""
color_analysis.py
=================

Phase 6 - CIELAB Colour Analysis Foundation
--------------------------------------------

This module accepts the extracted test-strip ROI (from Phase 5) and performs
colour-space conversion and robust statistical analysis in CIELAB colour space.

CIELAB Colour Space
-------------------
CIELAB (L*a*b*) is a perceptually uniform colour space defined by the
International Commission on Illumination (CIE).  Its axes are:

  L*  Lightness       0 = black,  100 = white
  a*  Green-Red       negative = green,  positive = red/magenta
  b*  Blue-Yellow     negative = blue,   positive = yellow

CIELAB is preferred for colorimetric analysis because equal numerical
differences correspond to approximately equal perceived colour differences.

OpenCV CIELAB Encoding
-----------------------
OpenCV's cv2.COLOR_RGB2Lab (and COLOR_BGR2Lab) maps the standard floating-
point CIELAB range into uint8 as follows:

  L*  standard range  [0, 100]   -> OpenCV uint8 range [0, 255]
      scaling: L_opencv = L* * 255 / 100

  a*  standard range  [-128, 127] -> OpenCV uint8 range [0, 255]
      offset: a_opencv = a* + 128

  b*  standard range  [-128, 127] -> OpenCV uint8 range [0, 255]
      offset: b_opencv = b* + 128

When computing statistics from uint8 Lab images, convert back to standard
CIELAB values using:

  L* = L_opencv * 100.0 / 255.0
  a* = a_opencv - 128.0
  b* = b_opencv - 128.0

This module performs that conversion internally and always reports values in
standard CIELAB units.

CIEDE2000
---------
The module also provides a reusable CIEDE2000 colour-difference function.
CIEDE2000 (Delta-E 2000, or dE00) is the current standard for measuring
perceptual colour difference between two Lab colours.

  dE00 < 1    :  difference not visible to the human eye
  dE00 1–2    :  barely perceptible
  dE00 2–10   :  clearly perceptible
  dE00 > 10   :  very large difference

IMPORTANT: The reference Lab colour for CIEDE2000 comparison must be supplied
explicitly by the caller.  This module does NOT hardcode any chemical reaction
reference colour, nor does it make Positive/Negative classification decisions.

Pipeline position
-----------------
  Image
  -> ArUco Detection       (Phase 2)
  -> Perspective Correction (Phase 3)
  -> Grey Calibration      (Phase 4)
  -> Test Strip Extraction  (Phase 5)
  -> CIELAB Analysis        (Phase 6)  <- this module

Future phases
-------------
  Positive/Negative classification, drug identification, and comparison
  against a reference colour database are NOT implemented here.
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# OpenCV Lab encoding constants
# ---------------------------------------------------------------------------
# See module docstring for derivation.

_L_SCALE:  float = 100.0 / 255.0   # multiply OpenCV L channel by this
_AB_OFFSET: float = 128.0           # subtract from OpenCV a/b channels


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bgr_to_lab_float(bgr_image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to standard CIELAB float32.

    Steps
    -----
    1. BGR uint8  ->  RGB uint8           (cv2.COLOR_BGR2RGB)
    2. RGB uint8  ->  Lab uint8 (OpenCV)  (cv2.COLOR_RGB2Lab)
    3. Lab uint8  ->  Lab float32 (std)   (apply _L_SCALE and _AB_OFFSET)

    Parameters
    ----------
    bgr_image : np.ndarray
        H x W x 3 uint8 BGR image.

    Returns
    -------
    np.ndarray
        H x W x 3 float32 array with standard CIELAB values:
        channel 0 = L* in [0, 100]
        channel 1 = a* in [-128, 127]
        channel 2 = b* in [-128, 127]
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    lab_uint8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
    lab = lab_uint8.astype(np.float32)
    lab[:, :, 0] *= _L_SCALE    # L: [0,255] -> [0,100]
    lab[:, :, 1] -= _AB_OFFSET  # a: [0,255] -> [-128,127]
    lab[:, :, 2] -= _AB_OFFSET  # b: [0,255] -> [-128,127]
    return lab


def _validate_bgr_image(image) -> Tuple[bool, str]:
    """
    Validate that ``image`` is a non-empty 3-channel uint8 BGR array.

    Returns
    -------
    (True, "")          if valid
    (False, reason_str) if invalid
    """
    if image is None:
        return False, "Input image is None."
    if not isinstance(image, np.ndarray):
        return False, "Input image is not a numpy ndarray."
    if image.size == 0:
        return False, "Input image is empty (size == 0)."
    if image.ndim != 3 or image.shape[2] != 3:
        return False, (
            "Expected a 3-channel BGR image, got shape {}.".format(image.shape)
        )
    if image.dtype != np.uint8:
        return False, (
            "Expected uint8 image, got dtype {}.".format(image.dtype)
        )
    return True, ""


# ---------------------------------------------------------------------------
# Public API — CIELAB analysis
# ---------------------------------------------------------------------------

def analyse_lab(
    roi_image: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute robust CIELAB colour statistics for a test-strip ROI.

    Parameters
    ----------
    roi_image : np.ndarray
        A BGR uint8 image of the extracted test-strip region (the
        ``roi_image`` value returned by Phase 5's ``extract_strip_roi``).
        Must be a 3-channel uint8 array (H x W x 3).

    mask : np.ndarray or None
        Optional boolean or uint8 mask of the same H x W shape.
        Non-zero pixels are included; zero pixels are excluded.
        If None, all pixels are included.

    Returns
    -------
    dict
        Always contains ``"success"`` (bool).

    On SUCCESS:
        "success"         : True
        "L_median"        : float  -- robust median L* value
        "a_median"        : float  -- robust median a* value
        "b_median"        : float  -- robust median b* value
        "L_mean"          : float  -- mean L* value
        "a_mean"          : float  -- mean a* value
        "b_mean"          : float  -- mean b* value
        "pixel_count"     : int    -- number of pixels included in statistics
        "roi_height"      : int    -- height of the input ROI in pixels
        "roi_width"       : int    -- width  of the input ROI in pixels
        "roi_shape"       : tuple  -- (height, width) of the input ROI

    On FAILURE:
        "success"         : False
        "reason"          : str  -- human-readable description of the problem
    """
    valid, reason = _validate_bgr_image(roi_image)
    if not valid:
        return {"success": False, "reason": reason}

    h, w = roi_image.shape[:2]

    # ------------------------------------------------------------------
    # Validate mask shape (if provided)
    # ------------------------------------------------------------------
    if mask is not None:
        if not isinstance(mask, np.ndarray):
            return {"success": False,
                    "reason": "Mask must be a numpy ndarray or None."}
        if mask.shape[:2] != (h, w):
            return {
                "success": False,
                "reason": (
                    "Mask shape {} does not match ROI shape {}.".format(
                        mask.shape[:2], (h, w)
                    )
                ),
            }

    # ------------------------------------------------------------------
    # Convert BGR -> standard CIELAB float32
    # ------------------------------------------------------------------
    lab_float = _bgr_to_lab_float(roi_image)   # (H, W, 3) float32

    # ------------------------------------------------------------------
    # Apply mask
    # ------------------------------------------------------------------
    if mask is not None:
        flat_mask = mask.flatten().astype(bool)
        pixels = lab_float.reshape(-1, 3)[flat_mask]
    else:
        pixels = lab_float.reshape(-1, 3)       # (N, 3)

    pixel_count = len(pixels)
    if pixel_count == 0:
        return {
            "success": False,
            "reason": "No valid pixels after applying mask.",
        }

    # ------------------------------------------------------------------
    # Compute robust statistics
    # ------------------------------------------------------------------
    # Median is preferred as the primary measurement (robust against
    # outliers such as edge artefacts or dust on the test strip).
    L_vals = pixels[:, 0]
    a_vals = pixels[:, 1]
    b_vals = pixels[:, 2]

    L_median = float(np.median(L_vals))
    a_median = float(np.median(a_vals))
    b_median = float(np.median(b_vals))

    L_mean   = float(np.mean(L_vals))
    a_mean   = float(np.mean(a_vals))
    b_mean   = float(np.mean(b_vals))

    return {
        "success":      True,
        "L_median":     L_median,
        "a_median":     a_median,
        "b_median":     b_median,
        "L_mean":       L_mean,
        "a_mean":       a_mean,
        "b_mean":       b_mean,
        "pixel_count":  pixel_count,
        "roi_height":   h,
        "roi_width":    w,
        "roi_shape":    (h, w),
    }


# ---------------------------------------------------------------------------
# Public API -- CIEDE2000
# ---------------------------------------------------------------------------

def calculate_delta_e_ciede2000(
    lab_reference: Tuple[float, float, float],
    lab_observed:  Tuple[float, float, float],
) -> float:
    """
    Calculate the CIEDE2000 perceptual colour difference between two Lab colours.

    CIEDE2000 is the current CIE standard for colour difference measurement.
    It is more perceptually uniform than older metrics (dE76, dE94).

    IMPORTANT
    ---------
    The reference Lab colour (``lab_reference``) must be supplied explicitly
    by the caller.  This function does NOT hardcode any chemical reaction
    colour or make Positive/Negative classification decisions.

    The grey reference sheet is used for illumination normalisation (Phase 4)
    and is NOT automatically the chemical reaction reference colour.

    Parameters
    ----------
    lab_reference : (L*, a*, b*)
        Reference colour in standard CIELAB units:
        L* in [0, 100], a* in [-128, 127], b* in [-128, 127].

    lab_observed : (L*, a*, b*)
        Observed colour in standard CIELAB units.

    Returns
    -------
    float
        dE00 -- the CIEDE2000 colour difference.
        0.0  -> identical colours.
        > 0  -> increasing perceptual difference.

    Raises
    ------
    ValueError
        If either Lab tuple does not have exactly 3 elements, or if L*
        is outside [0, 100].

    References
    ----------
    Sharma, G., Wu, W., & Dalal, E. N. (2005).
    "The CIEDE2000 color-difference formula: Implementation notes,
    supplementary test data, and mathematical observations."
    Color Research & Application, 30(1), 21-30.
    """
    if len(lab_reference) != 3 or len(lab_observed) != 3:
        raise ValueError(
            "Each Lab colour must be a 3-element tuple (L*, a*, b*)."
        )

    L1, a1, b1 = float(lab_reference[0]), float(lab_reference[1]), float(lab_reference[2])
    L2, a2, b2 = float(lab_observed[0]),  float(lab_observed[1]),  float(lab_observed[2])

    if not (0.0 <= L1 <= 100.0) or not (0.0 <= L2 <= 100.0):
        raise ValueError(
            "L* must be in the range [0, 100].  "
            "Got L1={}, L2={}.".format(L1, L2)
        )

    return _ciede2000(L1, a1, b1, L2, a2, b2)


def _ciede2000(
    L1: float, a1: float, b1: float,
    L2: float, a2: float, b2: float,
) -> float:
    """
    Core CIEDE2000 implementation.

    Follows the Sharma et al. (2005) formulation exactly.
    Variable names match the paper for auditability.
    """
    # --- Step 1: Compute C*ab and h*ab for each sample ---
    C1ab = math.sqrt(a1 ** 2 + b1 ** 2)
    C2ab = math.sqrt(a2 ** 2 + b2 ** 2)

    C_avg_ab7 = ((C1ab + C2ab) / 2.0) ** 7
    G = 0.5 * (1.0 - math.sqrt(C_avg_ab7 / (C_avg_ab7 + 25.0 ** 7)))

    a1p = a1 * (1.0 + G)
    a2p = a2 * (1.0 + G)

    C1p = math.sqrt(a1p ** 2 + b1 ** 2)
    C2p = math.sqrt(a2p ** 2 + b2 ** 2)

    def _h(ap: float, b: float) -> float:
        """Hue angle in degrees [0, 360)."""
        if ap == 0.0 and b == 0.0:
            return 0.0
        angle = math.degrees(math.atan2(b, ap))
        return angle if angle >= 0.0 else angle + 360.0

    h1p = _h(a1p, b1)
    h2p = _h(a2p, b2)

    # --- Step 2: Delta L', Delta C', Delta H' ---
    dLp = L2 - L1
    dCp = C2p - C1p

    # Delta h' (hue difference)
    if C1p * C2p == 0.0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dhp = h2p - h1p
    elif h2p - h1p > 180.0:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0

    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    # --- Step 3: Compute CIEDE2000 ---
    Lp_avg = (L1 + L2) / 2.0
    Cp_avg = (C1p + C2p) / 2.0

    # Average hue h'_bar
    if C1p * C2p == 0.0:
        hp_avg = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        hp_avg = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        hp_avg = (h1p + h2p + 360.0) / 2.0
    else:
        hp_avg = (h1p + h2p - 360.0) / 2.0

    # Weighting functions
    T = (
        1.0
        - 0.17 * math.cos(math.radians(hp_avg - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * hp_avg))
        + 0.32 * math.cos(math.radians(3.0 * hp_avg + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * hp_avg - 63.0))
    )

    SL = 1.0 + 0.015 * (Lp_avg - 50.0) ** 2 / math.sqrt(20.0 + (Lp_avg - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_avg
    SH = 1.0 + 0.015 * Cp_avg * T

    Cp_avg7 = Cp_avg ** 7
    RC = 2.0 * math.sqrt(Cp_avg7 / (Cp_avg7 + 25.0 ** 7))

    d_theta = 30.0 * math.exp(-((hp_avg - 275.0) / 25.0) ** 2)
    RT = -math.sin(math.radians(2.0 * d_theta)) * RC

    KL = KC = KH = 1.0   # parametric factors (unity for standard conditions)

    delta_e = math.sqrt(
        (dLp / (KL * SL)) ** 2
        + (dCp / (KC * SC)) ** 2
        + (dHp / (KH * SH)) ** 2
        + RT * (dCp / (KC * SC)) * (dHp / (KH * SH))
    )

    return delta_e
