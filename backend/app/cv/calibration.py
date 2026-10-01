"""
calibration.py
==============

Phase 4 - Grey Reference Calibration

Overview
--------
After Phase 3 produces a perspective-corrected image of the NarcScan card,
this module estimates and removes illumination colour-imbalance by sampling a
known neutral-grey region on the card and computing per-channel correction
factors.

Physical card assumption
------------------------
The NarcScan card is a PLAIN GREY SHEET with four ArUco markers at the
corners.  There are NO multi-colour calibration patches, no ColorChecker,
and no additional hardware.

The correction region (GREY_ROI_* constants below) is a rectangle in the
CENTRE of the perspective-corrected image.  The centre is chosen because:

  1. It is the area least affected by lens vignetting.
  2. It sits well away from the four marker corners.
  3. In future phases the test-strip ROI will occupy the lower-centre region
     of the card, so the calibration patch is placed in the upper-centre area
     to minimise overlap risk.

Assumption: the plain grey card has approximately uniform reflectance across
this central region.

What this module does mathematically
-------------------------------------
Given a perspective-corrected image I and a neutral-grey reference region R:

  1. Compute per-channel median of R:
         obs_B, obs_G, obs_R   (robust against local defects)

  2. Compute a scalar reference target as the mean of the three medians:
         target = (obs_B + obs_G + obs_R) / 3

     If the illumination were perfectly balanced, all three channels would
     read the same value and all gains would equal 1.0.

  3. Per-channel correction gain:
         gain_B = target / obs_B
         gain_G = target / obs_G
         gain_R = target / obs_R

  4. Apply correction to any pixel:
         pixel_c_corrected = clip(pixel_c * gain_c, 0, 255)

Limitations
-----------
  * This is a prototype grey-reference normalisation, NOT a full ICC-profile
    or spectral colour calibration.
  * It corrects channel imbalance (white balance) but does not account for
    non-linear gamma, lens distortion, or pixel-level vignetting gradients.
  * It assumes uniform reflectance in the sampling region.
  * Designed as a pre-processing step for Phase 6 (CIELAB / CIEDE2000
    comparison against a reference colour database).

Pipeline position
-----------------
  Image -> ArUco Detection -> Perspective Correction -> Grey Calibration -> ...
"""

import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Constants - Grey Reference ROI
# ---------------------------------------------------------------------------
# Expressed as FRACTIONS of the corrected image dimensions so the ROI scales
# automatically with OUTPUT_WIDTH / OUTPUT_HEIGHT from perspective.py.
#
# The ROI is a centred rectangle occupying the upper-middle region of the
# card, well away from:
#   - The marker corners (top/bottom edges)
#   - The left/right edges (potential vignetting)
#   - The lower-centre test-strip area (reserved for Phase 5+)

GREY_ROI_LEFT_FRAC:   float = 0.30   # left edge,   fraction of image width
GREY_ROI_RIGHT_FRAC:  float = 0.70   # right edge,  fraction of image width
GREY_ROI_TOP_FRAC:    float = 0.25   # top edge,    fraction of image height
GREY_ROI_BOTTOM_FRAC: float = 0.45   # bottom edge, fraction of image height

# Minimum ROI area (pixels squared).  Below this the sample is statistically
# meaningless.
MIN_ROI_AREA_PX: int = 100

# Valid range for per-channel correction gains.
# Outside this range the reference measurement is unusable.
GAIN_MIN: float = 0.10   # gain < GAIN_MIN -> channel near black (invalid)
GAIN_MAX: float = 10.0   # gain > GAIN_MAX -> channel near zero  (invalid)

# Valid range for observed per-channel median values.
MIN_VALID_MEDIAN: float = 5.0    # below this: card too dark to calibrate
MAX_VALID_MEDIAN: float = 250.0  # above this: channel is saturated / clipped


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_roi_pixels(
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert fractional ROI constants to integer pixel coordinates.

    Parameters
    ----------
    image_width  : width  of the corrected image in pixels
    image_height : height of the corrected image in pixels

    Returns
    -------
    (x1, y1, x2, y2)
        Top-left and bottom-right corners of the ROI.
        Image slicing: image[y1:y2, x1:x2]
    """
    x1 = int(round(GREY_ROI_LEFT_FRAC   * image_width))
    x2 = int(round(GREY_ROI_RIGHT_FRAC  * image_width))
    y1 = int(round(GREY_ROI_TOP_FRAC    * image_height))
    y2 = int(round(GREY_ROI_BOTTOM_FRAC * image_height))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_grey_reference(
    corrected_image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """
    Sample the grey reference region and compute per-channel correction gains.

    Parameters
    ----------
    corrected_image : np.ndarray
        A BGR uint8 image produced by Phase 3 (perspective.correct_perspective).
        Must be a 3-channel uint8 array.

    roi : (x1, y1, x2, y2) or None
        Optional pixel-coordinate override for the sampling rectangle.
        x1, y1 = top-left corner;  x2, y2 = bottom-right corner (exclusive).
        If None the ROI is derived from the GREY_ROI_*_FRAC constants.

    Returns
    -------
    dict — always contains ``"success"`` (bool).

    On SUCCESS the dict also contains:
        "roi"              : (x1, y1, x2, y2)  -- actual pixel ROI used
        "observed_bgr"     : [float, float, float]  -- median B, G, R
        "reference_target" : float  -- neutral-grey target value
        "gains"            : {"B": float, "G": float, "R": float}
        "quality"          : dict with diagnostic statistics

    On FAILURE the dict also contains:
        "reason"           : str  -- human-readable description of the problem
    """

    # ------------------------------------------------------------------
    # Guard: image validity
    # ------------------------------------------------------------------
    if corrected_image is None or corrected_image.size == 0:
        return {"success": False, "reason": "Input image is None or empty."}

    if len(corrected_image.shape) != 3 or corrected_image.shape[2] != 3:
        return {
            "success": False,
            "reason": (
                f"Expected a 3-channel BGR image, "
                f"got shape {corrected_image.shape}."
            ),
        }

    h, w = corrected_image.shape[:2]

    # ------------------------------------------------------------------
    # Determine ROI in pixel coordinates
    # ------------------------------------------------------------------
    if roi is None:
        x1, y1, x2, y2 = _compute_roi_pixels(w, h)
    else:
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])

    # ------------------------------------------------------------------
    # Validate ROI bounds
    # ------------------------------------------------------------------
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return {
            "success": False,
            "reason": (
                f"ROI ({x1}, {y1}, {x2}, {y2}) extends outside "
                f"the image bounds ({w} x {h} pixels)."
            ),
        }

    if x2 <= x1 or y2 <= y1:
        return {
            "success": False,
            "reason": (
                f"ROI ({x1}, {y1}, {x2}, {y2}) has zero or negative area "
                f"(x2 must be > x1 and y2 must be > y1)."
            ),
        }

    roi_area = (x2 - x1) * (y2 - y1)
    if roi_area < MIN_ROI_AREA_PX:
        return {
            "success": False,
            "reason": (
                f"ROI area ({roi_area} px\u00b2) is below the minimum of "
                f"{MIN_ROI_AREA_PX} px\u00b2.  Increase the ROI size."
            ),
        }

    # ------------------------------------------------------------------
    # Extract patch and compute per-channel medians
    # ------------------------------------------------------------------
    patch = corrected_image[y1:y2, x1:x2]   # shape: (roi_h, roi_w, 3)

    # Median is robust against local defects: dust, minor shadows, creases.
    median_b = float(np.median(patch[:, :, 0]))
    median_g = float(np.median(patch[:, :, 1]))
    median_r = float(np.median(patch[:, :, 2]))
    observed_bgr = [median_b, median_g, median_r]

    # ------------------------------------------------------------------
    # Quality checks on observed values
    # ------------------------------------------------------------------
    channel_names = ("B", "G", "R")
    for ch_name, value in zip(channel_names, observed_bgr):
        if value < MIN_VALID_MEDIAN:
            return {
                "success": False,
                "reason": (
                    f"Channel {ch_name} median ({value:.1f}) is too dark "
                    f"(< {MIN_VALID_MEDIAN}).  "
                    f"Check lighting or card placement."
                ),
            }
        if value > MAX_VALID_MEDIAN:
            return {
                "success": False,
                "reason": (
                    f"Channel {ch_name} median ({value:.1f}) is saturated "
                    f"(> {MAX_VALID_MEDIAN}).  "
                    f"Reduce exposure or check the card surface."
                ),
            }

    # ------------------------------------------------------------------
    # Neutral-grey target and per-channel gain factors
    # ------------------------------------------------------------------
    # target is the mean of all three channel medians.
    # For a perfectly neutral grey, target == median_b == median_g == median_r.
    reference_target = float(np.mean(observed_bgr))

    # gain_c = target / obs_c.
    # Multiplying any pixel's channel c by gain_c removes the colour cast.
    # Division is safe because obs values are > MIN_VALID_MEDIAN (> 5.0).
    gain_b = reference_target / median_b
    gain_g = reference_target / median_g
    gain_r = reference_target / median_r
    gains = {"B": gain_b, "G": gain_g, "R": gain_r}

    # ------------------------------------------------------------------
    # Validate gains are physically meaningful
    # ------------------------------------------------------------------
    for ch_name, gain in gains.items():
        if gain < GAIN_MIN or gain > GAIN_MAX:
            return {
                "success": False,
                "reason": (
                    f"Correction gain for channel {ch_name} ({gain:.4f}) "
                    f"is outside the valid range [{GAIN_MIN}, {GAIN_MAX}].  "
                    f"The grey reference may be incorrect, or the image is "
                    f"severely under- or over-exposed."
                ),
            }

    # ------------------------------------------------------------------
    # Quality / diagnostic statistics
    # ------------------------------------------------------------------
    quality = {
        # Deviation of observed channels from a perfect neutral grey.
        # Lower is better; 0.0 means all three channels read identically.
        "channel_imbalance": float(max(observed_bgr) - min(observed_bgr)),

        # Pixel value standard deviation inside the patch.
        # High std -> large brightness variation (texture, shadow, crease).
        "patch_std":         float(np.std(patch.astype(np.float32))),

        # Mean pixel value across all channels in the patch.
        "patch_mean":        float(np.mean(patch.astype(np.float32))),

        # Area of the sampled patch.
        "roi_area_px":       roi_area,
    }

    return {
        "success":          True,
        "roi":              (x1, y1, x2, y2),
        "observed_bgr":     observed_bgr,
        "reference_target": reference_target,
        "gains":            gains,
        "quality":          quality,
    }


def apply_grey_correction(
    image: np.ndarray,
    calibration_result: dict,
) -> np.ndarray:
    """
    Apply per-channel gain correction from a successful calibration result.

    For each pixel and channel c:
        corrected_c = clip(original_c * gain_c, 0, 255)

    This removes the colour cast introduced by the ambient light source.

    Parameters
    ----------
    image : np.ndarray
        A 3-channel BGR uint8 image to correct.
        Can be the full corrected card image or a smaller ROI crop (e.g. the
        test-strip region in Phase 5+).

    calibration_result : dict
        The dict returned by ``sample_grey_reference()`` when success=True.

    Returns
    -------
    np.ndarray
        A new uint8 BGR image with the colour cast removed.

    Raises
    ------
    ValueError
        If calibration_result does not indicate success, is missing the
        "gains" key, or if the image is invalid.
    """
    if not calibration_result.get("success"):
        raise ValueError(
            "Cannot apply correction: calibration_result indicates failure.  "
            f"Reason: {calibration_result.get('reason', 'unknown')}"
        )

    gains = calibration_result.get("gains")
    if gains is None:
        raise ValueError("calibration_result is missing the 'gains' key.")

    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image, got shape {image.shape}."
        )

    # Work in float32 to avoid integer overflow during multiplication.
    img_float = image.astype(np.float32)

    # OpenCV BGR order: index 0 = B, 1 = G, 2 = R
    img_float[:, :, 0] *= gains["B"]
    img_float[:, :, 1] *= gains["G"]
    img_float[:, :, 2] *= gains["R"]

    return np.clip(img_float, 0.0, 255.0).astype(np.uint8)
