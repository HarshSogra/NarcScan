"""
roi.py
======

Phase 5 - Test Strip ROI Extraction

Overview
--------
After Phase 4 produces a grey-reference-corrected image of the NarcScan card,
this module extracts the fixed rectangular region where the physical test strip
will be placed.

Physical card layout
--------------------
The NarcScan card is a plain grey A4 sheet with four ArUco markers at the
corners.  The test strip is placed in a FIXED CENTRAL AREA of the card:

    ┌──────────────────────────────────┐
    │  [ID 0]                  [ID 1]  │
    │                                  │
    │       GREY CALIBRATION REGION    │  ← y = 25 % – 45 %  (Phase 4)
    │                                  │
    │       ┌──────────────────┐       │
    │       │                  │       │
    │       │   TEST STRIP     │       │  ← y = 50 % – 80 %  (Phase 5)
    │       │                  │       │
    │       └──────────────────┘       │
    │                                  │
    │  [ID 3]                  [ID 2]  │
    └──────────────────────────────────┘

Design rationale
----------------
* Coordinates are expressed as FRACTIONS of the corrected image dimensions,
  not hardcoded pixel values.  This makes the module resolution-independent:
  it works regardless of OUTPUT_WIDTH / OUTPUT_HEIGHT from perspective.py.

* All ROI configuration is in ONE place (the STRIP_ROI_* constants below).
  Adjust those constants after testing with the real physical card.

* Phase 3 normalises the card to approximately 1000 × 700 pixels.
  The default fractions produce a strip region of ~400 × 210 pixels at
  that resolution, which is a generous bounding box for a standard lateral
  flow test strip.

* The strip ROI is intentionally FIXED (not auto-detected) for this
  prototype.  Automatic strip detection is out of scope for Phase 5.

Pipeline position
-----------------
  Image
  → ArUco Detection       (Phase 2)
  → Perspective Correction (Phase 3)
  → Grey Calibration      (Phase 4)
  → Test Strip Extraction  (Phase 5)  ← this module
  → CIELAB + CIEDE2000    (Phase 6, future)
"""

import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants — Test Strip ROI (all values are fractions of image dimensions)
# ---------------------------------------------------------------------------
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ADJUST THESE AFTER TESTING WITH THE REAL PHYSICAL CARD               │
# │  They are the ONLY values you need to change to reposition the strip.  │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Horizontal extent (fraction of image width):
#   0.30 … 0.70  →  central 40 % of the card width.
#   Keeps the ROI well away from the left/right marker edges.
#
# Vertical extent (fraction of image height):
#   0.50 … 0.80  →  lower-centre of the card.
#   Sits clearly BELOW the grey calibration region (y = 0.25 – 0.45)
#   and above the bottom marker row.

STRIP_ROI_LEFT_FRAC:   float = 0.30   # left  edge, fraction of image width
STRIP_ROI_RIGHT_FRAC:  float = 0.70   # right edge, fraction of image width
STRIP_ROI_TOP_FRAC:    float = 0.50   # top   edge, fraction of image height
STRIP_ROI_BOTTOM_FRAC: float = 0.80   # bottom edge, fraction of image height

# Minimum ROI area required for a meaningful extraction (pixels²).
MIN_STRIP_AREA_PX: int = 100

# BGR colour used to draw the ROI rectangle in the visualisation helper.
# Bright magenta — clearly visible against both grey and white backgrounds.
ROI_DRAW_COLOUR: Tuple[int, int, int] = (255, 0, 255)   # BGR magenta
ROI_DRAW_THICKNESS: int = 3


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_strip_pixels(
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert the fractional ROI constants to integer pixel coordinates.

    Parameters
    ----------
    image_width  : width  of the corrected image in pixels
    image_height : height of the corrected image in pixels

    Returns
    -------
    (x1, y1, x2, y2)
        Top-left (x1, y1) and exclusive bottom-right (x2, y2) of the ROI.
        Image slicing: image[y1:y2, x1:x2]
    """
    x1 = int(round(STRIP_ROI_LEFT_FRAC   * image_width))
    x2 = int(round(STRIP_ROI_RIGHT_FRAC  * image_width))
    y1 = int(round(STRIP_ROI_TOP_FRAC    * image_height))
    y2 = int(round(STRIP_ROI_BOTTOM_FRAC * image_height))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_strip_roi(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
) -> dict:
    """
    Extract the fixed test-strip region from a perspective-corrected,
    grey-calibration-corrected card image.

    Parameters
    ----------
    image : np.ndarray
        A BGR uint8 image produced by Phase 3 / Phase 4.
        Must be a 3-channel uint8 array (H × W × 3).

    roi : (x1, y1, x2, y2) or None
        Optional pixel-coordinate override.
        x1, y1 = top-left corner (inclusive);
        x2, y2 = bottom-right corner (exclusive).
        If None, coordinates are derived from the STRIP_ROI_*_FRAC constants.

    Returns
    -------
    dict — always contains ``"success"`` (bool).

    On SUCCESS:
        "success"         : True
        "roi_image"       : np.ndarray  — the extracted ROI (copy, not a view)
        "pixel_coords"    : (x1, y1, x2, y2)  — pixel coordinates used
        "norm_coords"     : (left, top, right, bottom)  — normalised fractions
        "roi_width"       : int  — width  of the extracted region in pixels
        "roi_height"      : int  — height of the extracted region in pixels

    On FAILURE:
        "success"         : False
        "reason"          : str  — human-readable description of the problem
    """

    # ------------------------------------------------------------------
    # Guard: image validity
    # ------------------------------------------------------------------
    if image is None or image.size == 0:
        return {"success": False, "reason": "Input image is None or empty."}

    if len(image.shape) != 3 or image.shape[2] != 3:
        return {
            "success": False,
            "reason": (
                f"Expected a 3-channel BGR image, "
                f"got shape {image.shape}."
            ),
        }

    h, w = image.shape[:2]

    # ------------------------------------------------------------------
    # Determine pixel coordinates
    # ------------------------------------------------------------------
    if roi is None:
        x1, y1, x2, y2 = _compute_strip_pixels(w, h)
        # Record the normalised fractions that produced these pixels.
        norm_coords = (
            STRIP_ROI_LEFT_FRAC,
            STRIP_ROI_TOP_FRAC,
            STRIP_ROI_RIGHT_FRAC,
            STRIP_ROI_BOTTOM_FRAC,
        )
    else:
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        # Back-calculate normalised coordinates from the overridden pixels.
        norm_coords = (
            x1 / w,
            y1 / h,
            x2 / w,
            y2 / h,
        )

    # ------------------------------------------------------------------
    # Validate bounds
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

    roi_width  = x2 - x1
    roi_height = y2 - y1

    if roi_width * roi_height < MIN_STRIP_AREA_PX:
        return {
            "success": False,
            "reason": (
                f"ROI area ({roi_width * roi_height} px\u00b2) is below the "
                f"minimum of {MIN_STRIP_AREA_PX} px\u00b2."
            ),
        }

    # ------------------------------------------------------------------
    # Extract ROI — use .copy() so the original image is never modified
    # ------------------------------------------------------------------
    roi_image = image[y1:y2, x1:x2].copy()

    return {
        "success":      True,
        "roi_image":    roi_image,
        "pixel_coords": (x1, y1, x2, y2),
        "norm_coords":  norm_coords,
        "roi_width":    roi_width,
        "roi_height":   roi_height,
    }


def draw_strip_roi(
    image: np.ndarray,
    roi_result: Optional[dict] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    colour: Tuple[int, int, int] = ROI_DRAW_COLOUR,
    thickness: int = ROI_DRAW_THICKNESS,
) -> np.ndarray:
    """
    Draw the test-strip ROI rectangle on a COPY of the image.

    The original image is never modified.

    Parameters
    ----------
    image : np.ndarray
        The source image to draw on (must be a 3-channel BGR uint8 array).

    roi_result : dict or None
        A successful result dict returned by ``extract_strip_roi()``.
        If provided, ``pixel_coords`` from this dict are used.

    roi : (x1, y1, x2, y2) or None
        Explicit pixel coordinates to draw.  Used when ``roi_result`` is None.
        If both ``roi_result`` and ``roi`` are None, the default fractional
        constants are applied to the image dimensions.

    colour : (B, G, R) tuple
        Rectangle colour in OpenCV BGR order.  Default: magenta.

    thickness : int
        Rectangle border thickness in pixels.

    Returns
    -------
    np.ndarray
        A new uint8 BGR image with the ROI rectangle drawn on it.

    Raises
    ------
    ValueError
        If ``image`` is None or not a 3-channel BGR array.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is None or empty.")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel BGR image, got shape {image.shape}."
        )

    # Work on a copy so the caller's original is unchanged.
    vis = image.copy()
    h, w = vis.shape[:2]

    # Resolve coordinates from whichever source was provided.
    if roi_result is not None:
        x1, y1, x2, y2 = roi_result["pixel_coords"]
    elif roi is not None:
        x1, y1, x2, y2 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    else:
        x1, y1, x2, y2 = _compute_strip_pixels(w, h)

    import cv2
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), colour, thickness)
    cv2.putText(
        vis,
        "Test Strip ROI",
        (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        colour,
        2,
    )
    return vis
