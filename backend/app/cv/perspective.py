"""
perspective.py
==============

Phase 3 - Perspective Correction / Homography

This module takes the raw ArUco marker detection result (produced by
aruco_detector.py) and uses it to un-warp a perspective-distorted image
of the NarcScan test card back into a flat, rectangular view.

Architecture
------------
aruco_detector.py  ->  detects markers, returns corner coordinates
perspective.py     ->  uses those coordinates to correct perspective

Each module has a single clear responsibility.

Marker layout on the physical card
------------------------------------

  ID 0 =========== ID 1
   |                 |
   |   (test card)   |
   |                 |
  ID 3 =========== ID 2

    ID 0 -> top-left corner of the card
    ID 1 -> top-right corner of the card
    ID 2 -> bottom-right corner of the card
    ID 3 -> bottom-left corner of the card

Why marker OUTER CORNERS?
--------------------------
OpenCV's ArUco detector returns four corner points for each marker in a
consistent clockwise order starting from the top-left corner of the marker:

    index 0 -> top-left  of marker
    index 1 -> top-right of marker
    index 2 -> bottom-right of marker
    index 3 -> bottom-left  of marker

We use the OUTER corner of each marker — the single corner that faces away
from the card center — and map it to the corresponding edge of the output
canvas.  This ensures the full marker is inside the output image.

    Marker ID 0 (top-left card corner)     -> use its top-left     corner (index 0)
    Marker ID 1 (top-right card corner)    -> use its top-right    corner (index 1)
    Marker ID 2 (bottom-right card corner) -> use its bottom-right corner (index 2)
    Marker ID 3 (bottom-left card corner)  -> use its bottom-left  corner (index 3)

Using marker centers mapped to canvas edges caused the outer half of every
marker to be clipped.  Using inner corners mapped to canvas edges had the
opposite problem — it clipped the entire marker off the other side.

Output dimensions
-----------------
The corrected image is always resized to OUTPUT_WIDTH x OUTPUT_HEIGHT.
These constants are defined at module level so they are easy to change.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The fixed output size of the perspective-corrected card image.
# Adjust these to change the resolution of the output.
OUTPUT_WIDTH: int = 1000
OUTPUT_HEIGHT: int = 700

# Which marker ID corresponds to which card corner.
# Changing this dict is the only change needed if the physical layout changes.
CORNER_MARKER_IDS: dict = {
    "top_left":     0,
    "top_right":    1,
    "bottom_right": 2,
    "bottom_left":  3,
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _marker_center(corners: list) -> tuple:
    """
    Calculate the center of a single marker from its four detected corners.

    Parameters
    ----------
    corners : list of 4 points, each point is [x, y]
        The four corner coordinates returned by the ArUco detector for one
        marker.

    Returns
    -------
    (center_x, center_y) as floats.

    How it works:
        center_x = (x0 + x1 + x2 + x3) / 4   (mean of all x values)
        center_y = (y0 + y1 + y2 + y3) / 4   (mean of all y values)

    This is the arithmetic mean - the geometric centroid of the four corners.
    For a convex quadrilateral this always lies inside the shape.
    """
    pts = np.array(corners, dtype=np.float32)   # shape: (4, 2)
    center_x = float(np.mean(pts[:, 0]))
    center_y = float(np.mean(pts[:, 1]))
    return center_x, center_y


def _marker_corner(corners: list, corner_index: int) -> tuple:
    """
    Return a specific corner of a marker by index.

    OpenCV returns ArUco corners in clockwise order starting from top-left:
        index 0 -> top-left  of marker
        index 1 -> top-right of marker
        index 2 -> bottom-right of marker
        index 3 -> bottom-left  of marker

    Parameters
    ----------
    corners      : list of 4 [x, y] points for the marker.
    corner_index : which of the four corners to return (0-3).

    Returns
    -------
    (x, y) as floats.
    """
    pts = np.array(corners, dtype=np.float32)   # shape: (4, 2)
    return float(pts[corner_index, 0]), float(pts[corner_index, 1])


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def correct_perspective(
    image: np.ndarray,
    detection_result: dict,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
) -> dict:
    """
    Apply perspective correction to an image using detected ArUco markers.

    Parameters
    ----------
    image : np.ndarray
        The original BGR (or grayscale) image as a NumPy array - the same
        format that OpenCV cv2.imread() returns.

    detection_result : dict
        The dictionary returned by aruco_detector.detect_markers().
        Expected keys:
            "detected"    : bool
            "markers"     : dict mapping marker_id (int) -> list of 4 corners
            "missing_ids" : list of int

    output_width : int
        Width in pixels of the output (perspective-corrected) image.
        Default: OUTPUT_WIDTH (1000).

    output_height : int
        Height in pixels of the output (perspective-corrected) image.
        Default: OUTPUT_HEIGHT (700).

    Returns
    -------
    dict with the following keys:

        On SUCCESS:
            "success"      : True
            "warped_image" : np.ndarray - the perspective-corrected image,
                             with shape (output_height, output_width, channels)
            "src_points"   : list of 4 [x, y] floats - the source points
                             used (marker centers in the original image)
            "dst_points"   : list of 4 [x, y] floats - the destination
                             rectangle corners in the output image

        On FAILURE (missing markers):
            "success"      : False
            "reason"       : str - human-readable description of why it failed
            "missing_ids"  : list of int - the required marker IDs that were
                             not detected in the image

    Notes
    -----
    - All four required markers (IDs 0, 1, 2, 3) must be present.
      If any are missing the function returns a failure dict instead of
      raising an exception.
    - The function never crashes on missing markers.
    """

    # ------------------------------------------------------------------
    # Step 1 - Check that all four markers are present
    # ------------------------------------------------------------------
    # Perspective correction requires exactly four source points (one per
    # card corner).  If any marker is absent we cannot determine that card
    # corner - attempting to proceed would produce a meaningless result.

    required_ids = list(CORNER_MARKER_IDS.values())   # [0, 1, 2, 3]
    markers      = detection_result.get("markers", {})
    missing_ids  = [mid for mid in required_ids if mid not in markers]

    if missing_ids:
        return {
            "success":     False,
            "reason":      "Missing required ArUco markers: perspective correction requires all four (IDs 0, 1, 2, 3).",
            "missing_ids": missing_ids,
        }

    # ------------------------------------------------------------------
    # Step 2 - Compute the four source points (marker outer corners)
    # ------------------------------------------------------------------
    # We use the OUTER corner of each marker — the corner that faces away
    # from the card interior — and map it to the corresponding canvas edge.
    # This keeps every marker fully visible inside the warped output.
    #
    # OpenCV corner ordering (clockwise from top-left of each marker):
    #   index 0 -> top-left     of marker
    #   index 1 -> top-right    of marker
    #   index 2 -> bottom-right of marker
    #   index 3 -> bottom-left  of marker
    #
    # Card layout mapping (outer corner = corner facing card edge):
    #   ID 0 (card top-left)     -> marker's top-left     corner (index 0)
    #   ID 1 (card top-right)    -> marker's top-right    corner (index 1)
    #   ID 2 (card bottom-right) -> marker's bottom-right corner (index 2)
    #   ID 3 (card bottom-left)  -> marker's bottom-left  corner (index 3)
    #
    # Source point order MUST match destination point order exactly.
    # We use: [top-left, top-right, bottom-right, bottom-left]

    tl = _marker_corner(markers[CORNER_MARKER_IDS["top_left"]],     0)  # top-left     of marker 0
    tr = _marker_corner(markers[CORNER_MARKER_IDS["top_right"]],    1)  # top-right    of marker 1
    br = _marker_corner(markers[CORNER_MARKER_IDS["bottom_right"]], 2)  # bottom-right of marker 2
    bl = _marker_corner(markers[CORNER_MARKER_IDS["bottom_left"]],  3)  # bottom-left  of marker 3

    # Build the source-point array that OpenCV expects: shape (4, 2), float32.
    # Ordering: top-left, top-right, bottom-right, bottom-left
    src_points = np.array([tl, tr, br, bl], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 3 - Define the destination rectangle
    # ------------------------------------------------------------------
    # The destination points describe where each card corner should end up
    # in the output image.  We want a perfect rectangle:
    #
    #   top-left     (0,           0          )
    #   top-right    (width-1,     0          )
    #   bottom-right (width-1,     height-1   )
    #   bottom-left  (0,           height-1   )
    #
    # The -1 accounts for zero-indexing (pixels go from 0 to width-1).

    w = float(output_width  - 1)
    h = float(output_height - 1)

    dst_points = np.array([
        [0.0,  0.0],   # top-left
        [w,    0.0],   # top-right
        [w,    h  ],   # bottom-right
        [0.0,  h  ],   # bottom-left
    ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Step 4 - Compute the homography matrix (H)
    # ------------------------------------------------------------------
    # cv2.getPerspectiveTransform() takes the source and destination
    # 4-point arrays and solves for the unique 3x3 homography matrix H
    # such that:
    #
    #   dst_point = H x src_point    (in homogeneous coordinates)
    #
    # H encodes the combination of rotation, translation, scaling, and
    # perspective tilt that maps the distorted card view to the flat view.
    # It requires EXACTLY 4 point correspondences - no more, no less.

    H = cv2.getPerspectiveTransform(src_points, dst_points)

    # ------------------------------------------------------------------
    # Step 5 - Warp the image
    # ------------------------------------------------------------------
    # cv2.warpPerspective() applies the homography H to every pixel in
    # the source image and writes the result into an output image of
    # size (output_width, output_height).
    #
    # For each pixel in the OUTPUT image, OpenCV:
    #   1. Applies the inverse of H to find where that pixel came from
    #      in the SOURCE image.
    #   2. Samples the source image at that location (with interpolation).
    #   3. Writes the sampled colour into the output pixel.
    #
    # This process is called inverse mapping and avoids holes in the output.

    warped = cv2.warpPerspective(
        image,
        H,
        (output_width, output_height),   # note: (width, height) order
    )

    # ------------------------------------------------------------------
    # Step 6 - Return the result
    # ------------------------------------------------------------------
    return {
        "success":      True,
        "warped_image": warped,
        "src_points":   src_points.tolist(),
        "dst_points":   dst_points.tolist(),
    }
