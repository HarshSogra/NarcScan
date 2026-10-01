"""
generate_a4_template.py
=======================

NarcScan Printable A4 Card Template Generator
----------------------------------------------

Generates a print-ready A4 PNG that defines the physical layout of the
NarcScan test card.  The template contains:

  1. A uniform GREY background (the physical grey reference sheet).
  2. Four real DICT_4X4_50 ArUco markers at the card corners.
  3. A clearly outlined TEST STRIP PLACEMENT AREA in the lower-centre.
  4. A GREY CALIBRATION REGION outline in the upper-centre.
  5. A 50 mm scale reference bar (bottom-left corner, outside all ROIs).
  6. Instructional labels.

PRINTING INSTRUCTIONS
---------------------
  * Print at exactly 100% / Actual Size -- do NOT scale to fit the page.
  * In your printer dialog: Page Scaling = "None" / "Actual Size".
  * Verify the 50 mm scale bar with a physical ruler before use.
  * If the bar measures differently, re-print at 100% without scaling.
  * Keep the ArUco markers uncropped -- leave the 15 mm outer margin.

PHYSICAL DIMENSIONS
-------------------
All configurable dimensions are in the CONFIGURATION section below.
Change only those constants; do not scatter values through the code.

OUTPUT
------
  backend/tools/narcscan_a4_template.png

SOFTWARE GEOMETRY
-----------------
The normalised ROI fractions used by Phase 4 (calibration.py) and
Phase 5 (roi.py) are derived from the same physical dimensions used
here.  See the geometry derivation at the bottom of this module.
"""

import os
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CONFIGURATION -- change these to adjust the template layout
# ---------------------------------------------------------------------------
# All measurements in millimetres (mm) unless noted.

# --- Page / resolution ---
PAGE_WIDTH_MM:   float = 210.0   # A4 width
PAGE_HEIGHT_MM:  float = 297.0   # A4 height
DPI:             int   = 300     # print resolution (dots per inch)

# --- ArUco markers ---
MARKER_SIZE_MM:   float = 30.0   # physical side length of each marker square
MARKER_MARGIN_MM: float = 15.0   # gap from page edge to outer marker corner
ARUCO_DICT_ID           = cv2.aruco.DICT_4X4_50
MARKER_IDS              = [0, 1, 2, 3]   # TL, TR, BR, BL -- do NOT change

# --- Grey calibration region (must match calibration.py) ---
# Expressed as fractions of the card interior (between inner marker edges).
# These MUST match GREY_ROI_*_FRAC in backend/app/cv/calibration.py.
GREY_ROI_LEFT_FRAC:   float = 0.30
GREY_ROI_RIGHT_FRAC:  float = 0.70
GREY_ROI_TOP_FRAC:    float = 0.25
GREY_ROI_BOTTOM_FRAC: float = 0.45

# --- Test-strip placement area ---
# Width and height in mm of the physical strip placement rectangle.
STRIP_WIDTH_MM:  float = 80.0   # horizontal extent of the placement box
STRIP_HEIGHT_MM: float = 28.0   # vertical extent  of the placement box

# Centre of the strip as a fraction of the card INTERIOR dimensions.
# 0.50 x = exactly centred horizontally in the card interior.
# 0.60 y = 60% down the card interior (below calibration zone, above bottom markers).
STRIP_CENTRE_X_INTERIOR_FRAC: float = 0.50
STRIP_CENTRE_Y_INTERIOR_FRAC: float = 0.60

# --- Scale reference bar ---
SCALE_BAR_MM:           float = 50.0   # physical length of the printed scale bar
SCALE_BAR_X_MM:         float = 10.0   # distance from left page edge to bar start
SCALE_BAR_Y_MM:         float = 280.0  # distance from top  page edge to bar centre
SCALE_BAR_THICKNESS_MM: float = 2.0

# --- Visual style ---
BG_GREY:          int = 180       # 0-255 grey level for the background
GREY_ROI_COLOUR       = (80, 80, 255)  # BGR -- blue  outline for calibration region
STRIP_COLOUR          = (0, 140, 0)    # BGR -- green outline for strip placement
SCALE_BAR_COLOUR      = (30, 30, 30)   # BGR -- near-black for the scale bar
TEXT_COLOUR           = (20, 20, 20)   # BGR -- dark text
OUTLINE_THICKNESS: int = 4             # pixel thickness for region outlines

# ---------------------------------------------------------------------------
# END OF CONFIGURATION
# ---------------------------------------------------------------------------


def mm_to_px(mm: float) -> int:
    """Convert millimetres to pixels at the configured DPI."""
    return int(round(mm / 25.4 * DPI))


def derive_geometry() -> dict:
    """
    Compute all pixel coordinates from the physical mm configuration above.

    Returns a dict of every pixel coordinate and normalised fraction used
    by both the template generator and the Phase 4/5 software.
    """
    page_w = mm_to_px(PAGE_WIDTH_MM)
    page_h = mm_to_px(PAGE_HEIGHT_MM)

    marker_px = mm_to_px(MARKER_SIZE_MM)
    margin_px = mm_to_px(MARKER_MARGIN_MM)

    # -----------------------------------------------------------------------
    # Marker outer-corner pixel positions
    # -----------------------------------------------------------------------
    # "Outer corner" = the corner of the marker facing away from the card centre.
    # This matches the convention used in perspective.py.

    # ID 0 -- top-left marker:     outer corner = top-left of marker square
    tl_outer = (margin_px, margin_px)
    # ID 1 -- top-right marker:    outer corner = top-right → stored as (x, y)
    tr_outer = (page_w - margin_px - marker_px, margin_px)
    # ID 2 -- bottom-right marker: outer corner = bottom-right → stored as (x, y)
    br_outer = (page_w - margin_px - marker_px, page_h - margin_px - marker_px)
    # ID 3 -- bottom-left marker:  outer corner = bottom-left → stored as (x, y)
    bl_outer = (margin_px, page_h - margin_px - marker_px)

    # -----------------------------------------------------------------------
    # Card interior = area between the INNER edges of the markers
    # -----------------------------------------------------------------------
    interior_x1 = margin_px + marker_px   # inner-left  of TL/BL markers
    interior_y1 = margin_px + marker_px   # inner-top   of TL/TR markers
    interior_x2 = page_w - margin_px      # inner-right of TR/BR markers
    interior_y2 = page_h - margin_px      # inner-bottom of BL/BR markers
    interior_w  = interior_x2 - interior_x1
    interior_h  = interior_y2 - interior_y1

    # -----------------------------------------------------------------------
    # Grey calibration region (page-absolute coordinates)
    # -----------------------------------------------------------------------
    grey_x1 = interior_x1 + int(round(GREY_ROI_LEFT_FRAC   * interior_w))
    grey_x2 = interior_x1 + int(round(GREY_ROI_RIGHT_FRAC  * interior_w))
    grey_y1 = interior_y1 + int(round(GREY_ROI_TOP_FRAC    * interior_h))
    grey_y2 = interior_y1 + int(round(GREY_ROI_BOTTOM_FRAC * interior_h))

    # -----------------------------------------------------------------------
    # Test-strip placement area (page-absolute coordinates)
    # -----------------------------------------------------------------------
    strip_w_px = mm_to_px(STRIP_WIDTH_MM)
    strip_h_px = mm_to_px(STRIP_HEIGHT_MM)

    # Centre the strip on the card INTERIOR (not the full page),
    # so the fractions are symmetric and match the perspective-corrected canvas.
    strip_cx = interior_x1 + int(round(STRIP_CENTRE_X_INTERIOR_FRAC * interior_w))
    strip_cy = interior_y1 + int(round(STRIP_CENTRE_Y_INTERIOR_FRAC * interior_h))

    strip_x1 = strip_cx - strip_w_px // 2
    strip_x2 = strip_x1 + strip_w_px
    strip_y1 = strip_cy - strip_h_px // 2
    strip_y2 = strip_y1 + strip_h_px

    # -----------------------------------------------------------------------
    # Normalised ROI fractions for Phase 5 (roi.py)
    # Derived from the physical template geometry above.
    # Fractions are relative to the card INTERIOR (what perspective.py outputs).
    # -----------------------------------------------------------------------
    strip_left_frac   = (strip_x1 - interior_x1) / interior_w
    strip_right_frac  = (strip_x2 - interior_x1) / interior_w
    strip_top_frac    = (strip_y1 - interior_y1) / interior_h
    strip_bottom_frac = (strip_y2 - interior_y1) / interior_h

    # -----------------------------------------------------------------------
    # Scale bar pixel coordinates
    # -----------------------------------------------------------------------
    sb_x1 = mm_to_px(SCALE_BAR_X_MM)
    sb_x2 = sb_x1 + mm_to_px(SCALE_BAR_MM)
    sb_y  = mm_to_px(SCALE_BAR_Y_MM)
    sb_t  = max(1, mm_to_px(SCALE_BAR_THICKNESS_MM))

    return {
        # Page
        "page_w": page_w,
        "page_h": page_h,
        # Markers
        "marker_px": marker_px,
        "margin_px": margin_px,
        "tl_outer": tl_outer,
        "tr_outer": tr_outer,
        "br_outer": br_outer,
        "bl_outer": bl_outer,
        # Card interior
        "interior_x1": interior_x1,
        "interior_y1": interior_y1,
        "interior_x2": interior_x2,
        "interior_y2": interior_y2,
        "interior_w": interior_w,
        "interior_h": interior_h,
        # Grey calibration region
        "grey_x1": grey_x1,
        "grey_y1": grey_y1,
        "grey_x2": grey_x2,
        "grey_y2": grey_y2,
        # Strip placement
        "strip_x1": strip_x1,
        "strip_y1": strip_y1,
        "strip_x2": strip_x2,
        "strip_y2": strip_y2,
        # Normalised fractions for Phase 5 roi.py
        "strip_left_frac":   strip_left_frac,
        "strip_right_frac":  strip_right_frac,
        "strip_top_frac":    strip_top_frac,
        "strip_bottom_frac": strip_bottom_frac,
        # Scale bar
        "sb_x1": sb_x1,
        "sb_x2": sb_x2,
        "sb_y":  sb_y,
        "sb_t":  sb_t,
    }


def generate_template(output_path: str = None) -> str:
    """
    Generate the A4 NarcScan template PNG and save it.

    Parameters
    ----------
    output_path : str or None
        Full path for the output PNG.  If None, saves to
        tools/narcscan_a4_template.png next to this script.

    Returns
    -------
    str
        The absolute path to the saved PNG file.
    """
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, "narcscan_a4_template.png")

    g = derive_geometry()
    page_w, page_h = g["page_w"], g["page_h"]

    # -----------------------------------------------------------------------
    # 1. Create canvas -- uniform grey background
    # -----------------------------------------------------------------------
    canvas = np.full((page_h, page_w, 3), BG_GREY, dtype=np.uint8)

    # -----------------------------------------------------------------------
    # 2. Draw ArUco markers
    # -----------------------------------------------------------------------
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    marker_px = g["marker_px"]

    # top-left corner (in pixels) of each marker square on the page
    marker_origins = {
        0: g["tl_outer"],
        1: g["tr_outer"],
        2: g["br_outer"],
        3: g["bl_outer"],
    }

    for marker_id in MARKER_IDS:
        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict,
            marker_id,
            marker_px,
        )
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        ox, oy = marker_origins[marker_id]
        canvas[oy: oy + marker_px, ox: ox + marker_px] = marker_bgr

    # -----------------------------------------------------------------------
    # 3. Draw grey calibration region outline
    # -----------------------------------------------------------------------
    cv2.rectangle(
        canvas,
        (g["grey_x1"], g["grey_y1"]),
        (g["grey_x2"] - 1, g["grey_y2"] - 1),
        GREY_ROI_COLOUR,
        OUTLINE_THICKNESS,
    )
    cv2.putText(
        canvas,
        "GREY CALIBRATION REGION",
        (g["grey_x1"] + 10, g["grey_y1"] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        GREY_ROI_COLOUR,
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------------
    # 4. Draw test-strip placement area
    # -----------------------------------------------------------------------
    cv2.rectangle(
        canvas,
        (g["strip_x1"], g["strip_y1"]),
        (g["strip_x2"] - 1, g["strip_y2"] - 1),
        STRIP_COLOUR,
        OUTLINE_THICKNESS,
    )

    # Label above the strip box (outside the ROI, no overlap)
    label = "PLACE TEST STRIP HERE"
    (label_w, _), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    label_x = g["strip_x1"] + (g["strip_x2"] - g["strip_x1"] - label_w) // 2
    label_y = g["strip_y1"] - 14
    cv2.putText(
        canvas,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        STRIP_COLOUR,
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------------
    # 5. Instruction text (below strip box, outside ROI)
    # -----------------------------------------------------------------------
    instruction = "Place strip inside marked area before capture"
    (ins_w, _), _ = cv2.getTextSize(
        instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    )
    ins_x = g["strip_x1"] + (g["strip_x2"] - g["strip_x1"] - ins_w) // 2
    ins_y = g["strip_y2"] + 36
    cv2.putText(
        canvas,
        instruction,
        (ins_x, ins_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        TEXT_COLOUR,
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------------
    # 6. Scale reference bar (bottom-left, well outside all ROIs)
    # -----------------------------------------------------------------------
    sb_t = g["sb_t"]
    # Horizontal filled bar
    cv2.rectangle(
        canvas,
        (g["sb_x1"], g["sb_y"] - sb_t // 2),
        (g["sb_x2"], g["sb_y"] + sb_t // 2),
        SCALE_BAR_COLOUR,
        -1,
    )
    # Left tick
    cv2.line(
        canvas,
        (g["sb_x1"], g["sb_y"] - mm_to_px(3)),
        (g["sb_x1"], g["sb_y"] + mm_to_px(3)),
        SCALE_BAR_COLOUR,
        sb_t,
    )
    # Right tick
    cv2.line(
        canvas,
        (g["sb_x2"], g["sb_y"] - mm_to_px(3)),
        (g["sb_x2"], g["sb_y"] + mm_to_px(3)),
        SCALE_BAR_COLOUR,
        sb_t,
    )
    # Scale label
    sb_label = "<-- {} mm -->".format(int(SCALE_BAR_MM))
    cv2.putText(
        canvas,
        sb_label,
        (g["sb_x1"], g["sb_y"] - mm_to_px(5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        SCALE_BAR_COLOUR,
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------------
    # 7. Print-at-100% reminder in top-centre (in red-ish)
    # -----------------------------------------------------------------------
    reminder = "PRINT AT 100% / ACTUAL SIZE -- do not scale"
    (rem_w, _), _ = cv2.getTextSize(
        reminder, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    rem_x = (page_w - rem_w) // 2
    rem_y = mm_to_px(8)
    cv2.putText(
        canvas,
        reminder,
        (rem_x, rem_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 180),
        2,
        cv2.LINE_AA,
    )

    # -----------------------------------------------------------------------
    # 8. Save
    # -----------------------------------------------------------------------
    cv2.imwrite(output_path, canvas)
    return output_path


def print_geometry_report(g: dict) -> None:
    """Print a human-readable geometry report to stdout."""

    def px_to_mm(px: float) -> float:
        return px / DPI * 25.4

    print()
    print("=" * 62)
    print("  NarcScan A4 Template -- Physical Geometry Report")
    print("=" * 62)
    print("  Page size      : {:.0f} mm x {:.0f} mm  (A4)".format(
        PAGE_WIDTH_MM, PAGE_HEIGHT_MM))
    print("  DPI            : {}".format(DPI))
    print("  Pixel size     : {} x {} px".format(g["page_w"], g["page_h"]))
    print()
    print("  ArUco markers")
    print("    Dictionary   : DICT_4X4_50")
    print("    IDs          : 0=TL  1=TR  2=BR  3=BL")
    print("    Physical size: {:.0f} mm  ({} px)".format(
        MARKER_SIZE_MM, g["marker_px"]))
    print("    Outer margin : {:.0f} mm  ({} px from edge)".format(
        MARKER_MARGIN_MM, g["margin_px"]))
    print()
    print("  Grey calibration region  (matches calibration.py)")
    gx_mm = px_to_mm(g["grey_x2"] - g["grey_x1"])
    gy_mm = px_to_mm(g["grey_y2"] - g["grey_y1"])
    print("    Pixel rect   : ({}, {}) -> ({}, {})".format(
        g["grey_x1"], g["grey_y1"], g["grey_x2"], g["grey_y2"]))
    print("    Physical size: {:.1f} mm x {:.1f} mm".format(gx_mm, gy_mm))
    print("    Norm fracs   : left={}  right={}".format(
        GREY_ROI_LEFT_FRAC, GREY_ROI_RIGHT_FRAC))
    print("                   top={}  bottom={}".format(
        GREY_ROI_TOP_FRAC, GREY_ROI_BOTTOM_FRAC))
    print()
    print("  Test-strip placement area")
    sx_mm = px_to_mm(g["strip_x2"] - g["strip_x1"])
    sy_mm = px_to_mm(g["strip_y2"] - g["strip_y1"])
    print("    Pixel rect   : ({}, {}) -> ({}, {})".format(
        g["strip_x1"], g["strip_y1"], g["strip_x2"], g["strip_y2"]))
    print("    Physical size: {:.1f} mm x {:.1f} mm".format(sx_mm, sy_mm))
    print("    Norm fracs (Phase 5 roi.py):")
    print("      STRIP_ROI_LEFT_FRAC   = {:.4f}".format(g["strip_left_frac"]))
    print("      STRIP_ROI_RIGHT_FRAC  = {:.4f}".format(g["strip_right_frac"]))
    print("      STRIP_ROI_TOP_FRAC    = {:.4f}".format(g["strip_top_frac"]))
    print("      STRIP_ROI_BOTTOM_FRAC = {:.4f}".format(g["strip_bottom_frac"]))
    print()
    print("  Scale reference bar")
    print("    Physical length: {:.0f} mm".format(SCALE_BAR_MM))
    print("    Pixel length   : {} px".format(g["sb_x2"] - g["sb_x1"]))
    print()
    print("  PRINTING INSTRUCTIONS")
    print("  ---------------------")
    print("  1. Open the PNG in your PDF/image viewer.")
    print("  2. In the print dialog, set:")
    print('       Page Scaling  = "None" / "Actual Size" / "100%"')
    print("       Do NOT select 'Fit to page' or 'Shrink to fit'.")
    print("  3. After printing, measure the scale bar with a ruler.")
    print("     It must be exactly {:.0f} mm.  If not, re-print at 100%.".format(
        SCALE_BAR_MM))
    print("  4. Do not crop the page -- the outer grey margin must be kept.")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Geometry derivation note for Phase 5 roi.py
# ---------------------------------------------------------------------------
#
# The normalised fractions below are computed by derive_geometry() and are
# the values that MUST be set in backend/app/cv/roi.py:
#
#   STRIP_ROI_LEFT_FRAC   <- strip_left_frac
#   STRIP_ROI_RIGHT_FRAC  <- strip_right_frac
#   STRIP_ROI_TOP_FRAC    <- strip_top_frac
#   STRIP_ROI_BOTTOM_FRAC <- strip_bottom_frac
#
# These are expressed relative to the perspective-corrected card interior,
# which matches what roi.py operates on (the output of perspective.py).
#
# Run this script with --report to print the values to copy.


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate the printable NarcScan A4 template."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: tools/narcscan_a4_template.png)",
    )
    args = parser.parse_args()

    g = derive_geometry()
    saved = generate_template(args.output)
    print("\n[NarcScan] Template saved to: {}".format(saved))
    print_geometry_report(g)
