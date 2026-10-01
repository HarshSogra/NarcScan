"""
validate_template.py
====================

NarcScan A4 Template Validation Utility
----------------------------------------

This utility:
  1. Generates the A4 template PNG.
  2. Saves it to a clearly documented location.
  3. Prints the physical and pixel dimensions.
  4. Displays the calculated normalised coordinates for:
       - Grey calibration region
       - Test-strip placement region
  5. Provides instructions for printing and verifying the template.

IMPORTANT
---------
Do not consider the physical strip dimensions FINAL until the printed
template has been inspected with a ruler and the ArUco markers confirmed
machine-readable.

Usage
-----
From the backend/ directory:

    python tools/validate_template.py

Or with a custom output path:

    python tools/validate_template.py --output /path/to/template.png
"""

import os
import sys
import argparse

# Make sure we can import the generator regardless of CWD.
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS_DIR)

from generate_a4_template import (
    derive_geometry,
    generate_template,
    print_geometry_report,
    PAGE_WIDTH_MM,
    PAGE_HEIGHT_MM,
    DPI,
    MARKER_SIZE_MM,
    MARKER_MARGIN_MM,
    STRIP_WIDTH_MM,
    STRIP_HEIGHT_MM,
    GREY_ROI_LEFT_FRAC,
    GREY_ROI_RIGHT_FRAC,
    GREY_ROI_TOP_FRAC,
    GREY_ROI_BOTTOM_FRAC,
)


def validate(output_path: str = None) -> None:
    """Run the full template validation sequence."""

    # ------------------------------------------------------------------
    # Determine output path
    # ------------------------------------------------------------------
    if output_path is None:
        output_path = os.path.join(_TOOLS_DIR, "narcscan_a4_template.png")

    # ------------------------------------------------------------------
    # Derive geometry
    # ------------------------------------------------------------------
    g = derive_geometry()

    # ------------------------------------------------------------------
    # Generate template
    # ------------------------------------------------------------------
    saved = generate_template(output_path)

    print()
    print("*" * 62)
    print("  NarcScan A4 Template Validation")
    print("*" * 62)
    print()
    print("  Template PNG saved to:")
    print("    {}".format(saved))
    print()

    # ------------------------------------------------------------------
    # Physical and pixel dimensions
    # ------------------------------------------------------------------
    def px_to_mm(px):
        return px / DPI * 25.4

    print("  Page dimensions")
    print("    A4 physical : {} mm x {} mm".format(
        int(PAGE_WIDTH_MM), int(PAGE_HEIGHT_MM)))
    print("    At {} DPI   : {} x {} px".format(DPI, g["page_w"], g["page_h"]))
    print()

    print("  ArUco markers")
    print("    Dictionary  : DICT_4X4_50")
    print("    IDs         : 0=top-left  1=top-right  2=bottom-right  3=bottom-left")
    print("    Physical    : {} mm x {} mm  ({} x {} px)".format(
        int(MARKER_SIZE_MM), int(MARKER_SIZE_MM),
        g["marker_px"], g["marker_px"]))
    print("    Page margin : {} mm  ({} px from each page edge)".format(
        int(MARKER_MARGIN_MM), g["margin_px"]))
    print()

    print("  Card interior  (area between inner marker edges)")
    int_w_mm = px_to_mm(g["interior_w"])
    int_h_mm = px_to_mm(g["interior_h"])
    print("    Physical : {:.1f} mm x {:.1f} mm".format(int_w_mm, int_h_mm))
    print("    Pixels   : {} x {} px".format(g["interior_w"], g["interior_h"]))
    print()

    print("  Grey calibration region  (matches backend/app/cv/calibration.py)")
    grey_w_mm = px_to_mm(g["grey_x2"] - g["grey_x1"])
    grey_h_mm = px_to_mm(g["grey_y2"] - g["grey_y1"])
    print("    Physical : {:.1f} mm x {:.1f} mm".format(grey_w_mm, grey_h_mm))
    print("    Pixels   : {} x {} px".format(
        g["grey_x2"] - g["grey_x1"],
        g["grey_y2"] - g["grey_y1"]))
    print("    Norm coords (interior fractions):")
    print("      left   = {}   right  = {}".format(
        GREY_ROI_LEFT_FRAC, GREY_ROI_RIGHT_FRAC))
    print("      top    = {}   bottom = {}".format(
        GREY_ROI_TOP_FRAC, GREY_ROI_BOTTOM_FRAC))
    print()

    print("  Test-strip placement area  (roi.py ROI target)")
    strip_w_mm = px_to_mm(g["strip_x2"] - g["strip_x1"])
    strip_h_mm = px_to_mm(g["strip_y2"] - g["strip_y1"])
    print("    Physical : {:.1f} mm x {:.1f} mm".format(strip_w_mm, strip_h_mm))
    print("    Pixels   : {} x {} px".format(
        g["strip_x2"] - g["strip_x1"],
        g["strip_y2"] - g["strip_y1"]))
    print("    Norm coords (interior fractions -- set in roi.py):")
    print("      STRIP_ROI_LEFT_FRAC   = {:.4f}".format(g["strip_left_frac"]))
    print("      STRIP_ROI_RIGHT_FRAC  = {:.4f}".format(g["strip_right_frac"]))
    print("      STRIP_ROI_TOP_FRAC    = {:.4f}".format(g["strip_top_frac"]))
    print("      STRIP_ROI_BOTTOM_FRAC = {:.4f}".format(g["strip_bottom_frac"]))
    print()

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    errors = []

    if g["strip_top_frac"] < GREY_ROI_BOTTOM_FRAC:
        errors.append(
            "FAIL: Strip top ({:.4f}) overlaps grey calibration bottom ({}).".format(
                g["strip_top_frac"], GREY_ROI_BOTTOM_FRAC
            )
        )

    if g["strip_right_frac"] > 1.0 or g["strip_left_frac"] < 0.0:
        errors.append(
            "FAIL: Strip horizontal fracs out of range "
            "({:.4f}, {:.4f}).".format(g["strip_left_frac"], g["strip_right_frac"])
        )

    if g["strip_bottom_frac"] > 1.0 or g["strip_top_frac"] < 0.0:
        errors.append(
            "FAIL: Strip vertical fracs out of range "
            "({:.4f}, {:.4f}).".format(g["strip_top_frac"], g["strip_bottom_frac"]))

    if errors:
        print("  !! VALIDATION ERRORS !!")
        for err in errors:
            print("    " + err)
    else:
        print("  Geometry checks  : ALL PASSED")
        print("    - Strip does not overlap calibration region.")
        print("    - Strip fractions are within [0, 1].")

    print()
    print("*" * 62)
    print("  PRINTING CHECKLIST")
    print("*" * 62)
    print()
    print("  [ ] 1. Open:  {}".format(saved))
    print()
    print("  [ ] 2. In the print dialog:")
    print('         Page Scaling = "None" / "Actual Size" / "100%"')
    print("         Do NOT select 'Fit to page' or 'Shrink to fit'.")
    print("         Orientation = Portrait.")
    print()
    print("  [ ] 3. Print the page.")
    print()
    print("  [ ] 4. Measure the 50 mm scale bar in the bottom-left")
    print("         with a physical ruler.")
    print("         It MUST measure exactly 50 mm.")
    print("         If it does not, re-print at 100% actual size.")
    print()
    print("  [ ] 5. Confirm the four ArUco markers are clearly")
    print("         printed in the corners without cropping.")
    print()
    print("  [ ] 6. Place a real test strip inside the green rectangle")
    print("         labelled 'PLACE TEST STRIP HERE'.")
    print()
    print("  NOTE: Do NOT treat the physical geometry as final until")
    print("        these steps have been completed and verified.")
    print()
    print("*" * 62)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the NarcScan A4 template geometry."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: tools/narcscan_a4_template.png)",
    )
    args = parser.parse_args()
    validate(args.output)
