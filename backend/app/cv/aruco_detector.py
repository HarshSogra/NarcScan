import cv2
import numpy as np

# The four marker IDs we expect to see on the NarcScan test card.
# They will mark the four corners of the card (one per physical corner).
EXPECTED_IDS = [0, 1, 2, 3]

# Load the ArUco dictionary once at module level so it is not
# recreated on every function call.
#
# DICT_4X4_50 means:
#   4x4  — each marker is a 4x4 grid of black/white squares (the "bits")
#   50   — the dictionary contains 50 unique marker patterns to choose from
#
# We only need 4 unique markers (IDs 0-3), so 50 is more than enough.
# A 4x4 grid gives a good balance between:
#   - being large enough to encode unique IDs reliably
#   - being small enough to detect even when the card is not perfectly flat
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# DetectorParameters controls the detection algorithm's behaviour.
# Using the defaults is fine for Phase 2.
DETECTOR_PARAMS = cv2.aruco.DetectorParameters()

# Build the detector object using the dictionary and parameters.
DETECTOR = cv2.aruco.ArucoDetector(ARUCO_DICT, DETECTOR_PARAMS)


def detect_markers(image: np.ndarray) -> dict:
    """
    Detect ArUco markers in an image.

    Parameters
    ----------
    image : np.ndarray
        A BGR or grayscale image as a NumPy array (the standard
        format that OpenCV uses for images).

    Returns
    -------
    dict with the following keys:

        "detected"    : bool  — True if at least one marker was found.

        "markers"     : dict  — Maps each found marker ID (int) to its
                                four corner coordinates.
                                Each corner set is a list of 4 points,
                                where each point is [x, y] in pixels.
                                Example:
                                {
                                    0: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
                                    1: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]],
                                }

        "missing_ids" : list  — The expected IDs (from EXPECTED_IDS)
                                that were NOT found in this image.
                                Empty list means all four were found.
    """
    # --- 1. Convert to grayscale if necessary ---
    # ArUco detection works on a single-channel (grayscale) image.
    # If the image has 3 channels (BGR colour) we convert it.
    # If it is already grayscale (1 channel) we use it directly.
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # --- 2. Run the detector ---
    # detectMarkers returns three things:
    #   corners : a list of arrays, one per detected marker.
    #             Each array has shape (1, 4, 2) — 4 corner points, each
    #             with an x and y pixel coordinate.
    #   ids     : a NumPy array of shape (N, 1) containing the integer ID
    #             for each detected marker, in the same order as corners.
    #             None if no markers were found.
    #   rejected: candidate regions that looked like markers but were
    #             rejected — we don't need these in Phase 2.
    corners, ids, _rejected = DETECTOR.detectMarkers(gray)

    # --- 3. Handle the "no markers found" case ---
    if ids is None:
        return {
            "detected": False,
            "markers": {},
            "missing_ids": list(EXPECTED_IDS),
        }

    # --- 4. Build the markers dictionary ---
    # ids comes as shape (N, 1) — we flatten it to a simple 1-D list
    # so we can loop over it easily.
    flat_ids = ids.flatten().tolist()

    markers = {}
    for marker_index, marker_id in enumerate(flat_ids):
        # corners[marker_index] has shape (1, 4, 2).
        # We squeeze away the outer dimension to get shape (4, 2),
        # then convert to a plain Python list so it is easy to read/print.
        corner_array = corners[marker_index][0]  # shape: (4, 2)
        markers[marker_id] = corner_array.tolist()

    # --- 5. Work out which expected markers are missing ---
    missing_ids = [eid for eid in EXPECTED_IDS if eid not in markers]

    return {
        "detected": True,
        "markers": markers,
        "missing_ids": missing_ids,
    }
