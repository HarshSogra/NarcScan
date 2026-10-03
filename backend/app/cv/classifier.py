"""
classifier.py
=============

Phase 8 Revision - NCB Reference-Based Classification

This module implements a presumptive field-screening decision engine that
classifies an observed chemical reaction colour by comparing it to an
NCBTestProfile.

IMPORTANT SCIENTIFIC DISCLAIMER
-------------------------------
The current classifier operates on qualitative reference profiles. If a test
profile lacks camera-validated Lab values, the classifier will safely halt
and report REFERENCE_DATA_REQUIRED. It does NOT invent thresholds.

The system is intended to provide a PRESUMPTIVE FIELD SCREENING result ONLY.
It is NOT a laboratory confirmation.
"""

import math
from typing import Optional, Tuple, Any
from dataclasses import dataclass

from app.cv.color_analysis import calculate_delta_e_ciede2000
from app.cv.reference_profiles import TargetReaction

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class ClassifierResult:
    """
    Structured result of a reference-based classification.
    """
    success: bool
    result: Optional[str] = None
    mode: str = "REFERENCE_PROFILE"
    target_category: Optional[str] = None
    reference_data_available: bool = False
    observed_lab: Optional[Tuple[float, float, float]] = None
    reference_lab: Optional[Tuple[float, float, float]] = None
    delta_e_00: Optional[float] = None
    validation_status: Optional[str] = None
    source: Optional[str] = None
    reason: str = ""


# ==============================================================================
# PUBLIC API
# ==============================================================================

def classify_reaction(
    observed_lab: Tuple[float, float, float],
    target_reaction: TargetReaction,
    decision_threshold: float = 5.0
) -> ClassifierResult:
    """
    Classify an observed Lab colour against a specific target reaction profile.

    Parameters
    ----------
    observed_lab : Tuple[float, float, float]
        The observed colour (e.g., from the center region of the test strip)
        in standard CIELAB format: (L*, a*, b*).
    target_reaction : TargetReaction
        The profile representing the target reaction.
    decision_threshold : float, optional
        The maximum CIEDE2000 distance allowable to be considered a match 
        (POSITIVE). If the distance exceeds this, the result is INCONCLUSIVE
        or NEGATIVE depending on logic (currently defaults to INCONCLUSIVE 
        if outside the immediate matching envelope).

    Returns
    -------
    ClassifierResult
        A structured result containing the presumptive classification, 
        computed distances, and explicit scientific disclaimers.
    """
    # ------------------------------------------------------------------
    # 1. Validate Input
    # ------------------------------------------------------------------
    if observed_lab is None:
        return ClassifierResult(success=False, reason="Observed Lab color is None.")
        
    if not isinstance(observed_lab, (tuple, list)) or len(observed_lab) != 3:
        return ClassifierResult(
            success=False, 
            reason=f"Observed Lab color must be a 3-element tuple, got {type(observed_lab)}."
        )

    for i, val in enumerate(observed_lab):
        if val is None or math.isnan(val) or math.isinf(val):
            return ClassifierResult(
                success=False, 
                reason=f"Observed Lab color contains invalid value at index {i}: {val}"
            )

    L_obs = float(observed_lab[0])
    if not (0.0 <= L_obs <= 100.0):
        return ClassifierResult(
            success=False, 
            reason=f"Observed L* must be between 0 and 100, got {L_obs}"
        )

    if target_reaction is None:
        return ClassifierResult(success=False, reason="Target reaction cannot be None.")

    # ------------------------------------------------------------------
    # 2. Check for Reference Data
    # ------------------------------------------------------------------
    if target_reaction.reference_lab is None:
        return ClassifierResult(
            success=True,
            result="REFERENCE_DATA_REQUIRED",
            target_category=target_reaction.target_category,
            reference_data_available=False,
            observed_lab=observed_lab,
            reference_lab=None,
            validation_status=target_reaction.validation_status,
            source=target_reaction.reference_source,
            reason="Quantitative reference Lab values are currently unavailable for this test."
        )

    # ------------------------------------------------------------------
    # 3. Distance Calculation
    # ------------------------------------------------------------------
    try:
        dist = calculate_delta_e_ciede2000(observed_lab, target_reaction.reference_lab)
    except ValueError as e:
        return ClassifierResult(success=False, reason=f"Distance calculation failed: {e}")

    # ------------------------------------------------------------------
    # 4. Decision Logic
    # ------------------------------------------------------------------
    # This is a PRESUMPTIVE match only. 
    # If the distance is sufficiently close (<= decision_threshold), it's a match.
    # Otherwise, it's INCONCLUSIVE (does not strongly match the known positive, 
    # but we do not blindly call it NEGATIVE as it may just be weak or degraded).
    if dist <= decision_threshold:
        decision = "POSITIVE"
    else:
        decision = "INCONCLUSIVE"

    # ------------------------------------------------------------------
    # 5. Result Structure
    # ------------------------------------------------------------------
    return ClassifierResult(
        success=True,
        result=decision,
        target_category=target_reaction.target_category,
        reference_data_available=True,
        observed_lab=observed_lab,
        reference_lab=target_reaction.reference_lab,
        delta_e_00=dist,
        validation_status=target_reaction.validation_status,
        source=target_reaction.reference_source
    )
