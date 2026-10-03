"""
reference_profiles.py
=====================

Phase 8 Revision - NCB Test Profiles & Reference-Colour Engine

This module defines the structured test-profile model for the NCB Narcotic 
Drugs Detection Kit. It replaces the old synthetic/demo logic with a robust 
architecture that records documented chemical colour reactions.

IMPORTANT SCIENTIFIC DISCLAIMER
-------------------------------
The profiles defined in this module are based on documented qualitative colour 
descriptions. They DO NOT definitively identify a drug. The system provides a 
PRESUMPTIVE FIELD-SCREENING result ONLY. 

Laboratory confirmation is always required for definitive identification.

REAL DATA ACQUISITION PATH (FUTURE WORK)
----------------------------------------
Because we do NOT invent Lab values from qualitative colour names (e.g., "blue"), 
the 'reference_lab' fields below are currently marked as None.

The intended workflow to acquire actual machine-readable reference data is:
1. Obtain official/documented reference standards and the physical NCB kit.
2. Perform controlled photographs of the chemical reactions.
3. Apply our grey-reference calibration pipeline.
4. Measure CIELAB values using the Phase 7 colour analysis module.
5. Compile a Reference Lab dataset.
6. Validate thresholds experimentally before applying them in the classifier.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

@dataclass
class TargetReaction:
    """
    Structured model representing a specific documented reaction 
    for a particular target category within a reagent test.
    """
    target_category: str
    documented_reaction: str
    reference_colour_description: str
    reference_lab: Optional[Tuple[float, float, float]]
    reference_source: str
    validation_status: str

@dataclass
class NCBTestProfile:
    """
    Structured model representing a specific reagent test from the 
    NCB Narcotic Drugs Detection Kit.
    """
    test_id: str
    test_name: str
    reagent_or_test_description: str
    target_reactions: Dict[str, TargetReaction]

# ---------------------------------------------------------------------------
# NCB TEST PROFILES
# ---------------------------------------------------------------------------

TEST_A = NCBTestProfile(
    test_id="TEST_A",
    test_name="Marquis Reagent",
    reagent_or_test_description="Formaldehyde and concentrated sulfuric acid.",
    target_reactions={
        "Opiates": TargetReaction(
            target_category="Opiates (Heroin, Morphine, Codeine)",
            documented_reaction="Purple/violet",
            reference_colour_description="Claret violet (RAL 4004) \u2014 closest documented standard-colour match to the purple/violet Marquis reaction reported for opiates",
            reference_lab=(32.0, 25.0, 1.0),
            reference_source="Colour reaction reported for morphine with Marquis reagent in Chang et al., 'Presumptive Tests for Xylazine - A Computer Vision Approach,' Analytical Science Advances, 2025, 6(1):e70008, DOI:10.1002/ansa.70008 (Table 3). Matched to RAL Classic colour RAL 4004 (Claret violet). Lab value per hextoral.com RAL-to-Lab conversion (RGB 112,60,76 -> Lab 32,25,1), a third-party calculated approximation, NOT an official RAL-published Lab measurement.",
            validation_status="REFERENCE_FROM_PUBLISHED_LITERATURE_RAL_MATCH_NOT_CAMERA_VALIDATED"
        ),
        "Amphetamines": TargetReaction(
            target_category="Amphetamine-type stimulants (ATS)",
            documented_reaction="Orange to brown",
            reference_colour_description="Orange/brown",
            reference_lab=None,
            reference_source="Standard Forensic Guidelines / UNODC",
            validation_status="CONFLICTING_SOURCES_NOT_RESOLVED"
        )
    }
)

TEST_B = NCBTestProfile(
    test_id="TEST_B",
    test_name="Mecke Reagent",
    reagent_or_test_description="Selenious acid and concentrated sulfuric acid.",
    target_reactions={
        "Heroin": TargetReaction(
            target_category="Opiates (Heroin)",
            documented_reaction="Green changing to blue-green.",
            reference_colour_description="Blue-green",
            reference_lab=None,
            reference_source="Standard Forensic Guidelines / UNODC",
            validation_status="REFERENCE_DOCUMENTED_NOT_CAMERA_VALIDATED"
        )
    }
)

TEST_C = NCBTestProfile(
    test_id="TEST_C",
    test_name="Dille-Koppanyi Reagent",
    reagent_or_test_description="Cobalt acetate and isopropylamine.",
    target_reactions={
        "Barbiturates": TargetReaction(
            target_category="Barbiturates",
            documented_reaction="Reddish-purple or lilac.",
            reference_colour_description="Reddish-purple",
            reference_lab=None,
            reference_source="Standard Forensic Guidelines / UNODC",
            validation_status="REFERENCE_DOCUMENTED_NOT_CAMERA_VALIDATED"
        )
    }
)

TEST_D = NCBTestProfile(
    test_id="TEST_D",
    test_name="Scott Reagent (Modified)",
    reagent_or_test_description="Cobalt thiocyanate solution.",
    target_reactions={
        "Cocaine": TargetReaction(
            target_category="Cocaine",
            documented_reaction="Blue precipitate/coloration.",
            reference_colour_description="Blue",
            reference_lab=None,
            reference_source="Standard Forensic Guidelines / UNODC",
            validation_status="REFERENCE_DOCUMENTED_NOT_CAMERA_VALIDATED"
        )
    }
)

TEST_E = NCBTestProfile(
    test_id="TEST_E",
    test_name="Ehrlich Reagent",
    reagent_or_test_description="p-Dimethylaminobenzaldehyde.",
    target_reactions={
        "LSD": TargetReaction(
            target_category="LSD",
            documented_reaction="Purple or violet colour development.",
            reference_colour_description="Purple",
            reference_lab=None,
            reference_source="Standard Forensic Guidelines / UNODC",
            validation_status="REFERENCE_DOCUMENTED_NOT_CAMERA_VALIDATED"
        )
    }
)

# A registry of available test profiles
NCB_PROFILES: Dict[str, NCBTestProfile] = {
    "TEST_A": TEST_A,
    "TEST_B": TEST_B,
    "TEST_C": TEST_C,
    "TEST_D": TEST_D,
    "TEST_E": TEST_E,
}
