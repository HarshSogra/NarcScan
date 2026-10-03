"""
synthetic_profiles.py
=====================

SYNTHETIC TEST FIXTURE

These are synthetic profiles used STRICTLY for unit testing the 
classifier logic. They MUST NEVER be used as actual NarcScan reference data.
"""

from app.cv.reference_profiles import TargetReaction

SYNTHETIC_POSITIVE_PROFILE = TargetReaction(
    target_category="Synthetic Pos",
    documented_reaction="Synthetic Red",
    reference_colour_description="Red",
    reference_lab=(35.0, 45.0, -10.0),
    reference_source="SYNTHETIC_TEST_FIXTURE",
    validation_status="SYNTHETIC_TEST_FIXTURE"
)

SYNTHETIC_NEGATIVE_PROFILE = TargetReaction(
    target_category="Synthetic Neg",
    documented_reaction="Synthetic Grey",
    reference_colour_description="Grey",
    reference_lab=(85.0, 2.0, 2.0),
    reference_source="SYNTHETIC_TEST_FIXTURE",
    validation_status="SYNTHETIC_TEST_FIXTURE"
)
