"""
test_classifier.py
==================

Phase 8 Revision automated tests for the reference-based decision engine.
"""

import sys
import os
import math
import pytest

# Add backend directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.classifier import classify_reaction
from app.cv.reference_profiles import TargetReaction, TEST_A
from tests.fixtures.synthetic_profiles import SYNTHETIC_POSITIVE_PROFILE, SYNTHETIC_NEGATIVE_PROFILE

class TestClassifierLogic:
    def test_missing_reference_lab_returns_required_state(self):
        """No fake \u0394E when reference Lab is unavailable."""
        res = classify_reaction(observed_lab=(50.0, 0.0, 0.0), target_reaction=TEST_A.target_reactions["Amphetamines"])
        assert res.success is True
        assert res.reference_data_available is False
        assert res.result == "REFERENCE_DATA_REQUIRED"
        assert res.delta_e_00 is None
        assert res.validation_status == "CONFLICTING_SOURCES_NOT_RESOLVED"

    def test_exact_positive_match(self):
        """Observed colour exactly equal to positive reference -> POSITIVE"""
        res = classify_reaction(
            observed_lab=SYNTHETIC_POSITIVE_PROFILE.reference_lab, 
            target_reaction=SYNTHETIC_POSITIVE_PROFILE
        )
        assert res.success is True
        assert res.reference_data_available is True
        assert res.delta_e_00 < 1e-6
        assert res.result == "POSITIVE"

    def test_positive_like_input(self):
        """Clearly positive-like synthetic input within threshold -> POSITIVE"""
        # Close to SYNTHETIC_POSITIVE_PROFILE (35.0, 45.0, -10.0)
        res = classify_reaction(
            observed_lab=(36.0, 44.0, -9.0), 
            target_reaction=SYNTHETIC_POSITIVE_PROFILE,
            decision_threshold=5.0
        )
        assert res.success is True
        assert res.delta_e_00 <= 5.0
        assert res.result == "POSITIVE"

    def test_inconclusive_input(self):
        """Input outside threshold -> INCONCLUSIVE"""
        # Far from SYNTHETIC_POSITIVE_PROFILE
        res = classify_reaction(
            observed_lab=(80.0, 0.0, 0.0), 
            target_reaction=SYNTHETIC_POSITIVE_PROFILE,
            decision_threshold=5.0
        )
        assert res.success is True
        assert res.delta_e_00 > 5.0
        assert res.result == "INCONCLUSIVE"


class TestClassifierValidation:
    def test_missing_input(self):
        """Missing input (None) should fail."""
        res = classify_reaction(observed_lab=None, target_reaction=TEST_A.target_reactions["Opiates"])
        assert res.success is False
        assert "None" in res.reason

    def test_invalid_length_input(self):
        """Invalid Lab input (tuple with 2 items) should fail."""
        res = classify_reaction(observed_lab=(50.0, 0.0), target_reaction=TEST_A.target_reactions["Opiates"])
        assert res.success is False
        assert "3-element" in res.reason
        
    def test_nan_input(self):
        """NaN input should fail."""
        res = classify_reaction(observed_lab=(50.0, float('nan'), 0.0), target_reaction=TEST_A.target_reactions["Opiates"])
        assert res.success is False
        assert "invalid value" in res.reason
        
    def test_inf_input(self):
        """Infinite input should fail."""
        res = classify_reaction(observed_lab=(float('inf'), 0.0, 0.0), target_reaction=TEST_A.target_reactions["Opiates"])
        assert res.success is False
        assert "invalid value" in res.reason
        
    def test_invalid_l_range(self):
        """L* outside [0, 100] should fail."""
        res = classify_reaction(observed_lab=(150.0, 0.0, 0.0), target_reaction=TEST_A.target_reactions["Opiates"])
        assert res.success is False
        assert "between 0 and 100" in res.reason
        
    def test_missing_profile(self):
        """Missing profile should fail."""
        res = classify_reaction(observed_lab=(50.0, 0.0, 0.0), target_reaction=None)
        assert res.success is False
        assert "None" in res.reason


class TestClassifierCustomization:
    def test_custom_test_profile(self):
        """Custom valid TargetReaction is processed correctly."""
        custom_reaction = TargetReaction(
            target_category="Target",
            documented_reaction="Reaction",
            reference_colour_description="Color",
            reference_lab=(50.0, 10.0, 10.0),
            reference_source="Source",
            validation_status="VALIDATED"
        )
        
        res = classify_reaction(
            observed_lab=(50.0, 10.0, 10.0),
            target_reaction=custom_reaction
        )
        assert res.success is True
        assert res.target_category == "Target"
        assert res.result == "POSITIVE"


class TestClassifierResultStructure:
    def test_returned_result_structure(self):
        """Returned result structure matches the specification."""
        res = classify_reaction(
            observed_lab=SYNTHETIC_POSITIVE_PROFILE.reference_lab, 
            target_reaction=SYNTHETIC_POSITIVE_PROFILE
        )
        assert hasattr(res, "success")
        assert hasattr(res, "result")
        assert hasattr(res, "mode")
        assert hasattr(res, "target_category")
        assert hasattr(res, "reference_data_available")
        assert hasattr(res, "observed_lab")
        assert hasattr(res, "reference_lab")
        assert hasattr(res, "delta_e_00")
        assert hasattr(res, "validation_status")
        assert hasattr(res, "source")


class TestPhase81Targets:
    def test_test_a_opiate_populated(self):
        """TEST_A opiate target_reaction has reference_lab populated."""
        target = TEST_A.target_reactions["Opiates"]
        assert target.reference_lab is not None
        assert target.reference_lab == (32.0, 25.0, 1.0)
        assert target.validation_status == "REFERENCE_FROM_PUBLISHED_LITERATURE_RAL_MATCH_NOT_CAMERA_VALIDATED"

    def test_test_a_ats_unpopulated(self):
        """TEST_A ATS target_reaction still has reference_lab = None."""
        target = TEST_A.target_reactions["Amphetamines"]
        assert target.reference_lab is None
        assert target.validation_status == "CONFLICTING_SOURCES_NOT_RESOLVED"

    def test_classify_opiate_returns_real_classification(self):
        """Classify against TEST_A opiate returns real classification (not REQUIRED)."""
        target = TEST_A.target_reactions["Opiates"]
        res = classify_reaction(observed_lab=(32.0, 25.0, 1.0), target_reaction=target)
        assert res.success is True
        assert res.reference_data_available is True
        assert res.result == "POSITIVE"

    def test_classify_ats_returns_required(self):
        """Classify against TEST_A ATS returns REQUIRED."""
        target = TEST_A.target_reactions["Amphetamines"]
        res = classify_reaction(observed_lab=(50.0, 0.0, 0.0), target_reaction=target)
        assert res.success is True
        assert res.reference_data_available is False
        assert res.result == "REFERENCE_DATA_REQUIRED"

    def test_b_through_e_return_required(self):
        """TEST_B through TEST_E all still return REFERENCE_DATA_REQUIRED."""
        from app.cv.reference_profiles import TEST_B, TEST_C, TEST_D, TEST_E
        for profile in (TEST_B, TEST_C, TEST_D, TEST_E):
            for name, target in profile.target_reactions.items():
                res = classify_reaction(observed_lab=(50.0, 0.0, 0.0), target_reaction=target)
                assert res.success is True
                assert res.reference_data_available is False
                assert res.result == "REFERENCE_DATA_REQUIRED"
