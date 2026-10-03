"""
manual_reference_profile_test.py
================================

Displays the available NCB test profiles and demonstrates the classifier 
safely returning REFERENCE_DATA_REQUIRED when quantitative data is unavailable.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.cv.classifier import classify_reaction
from app.cv.reference_profiles import NCB_PROFILES

def run_demo():
    print("==================================================")
    print("NarcScan \u2014 NCB Reference Profiles")
    print("==================================================")
    print()

    # Simulate an observed colour from the camera (Phase 7 output)
    simulated_observed_lab = (50.0, 10.0, -10.0)
    print(f"Simulated Observed Lab from Phase 7: {simulated_observed_lab}")
    print()

    for test_id, profile in NCB_PROFILES.items():
        print(f"Test: {profile.test_id}")
        print(f"Name: {profile.test_name}")
        print(f"Description: {profile.reagent_or_test_description}")
        
        for name, target in profile.target_reactions.items():
            print(f"  Target: {target.target_category}")
            print(f"  Documented reaction: {target.documented_reaction}")
            print(f"  Reference colour: {target.reference_colour_description}")
            
            ref_lab_str = target.reference_lab if target.reference_lab else "NOT AVAILABLE"
            print(f"  Reference Lab: {ref_lab_str}")
            print(f"  Validation: {target.validation_status}")
            
            # Run through the classifier
            res = classify_reaction(observed_lab=simulated_observed_lab, target_reaction=target)
            
            print("  Classifier Result:")
            if res.success:
                print(f"    Result: {res.result}")
                if res.delta_e_00 is not None:
                    print(f"    \u0394E: {res.delta_e_00:.2f}")
            else:
                print(f"    Failed: {res.reason}")
                
            print("-" * 50)

    print()
    print("==================================================")
    print("IMPORTANT:")
    print("No quantitative classification is performed when a validated Lab reference")
    print("is unavailable. The system provides presumptive field-screening only.")
    print("==================================================")

if __name__ == "__main__":
    run_demo()
