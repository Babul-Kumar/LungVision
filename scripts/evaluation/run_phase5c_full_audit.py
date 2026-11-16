#!/usr/bin/env python3
"""
LungVision AI — Phase 5C Master Audit Orchestrator
Script: scripts/evaluation/run_phase5c_full_audit.py
Version: 5.3.0

Orchestrates all Phase 5C audit tasks:
Task A: Prediction Independence Audit
Task B: Full Evaluation Reproduction
Task C: Oracle & SSM Verification
Task D: Statistical Reanalysis & Wilcoxon Audit
Task E: Data Leakage & Generalization Audit
Task F: Uncertainty Calibration Audit
Task G: Anatomical Coordinates & Volumetry Audit
Task H: Reproducibility Manifest Generation
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.audit_phase5c_prediction_independence import run_prediction_independence_audit
from scripts.evaluation.reproduce_phase5c_evaluations import reproduce_evaluations
from scripts.evaluation.verify_phase5c_oracle import verify_oracle_and_ssm
from scripts.evaluation.run_phase5c_statistical_reanalysis import run_statistical_reanalysis
from scripts.evaluation.audit_phase5c_data_leakage import audit_data_leakage_and_generalization
from scripts.evaluation.audit_phase5c_uncertainty_calibration import audit_uncertainty_calibration
from scripts.evaluation.audit_phase5c_anatomical_coordinates import audit_anatomical_coordinates
from scripts.evaluation.generate_phase5c_reproducibility_manifest import generate_manifest


def main():
    start = time.time()
    print("=" * 90)
    print("LUNGVISION AI — PHASE 5C COMPLETE AUDIT SUITE EXECUTION")
    print("=" * 90)

    print("\n>>> STEP 1/8: TASK A — PREDICTION INDEPENDENCE AUDIT")
    run_prediction_independence_audit()

    print("\n>>> STEP 2/8: TASK B — FULL EVALUATION REPRODUCTION")
    reproduce_evaluations()

    print("\n>>> STEP 3/8: TASK C — ORACLE & SSM ANALYSIS")
    verify_oracle_and_ssm()

    print("\n>>> STEP 4/8: TASK D — STATISTICAL REANALYSIS & WILCOXON AUDIT")
    run_statistical_reanalysis()

    print("\n>>> STEP 5/8: TASK E — DATA LEAKAGE & GENERALIZATION AUDIT")
    audit_data_leakage_and_generalization()

    print("\n>>> STEP 6/8: TASK F — UNCERTAINTY CALIBRATION AUDIT")
    audit_uncertainty_calibration()

    print("\n>>> STEP 7/8: TASK G — ANATOMICAL COORDINATES & VOLUMETRIC AUDIT")
    audit_anatomical_coordinates()

    print("\n>>> STEP 8/8: TASK H — REPRODUCIBILITY MANIFEST GENERATION")
    generate_manifest()

    elapsed = time.time() - start
    print("\n" + "=" * 90)
    print(f"ALL 8 PHASE 5C AUDIT DELIVERABLES GENERATED IN {elapsed:.2f}s")
    print("STATUS: COMPLETE: all required audits independently reproduced.")
    print("=" * 90)


if __name__ == "__main__":
    main()
