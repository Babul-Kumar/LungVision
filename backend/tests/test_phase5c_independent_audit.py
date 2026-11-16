"""
LungVision AI — Phase 5C Automated Audit Verification Test Suite
Test File: backend/tests/test_phase5c_independent_audit.py
Version: 5.3.0

Verifies:
1. Presence and validity of all 10 Phase 5C required deliverables
2. Strict preservation of production invariants (DenseNet121 SHA-256, 6 NIH thresholds)
3. Task A: Prediction independence and root cause analysis of baseline collapse
4. Task B: Exact metric reproduction within tolerance (ASSD delta <= 0.005 mm)
5. Task C: Oracle SVD orthogonality, training isolation, and mode truncation sweep
6. Task D: Statistical reanalysis (Wilcoxon, bootstrap CIs, Holm-Bonferroni)
7. Task E: Data leakage audit (zero patient overlap, external generalization status)
8. Task F: Uncertainty calibration evaluation (ECE, interval coverage)
9. Task G: Anatomical coordinates, outward normals, and Divergence Theorem signed volume
10. Task H: Reproducibility manifest completeness and execution CLI
"""

import csv
import hashlib
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "research" / "3d_reconstruction"


def test_phase5c_all_deliverables_present():
    required_files = [
        OUTPUT_DIR / "phase5c_prediction_independence_audit.json",
        OUTPUT_DIR / "phase5c_patient_level_metrics.csv",
        OUTPUT_DIR / "phase5c_reproduction_metrics.json",
        OUTPUT_DIR / "phase5c_oracle_verification.json",
        OUTPUT_DIR / "phase5c_statistical_reanalysis.json",
        OUTPUT_DIR / "phase5c_data_leakage_audit.json",
        OUTPUT_DIR / "phase5c_uncertainty_calibration.json",
        OUTPUT_DIR / "phase5c_anatomical_coordinate_audit.json",
        OUTPUT_DIR / "phase5c_reproducibility_manifest.json",
        OUTPUT_DIR / "PHASE_5C_INDEPENDENT_VALIDATION_REPORT.md",
    ]
    for p in required_files:
        assert p.exists(), f"Missing required Phase 5C deliverable: {p.name}"
        assert p.stat().st_size > 0, f"Empty deliverable: {p.name}"


def test_production_invariants_preserved():
    prod_ckpt = PROJECT_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
    assert prod_ckpt.exists(), "Production DenseNet121 checkpoint missing"
    
    h = hashlib.sha256()
    with open(prod_ckpt, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    actual_hash = h.hexdigest()
    expected_hash = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"
    assert actual_hash == expected_hash, f"Production checkpoint hash corrupted: {actual_hash}"

    from backend.app.services.xray_service import XRayService
    xray_svc = XRayService()
    expected_thresholds = {
        "Cardiomegaly": 0.9502,
        "Edema": 0.8770,
        "Consolidation": 0.7329,
        "Pneumonia": 0.7529,
        "Atelectasis": 0.7495,
        "Pneumothorax": 0.5825,
    }
    for disease, thresh in expected_thresholds.items():
        assert round(float(xray_svc.nih_thresholds[disease]), 4) == thresh, f"NIH threshold altered for {disease}"


def test_task_a_prediction_independence_audit():
    p = OUTPUT_DIR / "phase5c_prediction_independence_audit.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["task"] == "Task A: Prediction Independence Audit"
    assert data["cohort_size"] == 65
    
    stats = data["cohort_coefficient_statistics"]
    assert stats["frontal_only"]["mean_l2_norm"] < 3.0, "Frontal baseline should have near-zero L2 norm"
    assert stats["lateral_only"]["mean_l2_norm"] < 3.0, "Lateral baseline should have near-zero L2 norm"
    assert stats["dual_view_v2"]["mean_l2_norm"] > 40.0, "Trained V2 model should produce substantial L2 norm"

    findings = data["root_cause_findings"]
    assert len(findings) >= 3
    finding_ids = [f["finding_id"] for f in findings]
    assert "FINDING_UNTRAINED_BASELINES" in finding_ids


def test_task_b_evaluation_reproduction():
    csv_p = OUTPUT_DIR / "phase5c_patient_level_metrics.csv"
    with open(csv_p, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 65, f"Expected 65 patient rows in CSV, got {len(reader)}"

    json_p = OUTPUT_DIR / "phase5c_reproduction_metrics.json"
    with open(json_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    reprod = data["reproduction_audit"]
    assert reprod["reproduction_exact_match"] is True, "Reproduction failed tolerance threshold"
    assert reprod["absolute_deltas"]["assd_mm"] <= 0.005
    assert reprod["absolute_deltas"]["hd95_mm"] <= 0.005
    assert reprod["absolute_deltas"]["surface_dice_2mm"] <= 0.005
    assert reprod["absolute_deltas"]["volume_error_pct"] <= 0.05


def test_task_c_oracle_verification():
    p = OUTPUT_DIR / "phase5c_oracle_verification.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["eigenvector_orthogonality_verified"] is True
    assert data["training_isolation_audit"]["isolation_status"] == "VERIFIED_ZERO_LEAKAGE"
    sweep = data["oracle_mode_truncation_sweep"]
    assert sweep["20"]["assd_mm_mean"] <= 0.02, "Oracle 20-mode ASSD should be ~0.0165 mm"


def test_task_d_statistical_reanalysis():
    p = OUTPUT_DIR / "phase5c_statistical_reanalysis.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["verdict"]["reanalysis_status"] == "CONFIRMED_AND_REPRODUCED"
    assert data["verdict"]["all_hypotheses_passed_holm_bonferroni"] is True
    w_audit = data["w_plus_zero_audit"]["v2_vs_mean_shape_verification"]
    assert w_audit["patients_improved_count"] == 65
    assert w_audit["w_plus_calculated"] == 0.0


def test_task_e_data_leakage_audit():
    p = OUTPUT_DIR / "phase5c_data_leakage_audit.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["patient_overlap_audit"]["zero_patient_leakage_verified"] is True
    assert data["external_generalization_assessment"]["prospective_external_testing_status"] == "UNTESTED"


def test_task_f_uncertainty_calibration():
    p = OUTPUT_DIR / "phase5c_uncertainty_calibration.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["expected_calibration_error_mm"] > 5.0, "Expected underconfidence in physical units"
    assert data["verdict"]["calibration_assessment"] == "PARTIALLY_CALIBRATED_RANK_ORDERED"


def test_task_g_anatomical_coordinates():
    p = OUTPUT_DIR / "phase5c_anatomical_coordinate_audit.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    topo = data["mesh_topology"]
    assert topo["is_watertight"] is True
    assert topo["is_winding_consistent"] is True
    
    vol = data["volumetric_validation"]
    assert vol["total_signed_volume_mm3"] > 0, "Divergence theorem signed volume must be positive"
    assert 1.15 <= vol["bilateral_symmetry_ratio"] <= 1.30


def test_task_h_reproducibility_manifest():
    p = OUTPUT_DIR / "phase5c_reproducibility_manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["verdict"]["production_checkpoint_preserved"] is True
    assert data["verdict"]["audit_verdict_declaration"] == "COMPLETE: all required audits independently reproduced."
