"""
Unit and regression test suite for Phase 6C: Independent Research Model
Reproduction & Generalization Audit.

Verifies:
1. Production checkpoint SHA-256 byte hash and frozen NIH thresholds remain strictly unaltered.
2. All 9 Phase 6C deliverables exist, are valid, and contain required analytical sections.
3. Checkpoint reproduction audit correctly identifies the missing 512x512 dedicated checkpoint
   and declares the experiment status as PARTIAL.
4. Patient-cluster bootstrap statistics confirm positive variance inflation over naive image resampling.
5. Multiclass calibration metrics (Brier, ECE, logistic slope/intercept) are properly computed.
6. Operating point evaluations include cluster bootstrap 95% confidence intervals and non-diagnostic notices.
7. External dataset evaluation enforces CXR-only governance invariants.
"""

import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_CHECKPOINT = (
    REPO_ROOT / "models" / "densenet121" / "densenet121_nih_best.pth"
    if (REPO_ROOT / "models" / "densenet121" / "densenet121_nih_best.pth").exists()
    else REPO_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
)
EXPECTED_SHA256 = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"

FROZEN_THRESHOLDS = {
    "Cardiomegaly": 0.9502,
    "Edema": 0.8770,
    "Consolidation": 0.7329,
    "Pneumonia": 0.7529,
    "Atelectasis": 0.7495,
    "Pneumothorax": 0.5825,
}

PATHOLOGIES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
]

DELIVERABLES_DIR = REPO_ROOT / "reports" / "phase6c"
EXPECTED_DELIVERABLES = [
    "phase6c_checkpoint_reproduction.json",
    "phase6c_preprocessing_reproduction.json",
    "phase6c_resolution_validation.json",
    "phase6c_patient_cluster_statistics.json",
    "phase6c_calibration_report.json",
    "phase6c_operating_point_validation.json",
    "phase6c_error_analysis.json",
    "phase6c_external_evaluation.json",
    "PHASE_6C_INDEPENDENT_REPRODUCTION_REPORT.md",
]


def test_production_checkpoint_invariants():
    """Verify production DenseNet121 checkpoint SHA-256 hash has not been modified."""
    assert PRODUCTION_CHECKPOINT.exists(), f"Production checkpoint missing at {PRODUCTION_CHECKPOINT}"
    with open(PRODUCTION_CHECKPOINT, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_SHA256, (
        f"Production checkpoint SHA-256 mismatch! Expected {EXPECTED_SHA256}, got {file_hash}"
    )


def test_frozen_nih_thresholds():
    """Verify frozen NIH production thresholds match exact clinical reference."""
    from backend.app.services.xray_service import DEFAULT_NIH_THRESHOLDS
    for disease, threshold in FROZEN_THRESHOLDS.items():
        assert disease in DEFAULT_NIH_THRESHOLDS, f"Pathology {disease} missing from model thresholds"
        assert abs(DEFAULT_NIH_THRESHOLDS[disease] - threshold) < 1e-3, (
            f"Threshold for {disease} altered! Expected ~{threshold}, got {DEFAULT_NIH_THRESHOLDS[disease]}"
        )


def test_phase6c_deliverables_completeness():
    """Verify all 9 Phase 6C deliverables exist, are valid, and have sufficient content."""
    assert DELIVERABLES_DIR.exists(), f"Deliverables directory missing: {DELIVERABLES_DIR}"
    
    for filename in EXPECTED_DELIVERABLES:
        filepath = DELIVERABLES_DIR / filename
        assert filepath.exists(), f"Deliverable missing: {filepath}"
        assert filepath.stat().st_size > 500, f"Deliverable {filename} is suspiciously small ({filepath.stat().st_size} bytes)"
        
        if filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict), f"JSON file {filename} does not contain an object"
                assert "title" in data, f"Key 'title' missing from {filename}"
        elif filename.endswith(".md"):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                assert len(text) > 5000, "Markdown report too brief"
                assert "Executive Summary" in text
                assert "Distinction of Audit Findings" in text
                assert "Task A: Checkpoint & Experiment Reproduction Audit" in text
                assert "Task B: Controlled Preprocessing Comparison" in text
                assert "Task C: Multi-Resolution Reproduction" in text
                assert "Task D: Patient-Cluster Bootstrap" in text
                assert "Task E: Multiclass Probability Calibration" in text
                assert "Task F: Operating Point Validation" in text
                assert "Task G: Forensic Error Analysis" in text
                assert "Task H: External Evaluation" in text
                assert "PARTIAL" in text


def test_checkpoint_audit_verdict_partial():
    """Verify audit correctly identifies the missing 512x512 checkpoint and declares PARTIAL."""
    report_path = DELIVERABLES_DIR / "phase6c_checkpoint_reproduction.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["audit_verdict"] == "PARTIAL", f"Expected audit_verdict PARTIAL, got {data['audit_verdict']}"
    exp_info = data["experimental_model_d_candidate"]
    assert exp_info["actual_trained_resolution"] == 224
    assert exp_info["actual_epochs_trained"] == 1
    assert exp_info["dedicated_512_trained_checkpoint_exists"] is False


def test_patient_cluster_statistics_variance_inflation():
    """Verify patient-cluster bootstrapping widens confidence intervals across all diseases."""
    report_path = DELIVERABLES_DIR / "phase6c_patient_cluster_statistics.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["cohort_summary"]["total_radiographs"] == 25596
    assert data["cohort_summary"]["unique_patients"] == 2797
    
    cluster_ci = data["patient_cluster_confidence_intervals"]
    for pathology in PATHOLOGIES:
        assert pathology in cluster_ci
        c_width = cluster_ci[pathology]["patient_cluster_ci_width"]
        n_width = cluster_ci[pathology]["naive_image_ci_width"]
        # Cluster CI must be wider than naive image CI due to positive intra-patient correlation
        assert c_width > n_width, (
            f"Pathology {pathology}: cluster CI width {c_width} <= naive CI width {n_width}"
        )
        assert cluster_ci[pathology]["cluster_variance_inflation_pct"] > 0
        
    paired = data["paired_patient_comparisons"]
    for pathology in PATHOLOGIES:
        assert pathology in paired
        assert "benjamini_hochberg_critical" in paired[pathology]
        assert "significant_after_fdr" in paired[pathology]


def test_calibration_report_validity():
    """Verify multiclass calibration report contains all diagnostic metrics."""
    report_path = DELIVERABLES_DIR / "phase6c_calibration_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    metrics = data["disease_calibration_metrics"]
    for pathology in PATHOLOGIES:
        assert pathology in metrics
        p_data = metrics[pathology]
        assert 0.0 <= p_data["brier_score"] <= 1.0
        assert 0.0 <= p_data["expected_calibration_error"] <= 1.0
        assert "calibration_slope" in p_data
        assert "calibration_intercept" in p_data
        assert len(p_data["reliability_diagram_10_bins"]) == 10


def test_operating_points_validation_with_cis():
    """Verify operating points include patient-cluster bootstrap 95% CIs and notices."""
    report_path = DELIVERABLES_DIR / "phase6c_operating_point_validation.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert "NON-DIAGNOSTIC NOTICE" in data["regulatory_disclaimer"]
    
    evals = data["operating_point_evaluations"]
    for pathology in PATHOLOGIES:
        assert pathology in evals
        p_eval = evals[pathology]
        assert "point_1_frozen_nih_production" in p_eval
        assert "point_2_validation_youden_optimal" in p_eval
        assert "point_3_high_sensitivity_triage" in p_eval
        
        # Verify CI bounds exist and are properly ordered
        for pt_key in ["point_1_frozen_nih_production", "point_2_validation_youden_optimal", "point_3_high_sensitivity_triage"]:
            ci = p_eval[pt_key]["sensitivity_95_ci"]
            assert len(ci) == 2
            assert ci[0] <= ci[1]


def test_external_evaluation_governance_invariants():
    """Verify external evaluation covers independent datasets and enforces CXR-only governance."""
    report_path = DELIVERABLES_DIR / "phase6c_external_evaluation.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["total_external_samples"] == 1202
    assert data["total_external_patients"] == 1200
    assert "VinDr-CXR" in data["external_datasets_evaluated"]
    assert "Stanford CheXpert" in data["external_datasets_evaluated"]
    assert "No external 3D shape reconstruction validation is claimed" in data["critical_governance_invariant"]
