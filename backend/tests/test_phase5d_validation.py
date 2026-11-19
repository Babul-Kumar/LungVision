#!/usr/bin/env python3
"""
LungVision AI — Phase 5D Verification & Validation Test Suite
File: backend/tests/test_phase5d_validation.py

Verifies all acceptance criteria for Phase 5D:
- Preservation of DenseNet121 production invariants and frozen NIH thresholds
- Verification of Dual-View Predictor V2 and SSM PCA v2 invariants
- Existence and non-emptiness of all 9 Phase 5D deliverables
- Gate A: Test set independence audit & methodological classification
- Gate B: External dataset feasibility audit & BLOCKED status
- Gate C: Split-conformal uncertainty calibration & patient-cluster bootstrap metrics
- Gate D: Anatomical error stratification across volume, asymmetry, aspect ratio, depth
- Gate E: Missing-view and acquisition robustness audit
- Gate F & G: Reproducibility manifest and PARTIAL status declaration
"""

import csv
import hashlib
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_DIR = REPO_ROOT / "research" / "3d_reconstruction"
BACKEND_DIR = REPO_ROOT / "backend"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestPhase5DProductionInvariants:
    """Ensures production assets and frozen thresholds remain completely unaltered."""

    def test_production_densenet121_checkpoint_hash(self):
        primary_ckpt = REPO_ROOT / "models" / "densenet121" / "densenet121_nih_best.pth"
        legacy_ckpt = REPO_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
        ckpt_path = primary_ckpt if primary_ckpt.exists() else legacy_ckpt
        assert ckpt_path.exists(), f"Production checkpoint missing: {ckpt_path}"
        expected_sha = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"
        actual_sha = compute_sha256(ckpt_path)
        assert actual_sha == expected_sha, f"Production checkpoint altered! {actual_sha} != {expected_sha}"

    def test_production_frozen_nih_thresholds(self):
        from backend.app.services.xray_service import XRayService
        svc = XRayService()
        expected = {
            "Cardiomegaly": 0.9502,
            "Edema": 0.8770,
            "Consolidation": 0.7329,
            "Pneumonia": 0.7529,
            "Atelectasis": 0.7495,
            "Pneumothorax": 0.5825
        }
        for k, exp_val in expected.items():
            actual = round(float(svc.nih_thresholds[k]), 4)
            assert actual == exp_val, f"Threshold altered for {k}: {actual} != {exp_val}"

    def test_v2_predictor_and_ssm_invariants(self):
        ckpt_v2 = BACKEND_DIR / "data" / "checkpoints" / "dual_view_shape_predictor_v2_best.pth"
        assert ckpt_v2.exists(), "V2 predictor checkpoint missing"
        actual_v2_sha = compute_sha256(ckpt_v2)
        expected_v2_sha = "dd592d21d3d9f68587a88184b1f9275089fc037dbafff5e9e6c5a0114b5be413"
        assert actual_v2_sha == expected_v2_sha, f"V2 predictor hash mismatch: {actual_v2_sha}"

        pca_path = BACKEND_DIR / "data" / "3d_reconstruction" / "ssm_lung_pca_v2.json"
        assert pca_path.exists(), "SSM PCA v2 artifact missing"


class TestPhase5DDeliverablesExistence:
    """Verifies that all 9 required Phase 5D deliverables exist and are non-empty."""

    REQUIRED_DELIVERABLES = [
        "phase5d_test_set_independence_audit.json",
        "phase5d_external_dataset_feasibility.json",
        "phase5d_uncertainty_calibration.json",
        "phase5d_patient_level_uncertainty_metrics.csv",
        "phase5d_anatomical_stratification.json",
        "phase5d_failure_analysis.json",
        "phase5d_missing_view_robustness.json",
        "phase5d_reproducibility_manifest.json",
        "PHASE_5D_EXTERNAL_VALIDATION_REPORT.md"
    ]

    @pytest.mark.parametrize("filename", REQUIRED_DELIVERABLES)
    def test_deliverable_exists_and_non_empty(self, filename):
        path = RESEARCH_DIR / filename
        assert path.exists(), f"Required Phase 5D deliverable missing: {filename}"
        assert path.stat().st_size > 100, f"Deliverable too small/empty: {filename} ({path.stat().st_size} bytes)"


class TestPhase5DScientificAudits:
    """Verifies substantive mathematical and scientific properties of Phase 5D deliverables."""

    def test_gate_a_test_set_independence(self):
        fpath = RESEARCH_DIR / "phase5d_test_set_independence_audit.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        cfg = d["split_configuration"]
        assert cfg["training_cohort_size"] == 301
        assert cfg["validation_cohort_size"] == 64
        assert cfg["test_cohort_size"] == 65
        assert cfg["unassigned_patients_in_dataset"] == 0

        mc = d["methodological_classification"]
        assert mc["classification"] == "ITERATIVE_BENCHMARK_EVALUATION_COHORT"
        assert mc["is_pristine_blind_test_cohort"] is False
        assert mc["independent_cohort_availability"]["independent_cohort_available_in_repo"] is False

    def test_gate_b_external_dataset_feasibility(self):
        fpath = RESEARCH_DIR / "phase5d_external_dataset_feasibility.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert len(d["candidate_datasets"]) >= 6
        assert d["summary"]["datasets_with_simultaneous_real_dual_view_cxr_and_3d_ct_segmentations"] == 0
        assert d["summary"]["external_reconstruction_validation_status"] == "BLOCKED_NO_PUBLIC_PAIRED_COHORTS_EXIST"

    def test_gate_c_uncertainty_calibration(self):
        fpath = RESEARCH_DIR / "phase5d_uncertainty_calibration.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert d["partition_strategy"]["calibration_cohort_size"] == 64
        assert d["partition_strategy"]["test_cohort_size"] == 65

        m1 = d["method_comparison_on_test_cohort"]["method1_baseline_uncalibrated"]
        assert m1["coverage_pct"]["95"] < 5.0, "Uncalibrated raw uncertainty should be underconfident"

        m2 = d["method_comparison_on_test_cohort"]["method2_variance_temperature_scaling"]
        assert m2["coverage_pct"]["95"] >= 85.0, "Temperature scaling should achieve >= 85% coverage"

        m4a = d["method_comparison_on_test_cohort"]["method4a_split_conformal_marginal"]
        assert m4a["coverage_pct"]["95"] >= 80.0, "Marginal conformal should achieve >= 80% coverage"

        m4b = d["method_comparison_on_test_cohort"]["method4b_split_conformal_simultaneous_surface"]
        assert m4b["patient_level_surface_coverage_pct"]["95"] >= 80.0, "Simultaneous surface coverage should exceed 80%"

        ece = d["statistical_properties"]["calibration_error"]
        assert ece["calibrated_ece_mm"] < ece["uncalibrated_ece_mm"]
        assert ece["ece_reduction_pct"] > 50.0

        # Verify CSV has exactly 65 patient rows
        csv_path = RESEARCH_DIR / "phase5d_patient_level_uncertainty_metrics.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 65

    def test_gate_d_anatomical_stratification(self):
        fpath = RESEARCH_DIR / "phase5d_anatomical_stratification.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert 4.60 <= d["overall_test_metrics"]["mean_assd_mm"] <= 4.80

        vol = d["stratification_by_volume"]
        normal_assd = vol["medium_3500_4500ml"]["assd_mean_mm"]
        small_assd = vol["small_lt_3500ml"]["assd_mean_mm"]
        large_assd = vol["large_gt_4500ml"]["assd_mean_mm"]
        assert normal_assd < small_assd, "Normal volume lung shape should have lower ASSD than small volume"
        assert normal_assd < large_assd, "Normal volume lung shape should have lower ASSD than large volume"

        # Check failure analysis
        fail_path = RESEARCH_DIR / "phase5d_failure_analysis.json"
        with open(fail_path, "r", encoding="utf-8") as f:
            d_fail = json.load(f)
        assert len(d_fail["top_5_outlier_cases"]) == 5
        assert d_fail["top_5_outlier_cases"][0]["assd_mm"] >= 8.5

    def test_gate_e_missing_view_robustness(self):
        fpath = RESEARCH_DIR / "phase5d_missing_view_robustness.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert 4.60 <= d["baseline_reference"]["assd_mean_mm"] <= 4.80

        # Check single-view models are UNAVAILABLE_EXPERIMENT
        sv = d["single_view_architectural_models"]
        assert sv["frontal_only_trained_model"]["status"] == "UNAVAILABLE_EXPERIMENT"
        assert sv["lateral_only_trained_model"]["status"] == "UNAVAILABLE_EXPERIMENT"

    def test_gate_f_g_manifest_and_verdict(self):
        fpath = RESEARCH_DIR / "phase5d_reproducibility_manifest.json"
        with open(fpath, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert d["production_invariants"]["intact"] is True
        assert d["model_and_ssm_invariants"]["v2_intact"] is True
        assert d["final_status_declaration"]["final_verdict"] == "PARTIAL"
        assert "UNVALIDATED and BLOCKED" in d["final_status_declaration"]["explicit_unvalidated_statement"]
