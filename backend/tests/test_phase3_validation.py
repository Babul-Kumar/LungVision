import hashlib
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR, settings
from backend.app.main import app

client = TestClient(app)
EVAL_DIR = BASE_DIR / "research" / "evaluation"


def test_phase3_deliverables_exist():
    """Verifies that all 8 required Phase 3 research deliverables exist on disk."""
    required_files = [
        EVAL_DIR / "expanded_external_dataset_manifest.csv",
        EVAL_DIR / "external_cohort_comparison.json",
        EVAL_DIR / "threshold_optimization_report.json",
        EVAL_DIR / "uncertain_label_sensitivity_analysis.json",
        EVAL_DIR / "calibration_analysis.json",
        EVAL_DIR / "resolution_ablation_results.json",
        EVAL_DIR / "error_analysis_report.md",
        EVAL_DIR / "PHASE_3_EXTERNAL_VALIDATION_REPORT.md",
    ]
    for path in required_files:
        assert path.exists(), f"Phase 3 deliverable missing: {path}"
        assert path.stat().st_size > 50, f"Phase 3 deliverable empty: {path}"


def test_production_checkpoint_hash_invariant():
    """Verifies that the DenseNet121 production checkpoint hash remains strictly frozen."""
    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    h = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    assert h == "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af", "Checkpoint hash altered!"


def test_production_thresholds_invariant():
    """Verifies that the live backend enforces the frozen NIH production thresholds."""
    from backend.app.services.xray_service import XRayService
    service = XRayService()
    expected = {
        "Cardiomegaly": 0.9502,
        "Edema": 0.8770,
        "Consolidation": 0.7329,
        "Pneumonia": 0.7529,
        "Atelectasis": 0.7495,
        "Pneumothorax": 0.5825
    }
    for disease, thresh in expected.items():
        assert pytest.approx(service.nih_thresholds[disease], rel=1e-3) == thresh


def test_threshold_optimization_report_integrity():
    """Verifies Operating Points A, B, and C integrity on held-out test cohort."""
    path = EVAL_DIR / "threshold_optimization_report.json"
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["zero_patient_leakage"] is True
    assert data["test_cohort_size"] >= 150
    assert "operating_point_A_locked_nih" in data["operating_points"]
    assert "operating_point_B_external_youden" in data["operating_points"]
    assert "operating_point_C_high_sensitivity" in data["operating_points"]

    # Point C must have higher sensitivity than Point A
    macro_sens_a = data["operating_points"]["operating_point_A_locked_nih"]["macro_averages"]["macro_sensitivity"]
    macro_sens_c = data["operating_points"]["operating_point_C_high_sensitivity"]["macro_averages"]["macro_sensitivity"]
    assert macro_sens_c > macro_sens_a
    assert macro_sens_c >= 0.80  # Meets >=80% sensitivity requirement


def test_temperature_scaling_calibration_integrity():
    """Verifies temperature scaling convergence and ECE reduction."""
    path = EVAL_DIR / "calibration_analysis.json"
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["patient_leakage"] == 0.0
    assert data["macro_summary"]["macro_post_scaling_ece"] <= data["macro_summary"]["macro_pre_scaling_ece"]
    for disease in ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"]:
        d_cal = data["classwise_calibration"][disease]
        assert d_cal["optimal_temperature_T"] > 0.0
        assert len(d_cal["test_post_scaling"]["reliability_bins"]) == 10


def test_uncertainty_analysis_integrity():
    """Verifies that U-Ignore, U-Ones, and U-Zeros are analyzed for all 6 pathologies."""
    path = EVAL_DIR / "uncertain_label_sensitivity_analysis.json"
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_records"] > 100000
    for disease in ["Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"]:
        d_entry = data["classwise_uncertainty_analysis"][disease]
        strat = d_entry["prevalence_by_strategy"]
        assert strat["strategy_2_u_ones"] >= strat["strategy_3_u_zeros"]


def test_resolution_ablation_integrity():
    """Verifies resolution ablations and alternative loss functions are documented."""
    path = EVAL_DIR / "resolution_ablation_results.json"
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)

    res_levels = data["resolution_levels"]
    assert "224x224" in res_levels
    assert "512x512" in res_levels
    assert res_levels["512x512"]["pneumothorax_auroc"] > res_levels["224x224"]["pneumothorax_auroc"]
    assert "focal_loss_gamma_2" in data["loss_function_comparison"]


def test_api_research_metrics_endpoint_returns_phase3_data():
    """Verifies that /api/v1/research/metrics surfaces the Phase 3 research artifacts."""
    response = client.get("/api/v1/research/metrics")
    assert response.status_code == 200
    body = response.json()

    assert "expanded_cohort_comparison" in body
    assert body["expanded_cohort_comparison"] is not None
    assert "threshold_optimization_report" in body
    assert body["threshold_optimization_report"] is not None
    assert "uncertain_label_sensitivity" in body
    assert body["uncertain_label_sensitivity"] is not None
    assert "calibration_analysis" in body
    assert body["calibration_analysis"] is not None
    assert "resolution_ablation" in body
    assert body["resolution_ablation"] is not None
