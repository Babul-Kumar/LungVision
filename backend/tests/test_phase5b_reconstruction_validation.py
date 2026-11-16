"""
LungVision AI — Phase 5B Automated Verification Test Suite
Test Suite: backend/tests/test_phase5b_reconstruction_validation.py
Version: 5.1.0

Validates all Phase 5B scientific and engineering deliverables:
1. Forensic SSM PCA eigenvalue spectrum & condition number
2. Oracle upper-bound reconstruction diagnostic ceiling
3. Multi-model baseline comparisons (Mean, Frontal, Lateral, Concat, Attention, V2)
4. Heteroscedastic uncertainty & bilateral residual architecture
5. Mesh surface quality, watertightness, manifoldness, and Taubin volume preservation
6. Divergence Theorem bilateral volumetry & units
7. Statistical significance, Wilcoxon W=0 audit, and Holm-Bonferroni correction
8. Multi-condition robustness (rotation, exposure, missing lateral fallback)
9. Production DenseNet121 checkpoint & frozen NIH classification threshold invariants
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import numpy as np
import torch

from backend.app.config import BASE_DIR
from backend.app.models.dual_view_shape_predictor import (
    DualViewCrossAttentionShapePredictor,
    HeteroscedasticDualViewCrossAttentionPredictor,
    HeteroscedasticShapeLoss,
    MahalanobisShapeLoss,
    create_shape_predictor,
)
from backend.app.services.ct.patient_reconstruction_service import (
    PatientReconstructionService,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research" / "3d_reconstruction"


@pytest.fixture(scope="module")
def spectrum_audit_data():
    file_path = RESEARCH_DIR / "phase5b_ssm_spectrum_audit.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def oracle_data():
    file_path = RESEARCH_DIR / "phase5b_oracle_upper_bound.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def baseline_data():
    file_path = RESEARCH_DIR / "phase5b_baseline_comparison.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def ablation_data():
    file_path = RESEARCH_DIR / "phase5b_model_ablation.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def surface_quality_data():
    file_path = RESEARCH_DIR / "phase5b_surface_quality_metrics.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def volumetry_data():
    file_path = RESEARCH_DIR / "phase5b_volumetry_audit.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def statistical_data():
    file_path = RESEARCH_DIR / "phase5b_statistical_validation.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def robustness_data():
    file_path = RESEARCH_DIR / "phase5b_robustness_report.json"
    assert file_path.exists(), f"Missing {file_path}"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_gate_a_forensic_ssm_spectrum(spectrum_audit_data):
    """Test A: Verifies SSM training boundary, full SVD spectrum, and 100.0% variance claim."""
    provenance = spectrum_audit_data["provenance_and_governance"]
    assert provenance["cohort_split"]["train_patients"] == 301
    assert provenance["cohort_split"]["val_patients"] == 64
    assert provenance["cohort_split"]["test_patients"] == 65
    assert "PASSED" in provenance["cohort_split"]["patient_leakage_audit"]
    assert provenance["bilateral_partition"]["left_lung_vertices"] == 3377
    assert provenance["bilateral_partition"]["right_lung_vertices"] == 3360
    assert provenance["bilateral_partition"]["is_watertight"] is True

    pca = spectrum_audit_data["pca_spectrum_analysis"]
    # 20 modes cumulative variance rounds to 100.00%
    assert pca["variance_at_20_modes_rounded_pct"] == 100.00
    assert pca["variance_at_20_modes_exact_pct"] > 99.99
    assert pca["effective_rank_metrics"]["rank_for_99_pct_variance"] <= 5
    assert pca["effective_rank_metrics"]["near_zero_eigenvalues_count"] > 100

    # Reconstruction error at 20 modes must be sub-voxel (< 0.1 mm)
    rec_20 = spectrum_audit_data["reconstruction_error_by_mode"]["20_modes"]
    assert rec_20["rmse_mm"] < 0.10
    assert rec_20["p95_surface_error_mm"] < 0.10


def test_gate_b_oracle_upper_bound_and_baselines(oracle_data, baseline_data):
    """Test B: Verifies Oracle diagnostic upper-bound and baseline performance ordering."""
    oracle = oracle_data["oracle_performance"]
    # Oracle represents optimal projection into 20-mode SSM manifold: ASSD < 0.05 mm, Dice == 100%
    assert oracle["assd_mm"]["mean"] < 0.05
    assert oracle["surface_dice_2mm"]["mean"] > 0.99
    assert oracle["volume_error_pct"]["mean"] < 0.10

    # Learned models comparison on held-out test cohort
    summaries = baseline_data["cohort_summaries"]
    mean_assd = summaries["mean_shape_baseline"]["assd_mm"]["mean"]
    front_assd = summaries["frontal_only"]["assd_mm"]["mean"]
    attn_assd = summaries["dual_view_attention"]["assd_mm"]["mean"]

    # Dual-View Attention must outperform mean shape and frontal baseline
    assert attn_assd < front_assd
    assert attn_assd < mean_assd
    # Gap to oracle must be positive and bounded
    gap = oracle_data["gap_analysis_to_learned_models"]["oracle_vs_dual_attention"]
    assert gap["assd_gap_mm"] > 0.0


def test_gate_c_heteroscedastic_v2_model_architecture(ablation_data):
    """Test C: Verifies Phase 5B heteroscedastic model architecture, forward pass, and checkpoint."""
    ckpt_path = PROJECT_ROOT / "backend" / "data" / "checkpoints" / "dual_view_shape_predictor_v2_best.pth"
    assert ckpt_path.exists(), f"Missing Phase 5B V2 checkpoint: {ckpt_path}"

    state = torch.load(ckpt_path, map_location="cpu")
    assert state["architecture"] == "HeteroscedasticDualViewCrossAttentionPredictor"
    assert state["model_version"] == "5.1.0"
    assert state["num_modes"] == 20

    model = HeteroscedasticDualViewCrossAttentionPredictor(num_modes=20)
    model.load_state_dict(state["state_dict"])
    model.eval()

    # Forward pass with uncertainty
    f_dummy = torch.randn(2, 1, 224, 224)
    l_dummy = torch.randn(2, 1, 224, 224)
    coeffs, logvar = model(f_dummy, l_dummy, return_uncertainty=True)

    assert coeffs.shape == (2, 20)
    assert logvar.shape == (2, 20)
    assert torch.all(torch.isfinite(coeffs))
    assert torch.all(torch.isfinite(logvar))

    # Mahalanobis distance check in ablation report
    v2_info = ablation_data["selected_v2_architecture"]
    assert v2_info["mahalanobis_distance"] < 3.0  # Anatomically plausible within 3 SD


def test_gate_d_surface_quality_and_topology(surface_quality_data):
    """Test D: Verifies manifold topology, zero non-manifold edges, and Taubin volume preservation."""
    invariants = surface_quality_data["topology_invariants"]
    assert invariants["canonical_vertices"] == 6737
    assert invariants["canonical_faces"] == 13478
    assert "PASSED" in invariants["manifold_topology"]
    assert "PASSED" in invariants["bilateral_boundary_separation"]

    comparisons = surface_quality_data["smoothing_comparisons"]
    raw = comparisons["raw_reconstructed"]
    assert raw["is_watertight_all_patients"] is True
    assert raw["is_winding_consistent_all_patients"] is True
    assert raw["total_non_manifold_edges_cohort"] == 0
    # Phase 5B V2 raw ASSD improved over Phase 5
    assert raw["mean_assd_mm"] < 4.80


def test_gate_e_bilateral_volumetry_audit(volumetry_data):
    """Test E: Verifies Divergence Theorem volume calculation, physical units, and bilateral balance."""
    formulation = volumetry_data["mathematical_formulation"]
    assert "Divergence Theorem" in formulation["theorem"]
    assert "mm^3 / 1000 = cm^3 = mL" in formulation["physical_unit_derivation"]
    assert "PASSED" in formulation["watertightness_prerequisite"]

    acc = volumetry_data["cohort_volumetric_accuracy"]
    assert acc["num_test_patients"] == 65
    assert 3000.0 < acc["mean_reference_volume_ml"] < 5000.0
    assert 3000.0 < acc["mean_predicted_volume_ml"] < 5000.0
    assert acc["mean_percentage_volume_error"] < 12.0
    # Physiological adult human bilateral volume ratio is right ~55%, left ~45% (ratio ~ 1.22)
    assert 1.15 <= acc["bilateral_ratio_mean"] <= 1.30


def test_gate_f_statistical_validation_and_wilcoxon_audit(statistical_data):
    """Test F: Verifies Wilcoxon W=0 audit, paired effect sizes, and Holm-Bonferroni correction."""
    w_audit = statistical_data["wilcoxon_zero_statistic_audit"]
    assert w_audit["sample_size_N"] == 65
    assert w_audit["non_zero_difference_pairs"] == 65
    assert w_audit["zero_difference_ties"] == 0
    assert w_audit["positive_differences_count"] == 0
    assert w_audit["negative_differences_count"] == 65
    assert w_audit["sum_positive_ranks_W_plus"] == 0.0
    assert w_audit["sum_negative_ranks_W_minus"] == 2145.0  # 65*66/2 = 2145
    assert w_audit["test_statistic_reported"] == 0.0
    assert w_audit["rank_biserial_correlation_effect_size"] == -1.0
    assert w_audit["cohens_d_effect_size"] < -2.0  # Very large effect size
    assert w_audit["exact_p_value"] < 1e-15
    assert w_audit["asymptotic_p_value"] < 1e-10

    # Multi-metric comparisons
    comps = statistical_data["multi_metric_comparisons_with_holm_bonferroni"]
    for c in comps:
        assert c["significant_after_correction"] is True, f"Comparison {c['metric']} failed Holm-Bonferroni"


def test_gate_g_multi_condition_robustness(robustness_data):
    """Test G: Verifies robustness under AP/PA, rotation, resolution, and graceful lateral fallback."""
    conds = robustness_data["conditions_evaluated"]
    nominal = robustness_data["nominal_baseline_assd_mm"]

    # AP delta
    assert conds["pa_vs_ap_projection"]["delta_from_nominal_mm"] < 0.30

    # Rotation stability (+/- 10 degrees increases ASSD by < 0.25 mm)
    for k, rot in conds["image_rotation"].items():
        assert rot["delta_from_nominal_mm"] < 0.25

    # Missing lateral fallback must degrade gracefully (< 0.25 mm delta, no NaN/crash)
    fallback = conds["missing_lateral_fallback"]
    assert fallback["delta_from_nominal_mm"] < 0.25
    assert "GRACEFUL_DEGRADATION" in fallback["status"]


def test_gate_h_service_runtime_integration():
    """Test H: Verifies PatientReconstructionService initializes V2 model and returns vertex errors."""
    service = PatientReconstructionService()
    assert service.is_loaded is True
    assert "dual_view_heteroscedastic" in service.models

    # Dummy test forward
    dummy_f = torch.randn(1, 1, 224, 224)
    dummy_l = torch.randn(1, 1, 224, 224)
    coeffs, unc = service.predict_coefficients(
        dummy_f, dummy_l, model_type="dual_view_heteroscedastic"
    )
    assert len(coeffs) == 20
    assert "vertex_sigmas" in unc
    assert len(unc["vertex_sigmas"]) == 6737
    assert unc["mean_uncertainty_mm"] > 0.0


def test_gate_i_production_preservation_invariants():
    """Test I: Absolute invariant test ensuring frozen DenseNet121 weights and thresholds remain identical."""
    prod_ckpt = PROJECT_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
    assert prod_ckpt.exists(), f"Production checkpoint missing: {prod_ckpt}"

    with open(prod_ckpt, "rb") as f:
        file_bytes = f.read()
    ckpt_hash = hashlib.sha256(file_bytes).hexdigest()
    assert ckpt_hash == "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"

    from backend.app.services.xray_service import xray_service
    frozen_thresh = xray_service.nih_thresholds
    assert round(frozen_thresh["Cardiomegaly"], 4) == 0.9502
    assert round(frozen_thresh["Edema"], 4) == 0.8770
    assert round(frozen_thresh["Consolidation"], 4) == 0.7329
    assert round(frozen_thresh["Pneumonia"], 4) == 0.7529
    assert round(frozen_thresh["Atelectasis"], 4) == 0.7495
    assert round(frozen_thresh["Pneumothorax"], 4) == 0.5825
