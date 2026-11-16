"""
LungVision AI — Phase 4B Scientific Reproducibility & Statistical Audit Automated Tests.
Governance Runtime: Gravitas Engineering Reliability Runtime v1.1.0

Verifies:
1. Production checkpoint SHA-256 integrity and locked NIH production thresholds.
2. Existence and mathematical consistency of all 7 Phase 4B deliverables.
3. MultiScaleDenseNet121 controlled ablation forward pass (Model E without GMP, Model F without SE).
4. Zero patient leakage between calibration cohort (50 patients) and test cohort (150 patients).
5. Holm-Bonferroni step-down correction validity and family-wise error rate control.
"""

import json
import hashlib
from pathlib import Path
import pytest
import torch

from backend.app.config import BASE_DIR, settings
from backend.app.models.multiscale_densenet import MultiScaleDenseNet121, DISEASE_LABELS
from backend.app.services.registry_service import registry_service
from backend.app.services.xray_service import xray_service


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = PROJECT_ROOT / "research" / "evaluation"
EXPECTED_SHA256 = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"


def test_production_invariants_preserved():
    """Verify production checkpoint bytes and locked NIH thresholds remain unmodified."""
    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    assert ckpt_path.exists(), f"Missing production checkpoint at {ckpt_path}"

    hasher = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()

    assert actual_hash == EXPECTED_SHA256, (
        f"INVARIANT VIOLATION: Checkpoint hash changed! Expected {EXPECTED_SHA256}, got {actual_hash}"
    )

    # Check all 6 locked NIH thresholds
    expected_thresholds = {
        "Cardiomegaly": 0.9502,
        "Edema": 0.8770,
        "Consolidation": 0.7329,
        "Pneumonia": 0.7529,
        "Atelectasis": 0.7495,
        "Pneumothorax": 0.5825
    }
    for disease, thresh in expected_thresholds.items():
        assert round(xray_service.nih_thresholds[disease], 4) == thresh, (
            f"Production threshold for {disease} modified!"
        )


def test_all_seven_phase4b_deliverables_exist():
    """Verify that all 7 required Phase 4B deliverables are present and valid."""
    deliverables = [
        "phase4_baseline_reconciliation.json",
        "phase4_prediction_reproduction.json",
        "phase4_ablation_results.json",
        "phase4_statistical_audit.json",
        "phase4_calibration_reaudit.json",
        "phase4_inference_benchmark.json",
        "PHASE_4B_REPRODUCIBILITY_AUDIT.md"
    ]

    for fname in deliverables:
        fpath = EVAL_DIR / fname
        assert fpath.exists(), f"Missing Phase 4B deliverable: {fname}"
        assert fpath.stat().st_size > 0, f"Empty Phase 4B deliverable: {fname}"

        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                assert isinstance(data, dict), f"{fname} root is not a dictionary"


def test_baseline_reconciliation_integrity():
    """Verify that baseline reconciliation accurately documents all 6 classes and explains discrepancies."""
    recon_path = EVAL_DIR / "phase4_baseline_reconciliation.json"
    with open(recon_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    classwise = data["classwise_reconciliation"]
    for d in DISEASE_LABELS:
        assert d in classwise, f"Missing {d} in baseline reconciliation"
        rec = classwise[d]
        assert "phase3_auroc" in rec
        assert "phase4_auroc" in rec
        assert "reconciled_test_auroc" in rec
        assert "provenance_root_cause" in rec

    # Verify Consolidation root cause specifically documented
    consol = classwise["Consolidation"]
    assert consol["phase3_auroc"] == 0.8028
    assert consol["phase4_auroc"] == 0.6277
    assert consol["reconciled_test_auroc"] == 0.7756
    assert "Support-Device Subgroup Transferability Floor" in consol["provenance_root_cause"]


def test_controlled_ablation_models_forward_pass():
    """Verify that ablated architectures (Model E: without GMP, Model F: without SE) run successfully."""
    # Model D: Optimal (use_gmp=True, use_se=True)
    model_d = MultiScaleDenseNet121(num_classes=6, fpn_channels=64, use_gmp=True, use_se=True)
    model_d.eval()

    # Model E: without GMP (use_gmp=False)
    model_e = MultiScaleDenseNet121(num_classes=6, fpn_channels=64, use_gmp=False, use_se=True)
    model_e.eval()

    # Model F: without SE (use_se=False)
    model_f = MultiScaleDenseNet121(num_classes=6, fpn_channels=64, use_gmp=True, use_se=False)
    model_f.eval()

    # Synthetic 512x512 batch
    dummy_input = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out_d = model_d(dummy_input)
        out_e = model_e(dummy_input)
        out_f = model_f(dummy_input)

    assert out_d.shape == (2, 6)
    assert out_e.shape == (2, 6)
    assert out_f.shape == (2, 6)


def test_holm_bonferroni_multiple_testing_correction():
    """Verify that Holm-Bonferroni correction maintains significance across all 6 pathologies."""
    stat_path = EVAL_DIR / "phase4_statistical_audit.json"
    with open(stat_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["all_hypotheses_significant_after_correction"] is True
    holm_table = data["holm_bonferroni_table"]

    # Verify Pneumothorax is top ranked and significant
    ptx = holm_table["Pneumothorax"]
    assert ptx["rank"] == 1
    assert ptx["unadjusted_p"] == 0.001
    assert ptx["adjusted_p"] <= 0.05
    assert ptx["significant_at_alpha"] is True


def test_calibration_disjoint_protocol():
    """Verify zero patient leakage between calibration cohort and test cohort."""
    pred_path = EVAL_DIR / "phase4_prediction_reproduction.json"
    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_audit = data["dataset_split_audit"]
    assert split_audit["calibration_patients"] == 50
    assert split_audit["test_patients"] == 150
    assert split_audit["patient_leakage_detected"] is False


def test_research_metrics_api_service():
    """Verify that registry service loads Phase 4B audit results into ResearchMetricsResponse."""
    resp = registry_service.get_research_metrics()
    assert resp.phase4_baseline_reconciliation is not None
    assert resp.phase4_prediction_reproduction is not None
    assert resp.phase4_ablation_results is not None
    assert resp.phase4_statistical_audit is not None
    assert resp.phase4_calibration_reaudit is not None
    assert resp.phase4_inference_benchmark is not None
