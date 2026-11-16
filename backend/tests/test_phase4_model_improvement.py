"""
LungVision AI — Phase 4 Model Improvement Test Suite.

Verifies:
1. Production invariants: Checkpoint hash and locked NIH thresholds remain untouched.
2. MultiScaleDenseNet121 architecture forward pass across resolutions (224x224, 384x384, 512x512).
3. Intermediate multi-scale feature extraction (P2, P3, P4) and dual pooling integrity.
4. Loss function calculations (FocalLoss, ClassBalancedLoss, AsymmetricLoss).
5. All 8 Phase 4 deliverable files exist in research/evaluation/ with valid contents.
6. Experimental model registration with status="research_experimental".
7. Research metrics API endpoint returns Phase 4 fields.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR, settings
from backend.app.main import app
from backend.app.models.multiscale_densenet import MultiScaleDenseNet121, DISEASE_LABELS
from backend.app.models.loss_functions import FocalLoss, ClassBalancedLoss, AsymmetricLoss
from backend.app.services.registry_service import registry_service
from backend.app.services.xray_service import xray_service

client = TestClient(app)


def test_production_invariants_preserved():
    """Confirms production DenseNet121 checkpoint and thresholds are strictly immutable."""
    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    assert ckpt_path.exists(), f"Production checkpoint missing at {ckpt_path}"

    with open(ckpt_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    expected_hash = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"
    assert actual_hash == expected_hash, "Production checkpoint bytes have been altered!"

    # Verify locked production NIH thresholds
    expected_thresholds = {
        "Cardiomegaly": 0.9502,
        "Edema": 0.8770,
        "Consolidation": 0.7329,
        "Pneumonia": 0.7529,
        "Atelectasis": 0.7495,
        "Pneumothorax": 0.5825
    }
    rounded_thresholds = {k: round(v, 4) for k, v in xray_service.nih_thresholds.items()}
    assert rounded_thresholds == expected_thresholds, "Production NIH thresholds modified!"


def test_multiscale_densenet_forward_pass_across_resolutions():
    """Verifies MultiScaleDenseNet121 outputs exactly 6 logits across all experimental resolutions."""
    model = MultiScaleDenseNet121(num_classes=6, fpn_channels=256)
    model.eval()

    for size in [224, 384, 512]:
        x = torch.randn(2, 3, size, size)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 6), f"Failed for size {size}: expected (2, 6), got {logits.shape}"
        assert torch.isfinite(logits).all(), f"Non-finite logits detected at resolution {size}"


def test_multiscale_intermediate_feature_maps():
    """Verifies P2 (stride 8), P3 (stride 16), and P4 (stride 32) feature dimensions."""
    model = MultiScaleDenseNet121(num_classes=6, fpn_channels=256)
    model.eval()

    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        p2, p3, p4 = model.extract_multiscale_features(x)

    # At 512x512: P2 -> 64x64, P3 -> 32x32, P4 -> 16x16
    assert p2.shape == (1, 256, 64, 64), f"Unexpected P2 shape: {p2.shape}"
    assert p3.shape == (1, 256, 32, 32), f"Unexpected P3 shape: {p3.shape}"
    assert p4.shape == (1, 256, 16, 16), f"Unexpected P4 shape: {p4.shape}"


def test_loss_functions_computation():
    """Verifies FocalLoss, ClassBalancedLoss, and AsymmetricLoss formulations."""
    logits = torch.randn(4, 6)
    targets = torch.tensor([
        [1, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ], dtype=torch.float32)

    # 1. Focal Loss
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    loss_fl = fl(logits, targets)
    assert torch.isfinite(loss_fl) and loss_fl.item() > 0.0

    # 2. Class-Balanced Loss
    cb = ClassBalancedLoss(samples_per_class=[100, 50, 40, 20, 80, 10], beta=0.9999)
    loss_cb = cb(logits, targets)
    assert torch.isfinite(loss_cb) and loss_cb.item() > 0.0

    # 3. Asymmetric Loss
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0)
    loss_asl = asl(logits, targets)
    assert torch.isfinite(loss_asl) and loss_asl.item() > 0.0


def test_phase4_deliverable_artifacts_exist():
    """Verifies all 8 Phase 4 deliverables exist in research/evaluation/."""
    eval_dir = BASE_DIR / "research" / "evaluation"

    required_files = [
        "phase4_experiment_manifest.json",
        "resolution_comparison.json",
        "multiscale_architecture_report.md",
        "loss_function_comparison.json",
        "phase4_classwise_metrics.json",
        "phase4_calibration_report.json",
        "phase4_error_analysis.md",
        "PHASE_4_MODEL_IMPROVEMENT_REPORT.md"
    ]

    for fname in required_files:
        fpath = eval_dir / fname
        assert fpath.exists(), f"Phase 4 deliverable missing: {fname}"
        assert fpath.stat().st_size > 100, f"Phase 4 deliverable file too small or empty: {fname}"

    # Verify key JSON content
    with open(eval_dir / "resolution_comparison.json", "r", encoding="utf-8") as f:
        res_data = json.load(f)
        assert "512x512_multiscale_fpn" in res_data["resolutions_evaluated"]
        assert res_data["reproduction_findings"]["reported_pneumothorax_auroc_baseline"] == 0.6132
        assert res_data["reproduction_findings"]["reproduced_pneumothorax_auroc_highres_multiscale"] == 0.7690

    with open(eval_dir / "phase4_classwise_metrics.json", "r", encoding="utf-8") as f:
        cls_data = json.load(f)
        assert "multiscale_densenet121_fpn_512" in cls_data["models_evaluated"]
        assert "macro_summary" in cls_data


def test_research_experimental_model_registration():
    """Verifies the experimental multi-scale model is registered with status='research_experimental'."""
    models_resp = registry_service.get_models()
    model_ids = [m.id for m in models_resp.models]
    assert "densenet121_multiscale_highres" in model_ids

    exp_model = registry_service.get_model_by_id("densenet121_multiscale_highres")
    assert exp_model.status == "research_experimental"
    assert exp_model.architecture == "MultiScaleDenseNet121-FPN"
    assert exp_model.input_size == [512, 512, 3]
    assert exp_model.labels == DISEASE_LABELS


def test_research_metrics_api_endpoint():
    """Verifies that /api/v1/research/metrics returns populated Phase 4 research fields."""
    response = client.get("/api/v1/research/metrics")
    assert response.status_code == 200
    data = response.json()

    assert data.get("phase4_experiment_manifest") is not None
    assert data.get("resolution_comparison") is not None
    assert data.get("loss_function_comparison") is not None
    assert data.get("phase4_classwise_metrics") is not None
    assert data.get("phase4_calibration_report") is not None
    assert "0.7690" in str(data.get("notes"))
