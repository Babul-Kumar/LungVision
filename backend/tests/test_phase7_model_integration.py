"""
Unit, integration, and regression test suite for Phase 7 MultiScaleDenseNet121 integration.

Verifies:
1. Exact checkpoint SHA-256 integrity and strict state-dict parameter compatibility (8,071,942 parameters).
2. Deterministic preprocessing parity against reference training transform (atol=1e-6).
3. Clinical image quality assurance gates (dimension, blur, contrast, decodability).
4. Prediction engine forward pass, calibration thresholds, uncertainty quantification, and Grad-CAM.
5. Central model registry inclusion and operational health endpoints.
6. API routes: POST /api/v1/predictions, GET /api/v1/model/health, GET /api/v1/model/info,
   and backwards-compatible POST /api/v1/xray/predict.
"""

import hashlib
import io
from pathlib import Path
import pytest
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR, settings
from backend.app.main import app
from backend.app.models.multiscale_densenet import (
    DISEASE_LABELS,
    MultiScaleDenseNet121,
    Phase7MultiScaleDenseNet121
)
from backend.app.inference.calibration import (
    OperatingPoint,
    PHASE7_F1_OPTIMAL_THRESHOLDS,
    calibration_manager
)
from backend.app.inference.gradcam import gradcam_generator
from backend.app.inference.prediction_engine import prediction_engine
from backend.app.inference.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_phase7_transform,
    phase7_preprocessor
)
from backend.app.inference.quality_checker import (
    QualityStatus,
    quality_checker
)
from backend.app.inference.uncertainty import (
    ConfidenceLevel,
    uncertainty_analyzer
)
from backend.app.services.model_loader import model_loader
from backend.app.services.registry_service import registry_service


EXPECTED_SHA256 = "7dc735ee00d9e57ece14cd33f2bdbd1e7fd226adca41cdb692ba0af2a33a9b23"
CHECKPOINT_PATH = BASE_DIR / "models" / "multiscale_densenet121" / "msd121_224_weighted_bce_direct.pth"


# =============================================================================
# 1. CHECKPOINT INTEGRITY & ARCHITECTURAL VERIFICATION
# =============================================================================

def test_phase7_checkpoint_exists_and_sha256_matches():
    """Verify Phase 7 checkpoint exists and matches byte-level SHA-256."""
    assert CHECKPOINT_PATH.exists(), f"Missing Phase 7 checkpoint at {CHECKPOINT_PATH}"

    hasher = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    computed_sha = hasher.hexdigest()
    assert computed_sha == EXPECTED_SHA256, (
        f"SHA-256 mismatch! Computed: {computed_sha}, Expected: {EXPECTED_SHA256}"
    )


def test_phase7_architecture_and_parameter_count():
    """Verify MultiScaleDenseNet121 has exactly 8,071,942 parameters and loads strictly."""
    model = Phase7MultiScaleDenseNet121(pretrained=False, dropout=0.25)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 8071942, f"Expected 8071942 parameters, got {trainable_params}"

    # Verify strict state-dict loading with zero missing/unexpected keys
    ckpt_data = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0, f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    # Verify all parameters are finite
    for name, p in model.named_parameters():
        assert torch.isfinite(p).all(), f"Non-finite values found in {name}"


def test_phase7_factory_dispatch():
    """Verify MultiScaleDenseNet121 factory returns Phase7MultiScaleDenseNet121 by default."""
    model = MultiScaleDenseNet121()
    assert isinstance(model, Phase7MultiScaleDenseNet121)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_params == 8071942


# =============================================================================
# 2. PREPROCESSING PARITY VERIFICATION
# =============================================================================

def test_phase7_preprocessing_parity():
    """Verify production Phase7Preprocessor produces identical tensors to training transform."""
    # Reference transform from notebook Cell 14
    ref_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    np.random.seed(123)
    sample_array = np.random.randint(20, 230, (512, 512, 3), dtype=np.uint8)
    sample_pil = Image.fromarray(sample_array)

    ref_tensor = ref_transform(sample_pil).unsqueeze(0)
    prod_tensor = phase7_preprocessor.preprocess(sample_pil)

    assert ref_tensor.shape == (1, 3, 224, 224)
    assert prod_tensor.shape == (1, 3, 224, 224)
    assert torch.allclose(ref_tensor, prod_tensor, atol=1e-6), "Preprocessing parity violation!"


# =============================================================================
# 3. CLINICAL IMAGE QUALITY AUDITING
# =============================================================================

def test_image_quality_checker_valid_image():
    """Verify realistic radiographic test image passes quality checks."""
    np.random.seed(42)
    arr = np.random.randint(40, 200, (400, 400), dtype=np.uint8)
    report = quality_checker.evaluate(Image.fromarray(arr))
    assert report.can_proceed is True
    assert report.status in (QualityStatus.PASSED, QualityStatus.WARNING)


def test_image_quality_checker_rejections():
    """Verify quality checker blocks low-res, corrupt, or blank inputs."""
    # 1. Under minimum resolution (< 128x128)
    tiny_arr = np.random.randint(40, 200, (64, 64), dtype=np.uint8)
    rep_tiny = quality_checker.evaluate(Image.fromarray(tiny_arr))
    assert rep_tiny.can_proceed is False
    assert rep_tiny.status == QualityStatus.IMAGE_QUALITY_INSUFFICIENT

    # 2. Uniform blank image (zero contrast)
    blank_arr = np.full((256, 256), 128, dtype=np.uint8)
    rep_blank = quality_checker.evaluate(Image.fromarray(blank_arr))
    assert rep_blank.can_proceed is False
    assert rep_blank.status == QualityStatus.IMAGE_QUALITY_INSUFFICIENT

    # 3. Corrupt bitstream
    rep_corrupt = quality_checker.evaluate(b"invalid_non_image_bytes")
    assert rep_corrupt.can_proceed is False
    assert rep_corrupt.status == QualityStatus.IMAGE_QUALITY_INSUFFICIENT


# =============================================================================
# 4. PREDICTION ENGINE & GRAD-CAM EXPLAINABILITY
# =============================================================================

def test_prediction_engine_inference_and_uncertainty():
    """Verify prediction engine executes end-to-end forward pass with calibrated thresholds."""
    np.random.seed(77)
    arr = np.random.randint(30, 220, (300, 300, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr)

    res = prediction_engine.predict(pil_img, generate_cam=True)

    assert res["status"] == "COMPLETED"
    assert "findings" in res
    assert len(res["findings"]) == 6
    assert list(res["findings"].keys()) == DISEASE_LABELS

    # Verify each disease output contract
    for disease in DISEASE_LABELS:
        item = res["findings"][disease]
        assert 0.0 <= item["probability"] <= 1.0
        assert item["threshold"] == PHASE7_F1_OPTIMAL_THRESHOLDS[disease]
        assert item["is_positive"] == (item["probability"] >= item["threshold"])
        assert item["confidence_level"] in [c.value for c in ConfidenceLevel]

    # Verify uncertainty synthesis
    unc = res["uncertainty"]
    assert "overall_confidence" in unc
    assert "recommendation" in unc
    assert "mean_margin" in unc

    # Verify Grad-CAM overlay
    assert res["gradcam"] is not None
    assert res["gradcam"]["overlay_base64"].startswith("data:image/jpeg;base64,")


# =============================================================================
# 5. MODEL REGISTRY INTEGRATION
# =============================================================================

def test_model_registry_includes_phase7():
    """Verify centralized model registry includes verified Phase 7 model entry."""
    entry = registry_service.get_model_by_id("lungvision_phase7_msd121_224_direct")
    assert entry.id == "lungvision_phase7_msd121_224_direct"
    assert entry.architecture == "MultiScaleDenseNet121"
    assert entry.status == "production_validated"
    assert entry.sha256 == EXPECTED_SHA256
    assert entry.metrics["macro_val_auc"] == pytest.approx(0.819726, abs=1e-5)
    assert entry.metrics["parameters"] == 8071942


# =============================================================================
# 6. FASTAPI ENDPOINT INTEGRATION TESTS
# =============================================================================

def test_phase7_api_endpoints():
    """Verify HTTP endpoints for predictions, health, and info."""
    with TestClient(app) as client:
        # 1. Health endpoint
        r_health = client.get("/api/v1/model/health")
        assert r_health.status_code == 200
        health_data = r_health.json()
        assert health_data["status"] == "healthy"
        assert health_data["checkpoint_sha256"] == EXPECTED_SHA256

        # 2. Info endpoint
        r_info = client.get("/api/v1/model/info")
        assert r_info.status_code == 200
        info_data = r_info.json()
        assert info_data["architecture"] == "MultiScaleDenseNet121"
        assert info_data["validation_macro_auroc"] == pytest.approx(0.819726, abs=1e-5)

        # 3. Create test image byte stream
        img = Image.fromarray(np.random.randint(40, 220, (256, 256, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # 4. POST /api/v1/predictions
        r_pred = client.post(
            "/api/v1/predictions",
            files={"file": ("test_chest_xray.png", img_bytes, "image/png")},
            data={"generate_cam": "true", "operating_point": "operating_point_A_f1_optimal"}
        )
        assert r_pred.status_code == 200
        pred_data = r_pred.json()
        assert pred_data["status"] == "COMPLETED"
        assert len(pred_data["findings"]) == 6
        assert pred_data["gradcam"] is not None

        # 5. POST /api/v1/xray/predict with Phase 7
        r_xray_p7 = client.post(
            "/api/v1/xray/predict",
            files={"file": ("test_chest_xray.png", img_bytes, "image/png")},
            data={"model_id": "lungvision_phase7_msd121_224_direct"}
        )
        assert r_xray_p7.status_code == 200
        assert r_xray_p7.json()["model"]["id"] == "lungvision_phase7_msd121_224_direct"

        # 6. Backwards compatibility: POST /api/v1/xray/predict with DenseNet121 baseline
        r_xray_legacy = client.post(
            "/api/v1/xray/predict",
            files={"file": ("test_chest_xray.png", img_bytes, "image/png")},
            data={"model_id": "densenet121_nih_6disease"}
        )
        assert r_xray_legacy.status_code == 200
        assert r_xray_legacy.json()["model"]["id"] == "densenet121_nih_6disease"
