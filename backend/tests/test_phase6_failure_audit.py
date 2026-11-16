"""
backend/tests/test_phase6_failure_audit.py

Phase 6 Automated Regression Test Suite:
- Checkpoint integrity and SHA-256 verification
- Exact class order preservation across backend and frontend
- Frozen NIH threshold invariance
- Preprocessing contract and tensor statistics
- Probability and threshold mathematical consistency
- Six test image prediction reproducibility
- Grad-CAM layer attribution validation
- Phase 6 machine-readable audit deliverables verification
"""

import hashlib
import json
import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torchvision import models
from fastapi import status

from backend.app.config import settings
from backend.app.services.model_loader import model_loader
from backend.app.services.preprocessing import preprocessor, IMAGENET_MEAN, IMAGENET_STD
from backend.app.services.xray_service import (
    XRayService,
    LABELS_6_DISEASE,
    DEFAULT_NIH_THRESHOLDS
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = REPO_ROOT / "reports" / "phase6"
TEST_DIR = REPO_ROOT / "test"

AUTHORITATIVE_SHA256 = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"

FROZEN_THRESHOLDS_EXACT = {
    "Cardiomegaly": 0.9501953125,
    "Edema": 0.876953125,
    "Consolidation": 0.73291015625,
    "Pneumonia": 0.7529296875,
    "Atelectasis": 0.74951171875,
    "Pneumothorax": 0.58251953125,
}

EXPECTED_6_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]


def test_phase6_checkpoint_integrity():
    """Task A: Checkpoint file hash and architecture integrity."""
    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    assert ckpt_path.exists(), f"DenseNet121 checkpoint missing at {ckpt_path}"
    
    file_bytes = ckpt_path.read_bytes()
    computed_hash = hashlib.sha256(file_bytes).hexdigest()
    assert computed_hash == AUTHORITATIVE_SHA256, (
        f"Checkpoint SHA-256 hash mismatch! Computed: {computed_hash}, Authoritative: {AUTHORITATIVE_SHA256}"
    )

    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in raw_ckpt or isinstance(raw_ckpt, dict)
    state_dict = raw_ckpt.get("model_state_dict", raw_ckpt)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(1024, 6)
    load_res = model.load_state_dict(state_dict, strict=True)
    assert len(load_res.missing_keys) == 0, f"Missing keys in state_dict: {load_res.missing_keys}"
    assert len(load_res.unexpected_keys) == 0, f"Unexpected keys in state_dict: {load_res.unexpected_keys}"

    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == 6960006, f"Expected 6,960,006 parameters, got {total_params}"
    assert raw_ckpt.get("epoch") == 4
    assert raw_ckpt.get("image_size") == 224


def test_phase6_class_order_alignment():
    """Task C: Strict class ordering alignment across backend and schemas."""
    assert LABELS_6_DISEASE == EXPECTED_6_LABELS, (
        f"LABELS_6_DISEASE order violated! Expected {EXPECTED_6_LABELS}, got {LABELS_6_DISEASE}"
    )

    ckpt_path = settings.RESOLVED_DENSENET121_PATH
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_labels = raw_ckpt.get("labels", [])
    assert ckpt_labels == EXPECTED_6_LABELS, f"Checkpoint labels violated: {ckpt_labels}"


def test_phase6_frozen_thresholds_exactness():
    """Task D: Production thresholds must match exact verified values."""
    svc = XRayService()
    for disease, expected_val in FROZEN_THRESHOLDS_EXACT.items():
        loaded_val = svc.nih_thresholds[disease]
        assert math.isclose(loaded_val, expected_val, rel_tol=1e-6), (
            f"Frozen threshold altered for {disease}! Expected {expected_val}, got {loaded_val}"
        )


def test_phase6_preprocessing_contract():
    """Task B: Preprocessor transforms raw images to exact 4D tensor contract."""
    test_file = TEST_DIR / "Cardiomegaly.jpg"
    assert test_file.exists(), f"Missing test file: {test_file}"

    raw_bytes = test_file.read_bytes()
    tensor = preprocessor.preprocess_bytes(raw_bytes)

    assert tensor.shape == (1, 3, 224, 224), f"Wrong tensor shape: {tensor.shape}"
    assert tensor.dtype == torch.float32

    # Check normalization: ImageNet mean/std
    t_np = tensor.squeeze(0).numpy()
    for c in range(3):
        # Normalized values typically fall within [-3.0, 3.0]
        assert -3.0 <= t_np[c].mean() <= 3.0
        assert 0.1 <= t_np[c].std() <= 3.0


def test_phase6_probability_threshold_math(client):
    """Task D & J: Probability calculation and threshold comparison via API."""
    test_file = TEST_DIR / "Edema.jpg"
    assert test_file.exists()

    with open(test_file, "rb") as f:
        response = client.post("/api/v1/xray/predict", files={"file": ("Edema.jpg", f.read(), "image/jpeg")})

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    predictions = data["predictions"]

    for disease in EXPECTED_6_LABELS:
        pred = predictions[disease]
        prob = pred["probability"]
        thresh = pred["threshold"]
        flag = pred["experimental_flag"]

        assert 0.0 <= prob <= 1.0
        assert 0.0 < thresh <= 1.0
        # Check rule: flag must match prob >= thresh
        assert flag == (prob >= thresh)


def test_phase6_six_test_images_reproducibility():
    """Task E: Reproduces exact inference behavior on all 6 test images."""
    svc = XRayService()
    expected_flags = {
        "Atelectasis.jpg": [],
        "Cardiomegaly.jpg": [],
        "Edema.jpg": ["Cardiomegaly"],
        "Lung-Consolidation-x-ray-image.webp": ["Consolidation"],
        "pneumonia.png": ["Consolidation"],
        "Pneumothorax.jpeg": []
    }

    for fname, exp_f in expected_flags.items():
        fpath = TEST_DIR / fname
        assert fpath.exists(), f"Missing test image: {fpath}"
        data = fpath.read_bytes()
        res = svc.predict_multilabel(data)
        actual_flags = [d for d, p in res.predictions.items() if p.experimental_flag]
        assert actual_flags == exp_f, f"Mismatch on {fname}: expected {exp_f}, got {actual_flags}"


def test_phase6_gradcam_attribution():
    """Task G: Grad-CAM generates valid spatial heatmaps on denseblock4."""
    model_loader.load_all_models()
    model = model_loader.get_model("densenet121_nih_6disease")
    model.eval()

    target_layer = model.features.denseblock4.denselayer16.conv2

    activations = {}
    gradients = {}

    def fwd_hook(mod, inp, out):
        activations["conv"] = out
        out.register_hook(lambda grad: gradients.__setitem__("conv", grad))

    handle = target_layer.register_forward_hook(fwd_hook)

    fpath = TEST_DIR / "Cardiomegaly.jpg"
    tensor = preprocessor.preprocess_bytes(fpath.read_bytes()).to(model_loader.device)

    model.zero_grad()
    logits = model(tensor)
    # Target Cardiomegaly (index 0)
    logits[0, 0].backward()

    act = activations["conv"]
    grad = gradients["conv"]
    weights = torch.mean(grad, dim=[2, 3], keepdim=True)
    cam = torch.relu(torch.sum(weights * act, dim=1, keepdim=True))

    handle.remove()

    assert cam.shape == (1, 1, 7, 7)
    assert cam.min() >= 0.0
    assert cam.max() > 0.0


def test_phase6_deliverables_completeness():
    """Verifies all required machine-readable JSON reports and overlays exist."""
    required_json = [
        "phase6_image_inventory.json",
        "phase6_checkpoint_integrity.json",
        "phase6_preprocessing_audit.json",
        "phase6_class_order_audit.json",
        "phase6_probability_threshold_audit.json",
        "phase6_six_image_predictions.json",
        "phase6_image_error_analysis.json",
        "phase6_gradcam_analysis.json",
        "phase6_training_data_audit.json",
        "phase6_threshold_performance.json",
        "phase6_frontend_backend_trace.json"
    ]
    for r in required_json:
        p = REPORT_DIR / r
        assert p.exists(), f"Required Phase 6 report missing: {p}"
        assert p.stat().st_size > 100, f"Report file empty or corrupt: {p}"

    required_overlays = [
        "gradcam_atelectasis.png",
        "gradcam_cardiomegaly.png",
        "gradcam_edema.png",
        "gradcam_lung-consolidation-x-ray-image.png",
        "gradcam_pneumonia.png",
        "gradcam_pneumothorax.png"
    ]
    for o in required_overlays:
        p = REPORT_DIR / "gradcam" / o
        assert p.exists(), f"Required Grad-CAM overlay missing: {p}"
        assert p.stat().st_size > 50000, f"Overlay file truncated: {p}"
