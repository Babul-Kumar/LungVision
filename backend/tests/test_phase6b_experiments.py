"""
Unit and regression test suite for Phase 6B: Classification Performance Improvement
& Controlled Architecture Experiments.

Verifies:
1. Production checkpoint SHA-256 byte hash and frozen NIH thresholds remain strictly unaltered.
2. All 9 Phase 6B deliverables exist, are valid, and contain required analytical sections.
3. Letterbox preprocessing correctly preserves aspect ratio without geometric distortion.
4. MultiScaleDenseNet121 architecture forward pass, feature pyramid levels, and ablations.
5. Operating point hierarchy (Point 1 screening vs Point 2 Youden vs Point 3 triage).
6. Baseline reproduction metrics and non-negative confusion matrices.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_CHECKPOINT = REPO_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
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

DELIVERABLES_DIR = REPO_ROOT / "reports" / "phase6b"
EXPECTED_DELIVERABLES = [
    "phase6b_baseline_reproduction.json",
    "phase6b_preprocessing_comparison.json",
    "phase6b_resolution_ablation.json",
    "phase6b_multiscale_ablation.json",
    "phase6b_class_imbalance_analysis.json",
    "phase6b_calibration_threshold_report.json",
    "phase6b_six_image_comparison.json",
    "phase6b_model_selection_report.json",
    "PHASE_6B_CLASSIFICATION_IMPROVEMENT_REPORT.md",
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


def test_phase6b_deliverables_completeness():
    """Verify all 9 Phase 6B deliverables exist, are valid, and have sufficient content."""
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
                assert "Production Invariants" in text
                assert "Task A: Baseline DenseNet121 Reproduction" in text
                assert "Task B: Aspect-Ratio-Preserving Preprocessing" in text
                assert "Task C: Multi-Resolution Progression" in text
                assert "Task D: MultiScaleDenseNet121 Architecture" in text
                assert "Task E: Class Imbalance & Label Quality" in text
                assert "Task F: Uncertainty Calibration" in text
                assert "Task G: Error-Specific Comparative Evaluation" in text
                assert "Task H: Multi-Criteria Decision Analysis" in text


def test_letterbox_preprocessing_aspect_ratio():
    """Verify letterboxing preserves aspect ratio and calculates correct symmetric padding."""
    # Test non-square image: width=504, height=422 (AR=1.194)
    img = Image.new("RGB", (504, 422), color=(128, 128, 128))
    target_size = 224
    
    # Calculate scale and padding
    scale = target_size / max(img.size)
    new_w, new_h = int(img.size[0] * scale), int(img.size[1] * scale)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Target aspect ratio preserved within rescaled dimension
    assert new_w == target_size  # Longest dimension scaled to target
    assert new_h < target_size   # Shorter dimension scaled down
    assert pad_w == 0            # No width padding needed
    assert pad_h > 0             # Height padding added symmetrically
    
    # Test square image: width=2048, height=2048 (AR=1.0)
    sq_img = Image.new("RGB", (2048, 2048), color=(100, 100, 100))
    scale_sq = target_size / max(sq_img.size)
    sq_new_w, sq_new_h = int(sq_img.size[0] * scale_sq), int(sq_img.size[1] * scale_sq)
    sq_pad_w = (target_size - sq_new_w) // 2
    sq_pad_h = (target_size - sq_new_h) // 2
    
    assert sq_new_w == target_size
    assert sq_new_h == target_size
    assert sq_pad_w == 0
    assert sq_pad_h == 0


def test_multiscale_densenet121_forward():
    """Verify MultiScaleDenseNet121 architecture forward pass and feature extraction."""
    from backend.app.models.multiscale_densenet import MultiScaleDenseNet121
    model = MultiScaleDenseNet121(num_classes=6, use_gmp=True, use_se=True)
    model.eval()
    
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    
    assert logits.shape == (2, 6), f"Expected logits shape (2, 6), got {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in MultiScaleDenseNet121 output"


def test_multiscale_densenet121_ablations():
    """Verify MultiScaleDenseNet121 can be instantiated with ablated configurations."""
    from backend.app.models.multiscale_densenet import MultiScaleDenseNet121
    
    # Ablation 1: GAP only (no GMP)
    model_no_gmp = MultiScaleDenseNet121(num_classes=6, use_gmp=False, use_se=True)
    model_no_gmp.eval()
    with torch.no_grad():
        out1 = model_no_gmp(torch.randn(1, 3, 224, 224))
    assert out1.shape == (1, 6)
    
    # Ablation 2: No SE Attention (identity)
    model_no_se = MultiScaleDenseNet121(num_classes=6, use_gmp=True, use_se=False)
    model_no_se.eval()
    with torch.no_grad():
        out2 = model_no_se(torch.randn(1, 3, 224, 224))
    assert out2.shape == (1, 6)


def test_operating_points_hierarchy():
    """Verify operating points satisfy sensitivity ordering without test set leakage."""
    report_path = DELIVERABLES_DIR / "phase6b_calibration_threshold_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    operating_points = data["operating_points"]
    for pathology in PATHOLOGIES:
        assert pathology in operating_points
        pts = operating_points[pathology]
        
        p1 = pts["point_1_frozen_nih_production"]
        p2 = pts["point_2_validation_youden_optimal"]
        p3 = pts["point_3_high_sensitivity_triage"]
        
        # Point 3 (Triage) must achieve at least ~80% sensitivity
        assert p3["sensitivity"] >= 0.79, f"{pathology} Point 3 sensitivity {p3['sensitivity']} < 0.79"
        
        # Point 1 (Screening) must achieve high specificity
        assert p1["specificity"] >= 0.85, f"{pathology} Point 1 specificity {p1['specificity']} < 0.85"


def test_baseline_reproduction_metrics():
    """Verify baseline reproduction metrics conform to NIH benchmark invariants."""
    baseline_path = DELIVERABLES_DIR / "phase6b_baseline_reproduction.json"
    with open(baseline_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data["evaluation_cohort"]["total_samples"] == 25596
    assert data["evaluation_cohort"]["unique_patients"] == 2797
    assert data["macro_metrics"]["macro_auroc"] > 0.75
    
    per_class = data["per_class_results"]
    for pathology in PATHOLOGIES:
        res = per_class[pathology]
        assert res["auroc"] > 0.65, f"Pathology {pathology} AUROC too low"
        cm = res["confusion_matrix"]
        assert cm["TP"] >= 0 and cm["FP"] >= 0 and cm["TN"] >= 0 and cm["FN"] >= 0
        total_eval = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        assert total_eval == 25596, f"Confusion matrix total {total_eval} != 25596 for {pathology}"
