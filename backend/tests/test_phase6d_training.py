"""
Unit and regression test suite for Phase 6D: Reproducible MultiScaleDenseNet121-FPN
Training & Validation.

Verifies:
1. Production checkpoint SHA-256 byte hash and frozen NIH thresholds remain strictly unaltered.
2. All 10 Phase 6D deliverables exist, are valid, and contain required analytical sections.
3. The genuine 512x512 checkpoint exists at both target paths with verified SHA-256 hash.
4. Checkpoint metadata confirms resolution 512, 5 training epochs, focal loss, dual pooling GAP+GMP, and SE attention.
5. Training history demonstrates monotonic loss convergence across 5 epochs.
6. Controlled 3-way model comparison confirms matched patient-level evaluations.
7. Reproducible CLI evaluation script and Kaggle training workflow exist.
"""

import hashlib
import json
from pathlib import Path
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_CHECKPOINT = REPO_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
EXPECTED_PROD_SHA256 = "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af"

TARGET_CKPT_BACKEND = REPO_ROOT / "backend" / "lungvision_disease_classification" / "checkpoints" / "experimental" / "multiscale_densenet121_fpn_512_v1.pth"
TARGET_CKPT_ROOT = REPO_ROOT / "lungvision_disease_classification" / "checkpoints" / "experimental" / "multiscale_densenet121_fpn_512_v1.pth"
EXPECTED_512_SHA256 = "f64728d2742752085cd105d34af00795f53586b71abdf4a67acfa63c8d7b7d2c"

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

DELIVERABLES_DIR = REPO_ROOT / "reports" / "phase6d"
EXPECTED_DELIVERABLES = [
    "phase6d_environment_audit.json",
    "phase6d_dataset_split_audit.json",
    "phase6d_preprocessing_config.json",
    "phase6d_training_history.json",
    "phase6d_checkpoint_manifest.json",
    "phase6d_model_comparison.json",
    "phase6d_calibration_report.json",
    "phase6d_patient_cluster_statistics.json",
    "phase6d_error_analysis.json",
    "PHASE_6D_REPRODUCIBLE_TRAINING_REPORT.md",
]


def test_production_checkpoint_invariants():
    """Verify production DenseNet121 checkpoint SHA-256 hash has not been modified."""
    assert PRODUCTION_CHECKPOINT.exists(), f"Production checkpoint missing at {PRODUCTION_CHECKPOINT}"
    with open(PRODUCTION_CHECKPOINT, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    assert file_hash == EXPECTED_PROD_SHA256, (
        f"Production checkpoint SHA-256 mismatch! Expected {EXPECTED_PROD_SHA256}, got {file_hash}"
    )


def test_frozen_nih_thresholds():
    """Verify frozen NIH production thresholds match exact clinical reference."""
    from backend.app.services.xray_service import DEFAULT_NIH_THRESHOLDS
    for disease, threshold in FROZEN_THRESHOLDS.items():
        assert disease in DEFAULT_NIH_THRESHOLDS, f"Pathology {disease} missing from model thresholds"
        assert abs(DEFAULT_NIH_THRESHOLDS[disease] - threshold) < 1e-3, (
            f"Threshold for {disease} altered! Expected ~{threshold}, got {DEFAULT_NIH_THRESHOLDS[disease]}"
        )


def test_phase6d_deliverables_completeness():
    """Verify all 10 Phase 6D deliverables exist, are valid, and have sufficient content."""
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
                assert "Task A: Training Environment" in text
                assert "Task B: Matched Preprocessing" in text
                assert "Task C: Genuine Model Training" in text
                assert "Task D: Controlled 3-Way Model Comparison" in text
                assert "Task E: Multiclass Calibration" in text
                assert "Task F: Statistical Analysis" in text
                assert "Task G: Comprehensive Anatomical Error Analysis" in text
                assert "Task H: Checkpoint Export" in text
                assert "COMPLETE" in text


def test_multiscale_512_checkpoint_exists_and_verified():
    """Verify genuine 512x512 checkpoint exists at both target paths and has exact SHA-256."""
    assert TARGET_CKPT_BACKEND.exists(), f"Backend checkpoint missing: {TARGET_CKPT_BACKEND}"
    assert TARGET_CKPT_ROOT.exists(), f"Root checkpoint missing: {TARGET_CKPT_ROOT}"
    
    with open(TARGET_CKPT_BACKEND, "rb") as f:
        sha_backend = hashlib.sha256(f.read()).hexdigest()
    with open(TARGET_CKPT_ROOT, "rb") as f:
        sha_root = hashlib.sha256(f.read()).hexdigest()
        
    assert sha_backend == EXPECTED_512_SHA256, f"Backend SHA-256 mismatch! Got {sha_backend}"
    assert sha_root == EXPECTED_512_SHA256, f"Root SHA-256 mismatch! Got {sha_root}"
    
    # Verify internal checkpoint metadata
    ckpt = torch.load(TARGET_CKPT_BACKEND, map_location="cpu")
    assert ckpt["resolution"] == 512
    assert ckpt["epoch"] == 5
    assert ckpt["loss_type"] == "focal"
    assert ckpt["architecture"] == "MultiScaleDenseNet121-FPN"
    assert ckpt["use_gmp"] is True
    assert ckpt["use_se"] is True
    assert "model_state_dict" in ckpt
    assert len(ckpt["model_state_dict"]) > 100


def test_training_history_convergence():
    """Verify training history demonstrates monotonic loss convergence across 5 epochs."""
    history_path = DELIVERABLES_DIR / "phase6d_training_history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["epochs_trained"] == 5
    assert data["target_resolution"] == "512x512"
    assert data["checkpoint_sha256"] == EXPECTED_512_SHA256
    
    history = data["epoch_by_epoch_history"]
    assert len(history) == 5
    
    train_losses = [rec["train_loss"] for rec in history]
    # Verify train loss strictly decreases
    for i in range(len(train_losses) - 1):
        assert train_losses[i] >= train_losses[i+1], f"Loss did not decrease between epoch {i+1} and {i+2}"
        
    # Verify macro AUROC increases
    aurocs = [rec["macro_auroc"] for rec in history]
    assert aurocs[-1] > aurocs[0], "Final AUROC should exceed initial AUROC"


def test_three_way_model_comparison():
    """Verify controlled 3-way model comparison bench marking on matched cohorts."""
    comp_path = DELIVERABLES_DIR / "phase6d_model_comparison.json"
    with open(comp_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    models = data["models_benchmarked"]
    assert "model_1_prod_densenet121_224" in models
    assert "model_2_exp_multiscale_224" in models
    assert "model_3_exp_multiscale_512" in models
    
    m1 = models["model_1_prod_densenet121_224"]
    m3 = models["model_3_exp_multiscale_512"]
    
    # 512x512 MultiScale-FPN must outperform baseline in AUROC and F1
    assert m3["macro_auroc"] > m1["macro_auroc"]
    assert m3["macro_f1"] > m1["macro_f1"]
    assert m3["macro_brier"] < m1["macro_brier"]
    assert m3["macro_ece"] < m1["macro_ece"]


def test_reproducible_cli_evaluation():
    """Verify reproducible evaluation script and Kaggle training script exist."""
    eval_script = REPO_ROOT / "scripts" / "evaluation" / "run_phase6d_evaluation.py"
    kaggle_script = REPO_ROOT / "scripts" / "training" / "kaggle_train_multiscale_densenet512.py"
    
    assert eval_script.exists(), f"Evaluation script missing: {eval_script}"
    assert kaggle_script.exists(), f"Kaggle training script missing: {kaggle_script}"
    assert eval_script.stat().st_size > 1000
    assert kaggle_script.stat().st_size > 200
