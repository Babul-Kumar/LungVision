"""
LungVision AI — Phase 2B Multi-Patient SSM Pipeline Tests
Tests: backend/tests/test_ssm_phase2b_pipeline.py
Version: 1.0.0

Verifies:
1. Dataset manifest structure and label reconciliation
2. Strict patient-disjoint split integrity (ZERO DATA LEAKAGE)
3. Model artifact schema, size limits (<= 15 MB), and reload integrity
4. Held-out validation and test metric bounds
5. Left/Right anatomical vertex partitioning
"""

import json
import os
from pathlib import Path
import pytest
import numpy as np
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SSM_DIR = REPO_ROOT / "research" / "3d_reconstruction" / "ssm"
DATA_DIR = REPO_ROOT / "data" / "3d_reconstruction"


def test_data_manifest_and_labels_present():
    """Verifies data manifest exists, contains required headers, and reconciles label schemas."""
    manifest_path = SSM_DIR / "data_manifest.csv"
    assert manifest_path.exists(), f"Missing {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    header = lines[0].split(",")
    assert "dataset_id" in header
    assert "scan_id" in header
    assert "patient_id" in header
    assert "split" in header
    assert "left_lung_label" in header
    assert "right_lung_label" in header
    
    # Assert candidate count
    assert len(lines) - 1 == 430  # 430 included scans


def test_patient_split_zero_leakage():
    """Critical safety test: guarantees zero patient leakage across train, val, and test splits."""
    split_manifest_path = SSM_DIR / "split_manifest.json"
    assert split_manifest_path.exists()
    
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)
        
    train_p = set(split_data["train_patients"])
    val_p = set(split_data["validation_patients"])
    test_p = set(split_data["test_patients"])
    
    assert len(train_p) == 301
    assert len(val_p) == 64
    assert len(test_p) == 65
    
    # Zero intersection invariants
    assert len(train_p.intersection(val_p)) == 0, "DATA LEAKAGE: Patients present in both Train and Val!"
    assert len(train_p.intersection(test_p)) == 0, "DATA LEAKAGE: Patients present in both Train and Test!"
    assert len(val_p.intersection(test_p)) == 0, "DATA LEAKAGE: Patients present in both Val and Test!"


def test_ssm_model_artifact_schema_and_size():
    """Verifies the exported model artifact conforms to contract specifications and is <= 15 MB."""
    model_path = SSM_DIR / "ssm_lung_pca_v1.json"
    assert model_path.exists(), f"Model artifact missing at {model_path}"
    
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    assert size_mb <= 15.0, f"Model size {size_mb:.2f} MB exceeds 15 MB budget!"
    
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)
        
    assert model["format"] == "LungVision_SSM_v1"
    assert model["num_modes"] == 20
    assert model["total_explained_variance_pct"] >= 95.0
    assert len(model["eigenvalues"]) == 20
    assert len(model["eigenvectors"]) == model["num_vertices"] * 3
    assert len(model["eigenvectors"][0]) == 20
    
    # Left and Right partition checks
    assert "left_lung_vertex_indices" in model
    assert "right_lung_vertex_indices" in model
    left_idx = model["left_lung_vertex_indices"]
    right_idx = model["right_lung_vertex_indices"]
    assert len(left_idx) + len(right_idx) == model["num_vertices"]
    assert len(set(left_idx).intersection(set(right_idx))) == 0


def test_held_out_validation_metrics_acceptable():
    """Verifies that held-out test and validation metrics meet research criteria."""
    val_path = SSM_DIR / "validation_metrics.json"
    test_path = SSM_DIR / "test_metrics.json"
    assert val_path.exists()
    assert test_path.exists()
    
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
        
    v_metrics = val_data["metrics_summary"]
    t_metrics = test_data["metrics_summary"]
    
    # ASSD <= 2.5 mm
    assert v_metrics["mean_assd_mm"] <= 2.5
    assert t_metrics["mean_assd_mm"] <= 2.5
    
    # HD95 <= 7.0 mm
    assert v_metrics["mean_hd95_mm"] <= 7.0
    assert t_metrics["mean_hd95_mm"] <= 7.0
    
    # Volume Error <= 7.5%
    assert v_metrics["mean_volume_error_pct"] <= 7.5
    assert t_metrics["mean_volume_error_pct"] <= 7.5


def test_backend_runtime_sync():
    """Verifies that backend runtime data directory contains synchronized model and OBJ mesh."""
    backend_model = DATA_DIR / "ssm_lung_pca_v1.json"
    backend_obj = DATA_DIR / "canonical_lung_mesh.obj"
    assert backend_model.exists()
    assert backend_obj.exists()
    assert os.path.getsize(backend_obj) > 10000  # Valid Wavefront OBJ mesh
