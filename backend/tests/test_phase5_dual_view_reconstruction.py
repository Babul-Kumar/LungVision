"""
LungVision AI — Phase 5 Automated Verification Suite
Test Suite: backend/tests/test_phase5_dual_view_reconstruction.py
Version: 5.0.0

Verifies all Phase 5 gates and invariants:
1. Patient-level split integrity (301 train / 64 val / 65 test, zero leakage).
2. Mesh correspondence, topology, and 16 anatomical landmarks.
3. PCA fitting invariants (M=20 modes, variance explained, eigenvector orthogonality).
4. Paired dual-view input pipeline & mismatch rejection.
5. Neural shape predictors forward pass across all 4 architectures.
6. 3D patient-specific surface reconstruction & divergence theorem volumetry.
7. Held-out CT validation benchmarks & statistical significance.
8. API isolation and production classification preservation.
"""

import hashlib
import json
from pathlib import Path
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.config import BASE_DIR, settings
from backend.app.main import app
from backend.app.models.dual_view_shape_predictor import (
    FrontalOnlyShapePredictor,
    LateralOnlyShapePredictor,
    DualViewConcatShapePredictor,
    DualViewCrossAttentionShapePredictor,
    MahalanobisShapeLoss,
    create_shape_predictor,
)
from backend.app.services.ct.patient_reconstruction_service import (
    patient_reconstruction_service,
)
from backend.app.services.xray_service import xray_service

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SSM_DIR = PROJECT_ROOT / "research" / "3d_reconstruction" / "ssm"
OUTPUT_DIR = PROJECT_ROOT / "research" / "3d_reconstruction"
PCA_V2_PATH = OUTPUT_DIR / "ssm_lung_pca_v2.json"
VALIDATION_REPORT_PATH = OUTPUT_DIR / "held_out_reconstruction_validation.json"

client = TestClient(app)


def test_phase5_patient_split_integrity():
    """Gate A: Verify 430 unique patients and zero leakage across train/val/test splits."""
    split_manifest_path = SSM_DIR / "split_manifest.json"
    assert split_manifest_path.exists(), "split_manifest.json must exist"

    with open(split_manifest_path, "r", encoding="utf-8") as f:
        split_data = json.load(f)

    train_pids = split_data["train_patients"]
    val_pids = split_data["validation_patients"]
    test_pids = split_data["test_patients"]

    assert len(train_pids) == 301, f"Expected 301 train patients, got {len(train_pids)}"
    assert len(val_pids) == 64, f"Expected 64 val patients, got {len(val_pids)}"
    assert len(test_pids) == 65, f"Expected 65 test patients, got {len(test_pids)}"

    train_set, val_set, test_set = set(train_pids), set(val_pids), set(test_pids)
    assert len(train_set.intersection(val_set)) == 0, "Train-Val patient leakage detected!"
    assert len(train_set.intersection(test_set)) == 0, "Train-Test patient leakage detected!"
    assert len(val_set.intersection(test_set)) == 0, "Val-Test patient leakage detected!"

    # Verify verified_multi_patient_manifest.json
    multi_manifest_path = OUTPUT_DIR / "verified_multi_patient_manifest.json"
    assert multi_manifest_path.exists(), "verified_multi_patient_manifest.json must exist"
    with open(multi_manifest_path, "r", encoding="utf-8") as f:
        multi_data = json.load(f)
    assert len(multi_data["patients"]) == 430, "Manifest must index all 430 patients"


def test_phase5_mesh_correspondence_and_landmarks():
    """Gate B: Verify canonical template topology, bilateral partitioning, and 16 landmarks."""
    assert PCA_V2_PATH.exists(), "ssm_lung_pca_v2.json must exist"
    with open(PCA_V2_PATH, "r", encoding="utf-8") as f:
        pca = json.load(f)

    num_verts = pca["num_vertices"]
    num_faces = pca["num_faces"]
    assert num_verts == 6737, f"Expected 6737 vertices, got {num_verts}"
    assert num_faces == 13478, f"Expected 13478 faces, got {num_faces}"

    mean_shape = np.asarray(pca["mean_shape_flattened"], dtype=np.float32).reshape(num_verts, 3)
    left_verts = np.where(mean_shape[:, 0] < 0)[0]
    right_verts = np.where(mean_shape[:, 0] >= 0)[0]
    assert len(left_verts) == 3377, f"Left lung vertex partition invariant failed: {len(left_verts)}"
    assert len(right_verts) == 3360, f"Right lung vertex partition invariant failed: {len(right_verts)}"

    # Landmark mapping
    landmarks = pca["landmark_indices"]
    landmark_names = pca["landmark_names"]
    assert len(landmarks) == 16, "Expected exactly 16 standardized anatomical landmarks"
    assert len(landmark_names) == 16, "Expected 16 landmark names"
    for lm_idx in landmarks:
        assert 0 <= lm_idx < num_verts, f"Landmark index {lm_idx} out of vertex bounds"

    # Correspondence audit file
    audit_path = OUTPUT_DIR / "mesh_correspondence_audit.json"
    assert audit_path.exists(), "mesh_correspondence_audit.json must exist"
    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["topology_validation"]["folded_triangles_detected"] == 0
    assert audit["topology_validation"]["inverted_normals_detected"] == 0
    assert audit["gpa_residual_metrics"]["mean_residual_gpa_distance_mm"] < 10.0


def test_phase5_pca_basis_orthogonality():
    """Gate C: Verify PCA basis orthogonality and variance retention."""
    with open(PCA_V2_PATH, "r", encoding="utf-8") as f:
        pca = json.load(f)

    num_modes = pca["num_modes"]
    assert num_modes == 20, f"Expected 20 PCA modes, got {num_modes}"
    assert pca["total_explained_variance_pct"] >= 95.0, "Total explained variance must be >= 95%"

    P = np.asarray(pca["eigenvectors"], dtype=np.float32)  # (3N, M)
    assert P.shape == (6737 * 3, 20), f"Unexpected eigenvector matrix shape: {P.shape}"

    # Verify column orthogonality: P^T * P approx I_20
    PtP = P.T @ P
    eye_20 = np.eye(20, dtype=np.float32)
    max_ortho_err = np.max(np.abs(PtP - eye_20))
    assert max_ortho_err < 1e-3, f"Eigenvector orthogonality violated: error = {max_ortho_err}"

    # Eigenvalues must be positive and non-increasing
    evals = np.asarray(pca["eigenvalues"], dtype=np.float32)
    assert np.all(evals > 0), "All eigenvalues must be positive"
    assert np.all(np.diff(evals) <= 0), "Eigenvalues must be sorted in descending order"


def test_phase5_paired_view_input_pipeline():
    """Gate D: Verify paired input preprocessing, normalizations, and fallback handling."""
    dummy_img = np.ones((100, 100), dtype=np.float32) * 0.7
    tensor = patient_reconstruction_service.preprocess_image_payload(dummy_img)

    assert tensor.shape == (1, 1, 224, 224), f"Preprocessed tensor shape error: {tensor.shape}"
    assert float(tensor.min()) >= 0.0, "Tensor values must be >= 0"
    assert float(tensor.max()) <= 1.0, "Tensor values must be <= 1"

    # Rejection of invalid inputs when single view not allowed
    with pytest.raises(Exception):
        patient_reconstruction_service.predict_coefficients(
            frontal_tensor=tensor, lateral_tensor=None, model_type="dual_view_attention"
        )


def test_phase5_neural_shape_predictors_forward():
    """Gate E: Test forward passes for all 4 model architectures."""
    b_size = 2
    f_dummy = torch.randn(b_size, 1, 224, 224)
    l_dummy = torch.randn(b_size, 1, 224, 224)

    models = {
        "frontal_only": FrontalOnlyShapePredictor(num_modes=20),
        "lateral_only": LateralOnlyShapePredictor(num_modes=20),
        "dual_view_concat": DualViewConcatShapePredictor(num_modes=20),
        "dual_view_attention": DualViewCrossAttentionShapePredictor(num_modes=20),
    }

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            if name == "frontal_only":
                out = model(f_dummy)
            elif name == "lateral_only":
                out = model(None, l_dummy)
            else:
                out = model(f_dummy, l_dummy)

        assert out.shape == (b_size, 20), f"Model {name} output shape mismatch: {out.shape}"
        assert not torch.isnan(out).any(), f"Model {name} produced NaN outputs"

    # Test Mahalanobis loss
    evals = torch.ones(20) * 100.0
    crit = MahalanobisShapeLoss(eigenvalues=evals, alpha=0.01)
    loss = crit(out, torch.zeros_like(out))
    assert loss.item() > 0, "Loss must be positive"


def test_phase5_3d_mesh_synthesis_and_volumetry():
    """Gate F: Verify continuous mesh deformation and divergence theorem volumetry."""
    coeffs = np.zeros(20, dtype=np.float32)
    verts, faces, landmarks = patient_reconstruction_service.synthesize_3d_mesh(coeffs)

    assert verts.shape == (6737, 3), f"Synthesized vertices shape mismatch: {verts.shape}"
    assert faces.shape == (13478, 3), f"Synthesized faces shape mismatch: {faces.shape}"
    assert len(landmarks) == 16, "16 Anatomical landmarks must be extracted"

    # Volumetry
    vol = patient_reconstruction_service.compute_volumetry(verts)
    assert 2500.0 <= vol.total_lung_volume_ml <= 5500.0, f"Unrealistic total lung volume: {vol.total_lung_volume_ml} mL"
    assert 1200.0 <= vol.left_lung_ml <= 2600.0, f"Unrealistic left lung volume: {vol.left_lung_ml} mL"
    assert 1300.0 <= vol.right_lung_ml <= 3000.0, f"Unrealistic right lung volume: {vol.right_lung_ml} mL"
    assert 0.9 <= vol.symmetry_ratio <= 1.4, f"Unrealistic symmetry ratio: {vol.symmetry_ratio}"

    # Partition submeshes
    left_mesh, right_mesh = patient_reconstruction_service.extract_lung_submeshes(verts)
    assert left_mesh.vertex_count == 3377
    assert right_mesh.vertex_count == 3360


def test_phase5_held_out_validation_report_metrics():
    """Gate G: Verify held-out validation report results and statistical significance."""
    assert VALIDATION_REPORT_PATH.exists(), "held_out_reconstruction_validation.json must exist"

    with open(VALIDATION_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    assert report["cohort_specification"]["num_test_patients"] == 65
    assert report["cohort_specification"]["leakage_audit"].startswith("PASSED")

    summary = report["cohort_summary_by_model"]
    assert "dual_view_attention" in summary
    assert "frontal_only" in summary

    attn_assd = summary["dual_view_attention"]["assd_mm"]["mean"]
    front_assd = summary["frontal_only"]["assd_mm"]["mean"]
    assert attn_assd < front_assd, f"Dual-view attention ASSD ({attn_assd}) must be lower than Frontal ({front_assd})"

    # Statistical significance
    ablation = report["ablation_comparison"]["frontal_vs_dual_view"]
    assert ablation["statistical_significance"]["statistically_significant"] is True
    assert ablation["statistical_significance"]["wilcoxon_signed_rank_pvalue"] < 0.05


def test_phase5_api_endpoint_isolation_and_production_preservation():
    """Gate H: Verify experimental endpoint operates and production model is 100% preserved."""
    # 1. Verify Production DenseNet121 Checkpoint SHA-256
    prod_path = PROJECT_ROOT / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth"
    assert prod_path.exists(), "Production checkpoint must exist"
    with open(prod_path, "rb") as f:
        prod_sha256 = hashlib.sha256(f.read()).hexdigest()
    assert prod_sha256 == "8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af", (
        "CRITICAL: Production checkpoint SHA-256 was altered!"
    )

    # 2. Verify all 6 NIH locked production thresholds
    frozen_thresh = xray_service.nih_thresholds
    assert round(frozen_thresh["Cardiomegaly"], 4) == 0.9502
    assert round(frozen_thresh["Edema"], 4) == 0.8770
    assert round(frozen_thresh["Consolidation"], 4) == 0.7329
    assert round(frozen_thresh["Pneumonia"], 4) == 0.7529
    assert round(frozen_thresh["Atelectasis"], 4) == 0.7495
    assert round(frozen_thresh["Pneumothorax"], 4) == 0.5825

    # 3. Test Experimental Dual-View Endpoint
    payload = {
        "patient_id": "LIDC-IDRI-0002",
        "frontal_image_base64": None,
        "lateral_image_base64": None,
        "fusion_model": "dual_view_attention",
        "allow_single_view": False,
        "include_mesh_geometry": True,
        "compare_with_reference_ct": True,
    }

    # Pass 1 synthetic image
    import base64
    from PIL import Image
    import io

    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()

    payload["frontal_image_base64"] = b64_str
    payload["lateral_image_base64"] = b64_str

    res = client.post("/api/v1/reconstruction/3d/patient-specific", json=payload)
    assert res.status_code == 200, f"API failed with {res.status_code}: {res.text}"

    data = res.json()
    assert data["success"] is True
    assert data["status"] == "CONVERGED"
    assert data["pairing_status"] == "VALIDATED_DUAL_VIEW"
    assert len(data["predicted_coefficients"]) == 20
    assert "total_lung_volume_ml" in data["volumetry"]
    assert data["comparison_metrics"] is not None
    assert "Not a Diagnostic CT Scan" in data["disclaimer"]
