"""
LungVision AI — Statistical Shape Model (SSM) 3D Reconstruction Integration Tests
Tests: backend/tests/test_ssm_reconstruction.py
"""

import pytest
from fastapi.testclient import TestClient


def test_ssm_model_info_endpoint(client: TestClient):
    """Verifies SSM metadata endpoint returns canonical model geometry and variance."""
    response = client.get("/api/v1/reconstruction/ssm/model-info")
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["loaded"] is True
    assert data["num_vertices"] > 5000
    assert data["num_faces"] > 10000
    assert data["num_modes"] >= 15
    assert data["total_explained_variance_pct"] >= 95.0
    assert data["reference_volume_ml"] > 0.0
    
    # Verify landmarks
    landmarks = data["landmarks"]
    assert len(landmarks) == 16
    landmark_names = [lm["name"] for lm in landmarks]
    assert any("Carina" in name for name in landmark_names)
    assert "Right Lung Apex" in landmark_names
    assert "Left Lung Apex" in landmark_names
    
    # Verify modes list
    assert len(data["modes"]) >= 15
    assert data["modes"][0]["mode_index"] == 1
    assert data["modes"][0]["explained_variance_pct"] > 30.0


def test_ssm_fit_default_endpoint(client: TestClient):
    """Verifies multi-view reconstruction fit converges rapidly with valid volumetry and uncertainty."""
    payload = {
        "frontal_aspect_ratio": 1.05,
        "lateral_aspect_ratio": 0.95,
        "estimated_inspiratory_effort": 0.2,
        "silhouette_scale": 1.0,
        "n_iterations": 40,
        "include_mesh_geometry": True
    }
    response = client.post("/api/v1/reconstruction/ssm/fit", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["status"] in ("converged", "partial")
    assert len(data["optimal_coefficients"]) == 10
    
    # Verify organ volumetry
    vol = data["volumetry"]
    assert vol["total_lung_volume_ml"] > 0.0
    assert vol["left_lung_ml"] > 0.0
    assert vol["right_lung_ml"] > 0.0
    assert 0.6 <= vol["symmetry_ratio"] <= 1.5
    
    # Verify optimization metrics
    metrics = data["metrics"]
    assert metrics["estimated_dice_score"] >= 0.85
    assert metrics["estimated_assd_mm"] > 0.0
    assert metrics["mean_vertex_uncertainty_mm"] > 0.0
    assert metrics["max_vertex_uncertainty_mm"] >= metrics["mean_vertex_uncertainty_mm"]
    assert metrics["confidence_margin_pct"] == 95.0
    
    # Verify mesh descriptor and geometry
    surface_mesh = data["surface_mesh"]
    assert surface_mesh["vertex_count"] > 5000
    assert surface_mesh["face_count"] > 10000
    assert len(surface_mesh["landmarks"]) == 16
    assert len(surface_mesh["vertices"]) == surface_mesh["vertex_count"]
    assert len(surface_mesh["faces"]) == surface_mesh["face_count"]
    
    # Verify scientific disclaimer is present
    assert "macroscopic organ boundary geometry" in data["disclaimer"]


def test_ssm_fit_frontal_only(client: TestClient):
    """Verifies SSM fitting works when only frontal radiograph is available."""
    payload = {
        "frontal_aspect_ratio": 1.10,
        "lateral_aspect_ratio": None,
        "n_iterations": 30
    }
    response = client.post("/api/v1/reconstruction/ssm/fit", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["success"] is True
    assert data["volumetry"]["total_lung_volume_ml"] > 0.0
    assert data["surface_mesh"]["vertex_count"] > 5000


def test_ssm_fit_with_2d_landmarks(client: TestClient):
    """Verifies SSM fitting incorporates 2D anatomical landmark constraints."""
    payload = {
        "frontal_aspect_ratio": 1.0,
        "lateral_aspect_ratio": 1.0,
        "target_landmarks_2d": {
            "right_apex": [155.0, 42.0],
            "left_apex": [355.0, 44.0],
            "carina": [256.0, 160.0]
        },
        "n_iterations": 25
    }
    response = client.post("/api/v1/reconstruction/ssm/fit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["metrics"]["landmark_residual_mm"] is not None
    assert data["metrics"]["landmark_residual_mm"] > 0.0


def test_ssm_template_mesh_obj(client: TestClient):
    """Verifies streaming of canonical template mesh in standard Wavefront OBJ format."""
    response = client.get("/api/v1/reconstruction/ssm/template-mesh")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    
    content = response.text
    assert "# LungVision AI — Statistical Shape Model 3D Reconstruction" in content
    assert "# Vertices:" in content
    assert "o lungvision_reconstructed_lungs" in content
    assert "\nv " in content
    assert "\nf " in content


def test_ssm_export_obj_endpoint(client: TestClient):
    """Verifies exporting a patient-specific fitted mesh with deformation coefficients."""
    payload = {
        "shape_coefficients": [1.5, -0.8, 0.4, 0.0, 0.2]
    }
    response = client.post("/api/v1/reconstruction/ssm/export-obj", json=payload)
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    
    content = response.text
    assert "# LungVision AI — Statistical Shape Model 3D Reconstruction" in content
    assert "# Vertices:" in content
    assert "\nv " in content
    assert "\nf " in content


def test_ssm_deform_endpoint(client: TestClient):
    """Verifies interactive deformation endpoint for bilateral, left, and right structures."""
    # 1. Bilateral default
    payload = {
        "shape_coefficients": [500.0, -200.0, 100.0],
        "structure": "bilateral"
    }
    response = client.post("/api/v1/reconstruction/ssm/deform", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["num_vertices"] == 6737
    assert data["num_faces"] == 13478
    assert data["volumetry"]["total_lung_volume_ml"] > 0
    assert data["metrics"]["assd_mm"] <= 2.5
    assert len(data["vertices"]) == 6737
    assert len(data["faces"]) == 13478

    # 2. Left lung partition
    payload_left = {
        "shape_coefficients": [0.0],
        "structure": "left_lung"
    }
    resp_left = client.post("/api/v1/reconstruction/ssm/deform", json=payload_left)
    assert resp_left.status_code == 200
    data_left = resp_left.json()
    assert data_left["num_vertices"] == 3377

    # 3. Right lung partition
    payload_right = {
        "shape_coefficients": [0.0],
        "structure": "right_lung"
    }
    resp_right = client.post("/api/v1/reconstruction/ssm/deform", json=payload_right)
    assert resp_right.status_code == 200
    data_right = resp_right.json()
    assert data_right["num_vertices"] == 3360


