"""
LungVision AI — Digitally Reconstructed Radiograph (DRR) Integration Tests
Tests: backend/tests/test_ct_drr.py
"""

import pytest
from fastapi.testclient import TestClient


def test_generate_frontal_drr(client: TestClient):
    """Verifies frontal (PA) DRR projection synthesis from reference 3D CT volume."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/drr?view=frontal")
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["scan_id"] == "demo_thoracic_scan"
    assert data["view"] == "frontal_pa"
    assert data["view_name"] == "Frontal Posteroanterior (PA)"
    assert data["image_base64"].startswith("data:image/png;base64,")

    # Dimensions check
    dim = data["dimensions"]
    assert dim["width"] == 128
    assert dim["height"] == 64

    # Attenuation check
    stats = data["attenuation_stats"]
    assert stats["min_line_integral"] >= 0.0
    assert stats["max_line_integral"] > stats["min_line_integral"]

    # Geometry check
    geo = data["geometry"]
    assert geo["sdd_mm"] == 1800.0
    assert geo["sod_mm"] == 1450.0
    assert geo["magnification_factor"] == 1.241
    assert "Y (Posterior -> Anterior)" in geo["axis_integrated"]


def test_generate_lateral_drr(client: TestClient):
    """Verifies lateral (LL) DRR projection synthesis from reference 3D CT volume."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/drr?view=lateral")
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["view"] == "lateral_ll"
    assert data["view_name"] == "Lateral Left-to-Right (LL)"
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert "X (Right -> Left)" in data["geometry"]["axis_integrated"]


def test_generate_drr_with_silhouette_overlay(client: TestClient):
    """Verifies DRR generation with projected 2D lung mask boundary overlay."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/drr?view=frontal&overlay_silhouette=true")
    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["mask_overlay_base64"] is not None
    assert data["mask_overlay_base64"].startswith("data:image/png;base64,")


def test_generate_drr_with_poisson_noise(client: TestClient):
    """Verifies DRR simulation with Poisson quantum photon noise."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/drr?view=frontal&add_noise=true")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["attenuation_stats"]["max_line_integral"] > 0.0


def test_drr_unknown_scan_returns_404(client: TestClient):
    """Verifies unknown scan_id returns 404 ResourceNotFoundError."""
    response = client.get("/api/v1/ct/non_existent_scan/drr")
    assert response.status_code == 404

