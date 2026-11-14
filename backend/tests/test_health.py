from fastapi import status


def test_root_endpoint(client):
    """Verifies root endpoint metadata and links."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "docs" in data
    assert "health" in data
    assert "disclaimer" in data


def test_health_check(client):
    """Verifies liveness health check."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_readiness_check(client):
    """Verifies model loading readiness check."""
    response = client.get("/api/v1/ready")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ready"
    assert "device" in data
    assert data["models_loaded"]["densenet121_6disease"] is True
    assert data["models_loaded"]["resnet18_original"] is True
    assert data["models_loaded"]["resnet18_augmented"] is True

