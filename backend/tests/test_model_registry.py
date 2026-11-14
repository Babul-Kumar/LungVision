from fastapi import status


def test_list_models(client):
    """Verifies listing of all registered neural networks."""
    response = client.get("/api/v1/models")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    models = data["models"]
    assert len(models) >= 3
    
    ids = [m["id"] for m in models]
    assert "densenet121_nih_6disease" in ids
    assert "original_resnet18" in ids
    assert "blur_augmented_resnet18" in ids


def test_get_model_by_id(client):
    """Verifies single model metadata query."""
    response = client.get("/api/v1/models/densenet121_nih_6disease")
    assert response.status_code == status.HTTP_200_OK
    model = response.json()
    assert model["id"] == "densenet121_nih_6disease"
    assert model["architecture"] == "DenseNet121"
    assert len(model["labels"]) == 6
    assert "Cardiomegaly" in model["labels"]


def test_get_unknown_model_returns_404(client):
    """Verifies 404 for unknown model ID."""
    response = client.get("/api/v1/models/unknown_network_xyz")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert response.json()["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_research_metrics(client):
    """Verifies retrieval of historical benchmark metrics."""
    response = client.get("/api/v1/research/metrics")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert "nih_test_metrics" in data
    assert "chexpert_external_metrics" in data
    assert "vindr_external_metrics" in data
    assert len(data["notes"]) > 0
    assert "disclaimer" in data

