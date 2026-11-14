import math
from fastapi import status


EXPECTED_6_DISEASES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]


def test_multilabel_xray_inference_png(client, sample_xray_png_bytes):
    """Verifies complete multi-label DenseNet121 inference on PNG input."""
    files = {"file": ("test_chest.png", sample_xray_png_bytes, "image/png")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    assert data["success"] is True
    assert data["model"]["architecture"] == "DenseNet121"
    assert data["model"]["id"] == "densenet121_nih_6disease"
    assert data["model"]["checkpoint_epoch"] == 4
    
    predictions = data["predictions"]
    assert len(predictions) == 6
    
    # Verify all 6 pathologies and their properties
    for disease in EXPECTED_6_DISEASES:
        assert disease in predictions
        entry = predictions[disease]
        prob = entry["probability"]
        thresh = entry["threshold"]
        flag = entry["experimental_flag"]
        
        assert 0.0 <= prob <= 1.0
        assert not math.isnan(prob)
        assert thresh > 0.0
        assert flag == (prob >= thresh)
    
    # Verify summary fields
    summary = data["summary"]
    assert "flagged_conditions_count" in summary
    assert summary["highest_probability_disease"] in EXPECTED_6_DISEASES
    assert 0.0 <= summary["highest_probability_value"] <= 1.0
    
    # Verify mandatory disclaimer
    assert "disclaimer" in data
    assert "research" in data["disclaimer"].lower()
    assert "not a medical diagnosis" in data["disclaimer"].lower()
    assert data["inference_time_ms"] > 0.0


def test_multilabel_xray_inference_jpeg(client, sample_xray_rgb_jpeg_bytes):
    """Verifies multi-label inference on RGB JPEG input."""
    files = {"file": ("test_chest.jpg", sample_xray_rgb_jpeg_bytes, "image/jpeg")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert len(data["predictions"]) == 6


def test_binary_xray_inference_original(client, sample_xray_png_bytes):
    """Verifies binary classification with original ResNet18."""
    files = {"file": ("chest.png", sample_xray_png_bytes, "image/png")}
    data_form = {"model_id": "original_resnet18"}
    response = client.post("/api/v1/xray/predict-binary", files=files, data=data_form)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    
    assert data["success"] is True
    assert data["model"]["architecture"] == "ResNet18"
    assert data["model"]["id"] == "original_resnet18"
    
    pred = data["prediction"]
    assert 0.0 <= pred["abnormal_probability"] <= 1.0
    assert pred["classification"] in ("Normal", "Abnormal")
    assert pred["experimental_flag"] == (pred["abnormal_probability"] >= pred["calibrated_threshold"])
    assert "disclaimer" in data


def test_binary_xray_inference_augmented(client, sample_xray_png_bytes):
    """Verifies binary classification with blur-augmented ResNet18."""
    files = {"file": ("chest.png", sample_xray_png_bytes, "image/png")}
    data_form = {"model_id": "blur_augmented_resnet18"}
    response = client.post("/api/v1/xray/predict-binary", files=files, data=data_form)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model"]["id"] == "blur_augmented_resnet18"
    assert data["prediction"]["classification"] in ("Normal", "Abnormal")


def test_multilabel_xray_inference_real_radiograph(client):
    """Verifies DenseNet121 multi-label inference on an authentic, full-resolution chest X-ray image."""
    from pathlib import Path
    img_path = Path("LungVision_AI_Final_Research_Bundle/external_validation_readiness/vindr_dicom_preview.png")
    assert img_path.exists(), "Authentic chest X-ray sample must exist in research bundle"
    
    with open(img_path, "rb") as fp:
        real_img_bytes = fp.read()
    
    files = {"file": ("vindr_chest_xray.png", real_img_bytes, "image/png")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["model"]["architecture"] == "DenseNet121"
    
    preds = data["predictions"]
    for disease in EXPECTED_6_DISEASES:
        assert disease in preds
        prob = preds[disease]["probability"]
        assert 0.0 <= prob <= 1.0
        assert not math.isnan(prob)
    
    # Real image has high probability for pneumonia/consolidation
    assert preds["Pneumonia"]["probability"] > 0.50


