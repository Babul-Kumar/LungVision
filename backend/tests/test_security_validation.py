from fastapi import status


def test_reject_non_image_text_file(client, invalid_text_file_bytes):
    """Verifies rejection of plain text files disguised or uploaded as images."""
    files = {"file": ("document.txt", invalid_text_file_bytes, "text/plain")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "FILE_VALIDATION_ERROR"
    assert "Unsupported image extension" in data["error"]["message"]


def test_reject_unsupported_extension(client, sample_xray_png_bytes):
    """Verifies rejection of unsupported image extension (.gif or .bmp)."""
    files = {"file": ("scan.gif", sample_xray_png_bytes, "image/gif")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["error"]["code"] == "FILE_VALIDATION_ERROR"


def test_reject_corrupted_image(client, corrupted_image_bytes):
    """Verifies handling of unparseable or corrupted image bytes."""
    files = {"file": ("corrupted.png", corrupted_image_bytes, "image/png")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["error"]["code"] == "FILE_VALIDATION_ERROR"


def test_reject_empty_file(client):
    """Verifies rejection of 0-byte file payload."""
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/v1/xray/predict", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["error"]["code"] == "FILE_VALIDATION_ERROR"


def test_reject_invalid_binary_model_id(client, sample_xray_png_bytes):
    """Verifies 404 response when querying an unknown binary model."""
    files = {"file": ("chest.png", sample_xray_png_bytes, "image/png")}
    data_form = {"model_id": "non_existent_model_id"}
    response = client.post("/api/v1/xray/predict-binary", files=files, data=data_form)
    
    assert response.status_code == status.HTTP_404_NOT_FOUND
    data = response.json()
    assert data["error"]["code"] == "RESOURCE_NOT_FOUND"


def test_reject_fake_ct_zip_archive(client):
    """Verifies rejection of a text file disguised with a .zip extension (magic byte mismatch)."""
    fake_zip_bytes = b"This is just plain text content with more than 32 characters, not a real zip file."
    files = {"file": ("fake_archive.zip", fake_zip_bytes, "application/zip")}
    response = client.post("/api/v1/ct/upload", files=files)
    
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    data = response.json()
    assert data["error"]["code"] == "FILE_VALIDATION_ERROR"
    assert "magic bytes" in data["error"]["message"].lower()


