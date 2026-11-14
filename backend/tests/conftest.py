import io
import sys
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app
from backend.app.services.model_loader import model_loader


@pytest.fixture(scope="session", autouse=True)
def initialize_models():
    """Ensures models are loaded into memory for the entire test session."""
    model_loader.load_all_models()
    yield


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_xray_png_bytes() -> bytes:
    """Generates a synthetic 224x224 grayscale chest X-ray image in PNG format."""
    # Create realistic gradient image simulating thoracic exposure
    img = Image.new("L", (224, 224), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def sample_xray_rgb_jpeg_bytes() -> bytes:
    """Generates a synthetic 3-channel 256x256 RGB JPEG image."""
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def invalid_text_file_bytes() -> bytes:
    """Returns invalid non-image text content."""
    return b"This is a plain text file, not an image or medical file."


@pytest.fixture
def corrupted_image_bytes() -> bytes:
    """Returns invalid file starting with PNG header but corrupted payload."""
    return b"\x89PNG\r\n\x1a\ncorrupted_data_that_fails_image_decode"


@pytest.fixture
def sample_ct_zip_bytes() -> bytes:
    """Generates a valid zip archive for CT testing."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("slice_001.dcm", b"DICOM_HEADER_MOCK_DATA_FOR_TESTING")
    return buf.getvalue()

