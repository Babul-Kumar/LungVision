import numpy as np
from fastapi import status
from backend.app.services.ct.geometry import (
    continuous_index_to_physical,
    is_inside_volume,
    physical_to_continuous_index,
    validate_geometry,
    voxel_zyx_to_physical
)
from backend.app.services.ct.loader import create_reference_thoracic_volume


# -------------------------------------------------------------
# 1. Coordinate Mathematics & Geometry Validation Unit Tests
# -------------------------------------------------------------

def test_coordinate_transformation_roundtrip():
    """Verifies exact round-trip conversion between physical mm and continuous voxel indices."""
    origin = np.array([-193.1, -166.7, -307.25])
    spacing = np.array([0.742188, 0.742188, 1.25])
    direction = np.eye(3)

    test_point = (-145.2, -132.4, -270.5)

    cont_idx = physical_to_continuous_index(test_point, origin, spacing, direction)
    recovered_pt = continuous_index_to_physical(cont_idx, origin, spacing, direction)

    assert np.allclose(test_point, recovered_pt, atol=1e-5)


def test_voxel_zyx_to_physical_conversion():
    """Verifies that marching cubes ZYX vertex coordinates map accurately to physical millimeters."""
    origin = np.array([-100.0, -100.0, -50.0])
    spacing = np.array([1.0, 1.0, 2.0])
    direction = np.eye(3)

    # 2 vertices in (Z, Y, X) order
    vertices_zyx = np.array([
        [0.0, 10.0, 20.0],
        [25.0, 50.0, 60.0]
    ])

    # Vertex 0: z=0, y=10, x=20 -> X = -100 + 20*1 = -80, Y = -100 + 10*1 = -90, Z = -50 + 0*2 = -50
    physical = voxel_zyx_to_physical(vertices_zyx, origin, spacing, direction, step=1, pad_offset=0)

    assert np.allclose(physical[0], [-80.0, -90.0, -50.0])
    assert np.allclose(physical[1], [-40.0, -50.0, 0.0])


def test_geometry_validation():
    """Verifies sub-millimeter geometry comparison between CT and segmentation."""
    shape = (64, 128, 128)
    spacing = (0.742188, 0.742188, 1.25)
    origin = (-193.1, -166.7, -307.25)
    direction = tuple(np.eye(3).flatten())

    # Matching geometry
    is_valid, mismatches = validate_geometry(
        ct_shape=shape, mask_shape=shape,
        ct_spacing=spacing, mask_spacing=spacing,
        ct_origin=origin, mask_origin=origin,
        ct_direction=direction, mask_direction=direction
    )
    assert is_valid is True
    assert len(mismatches) == 0

    # Mismatching spacing
    is_valid_bad, mismatches_bad = validate_geometry(
        ct_shape=shape, mask_shape=shape,
        ct_spacing=spacing, mask_spacing=(1.0, 1.0, 1.0),
        ct_origin=origin, mask_origin=origin,
        ct_direction=direction, mask_direction=direction
    )
    assert is_valid_bad is False
    assert len(mismatches_bad) > 0


# -------------------------------------------------------------
# 2. REST API Integration Tests
# -------------------------------------------------------------

def test_ct_upload_and_metadata(client, sample_ct_zip_bytes):
    """Verifies CT archive ingestion and metadata extraction."""
    files = {"file": ("chest_ct.zip", sample_ct_zip_bytes, "application/zip")}
    response = client.post("/api/v1/ct/upload", files=files)
    
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["success"] is True
    scan_id = data["scan_id"]
    assert scan_id.startswith("ct_")
    
    # Query metadata
    meta_resp = client.get(f"/api/v1/ct/{scan_id}/metadata")
    assert meta_resp.status_code == status.HTTP_200_OK
    meta = meta_resp.json()
    assert meta["scan_id"] == scan_id
    assert "dimensions" in meta
    assert "window_presets" in meta
    assert "lung" in meta["window_presets"]
    assert meta["dimensions"]["z"] == 64
    assert meta["annotation_count"] == 12


def test_ct_slice_extraction_with_overlays(client):
    """Verifies orthogonal slice extraction with HU windowing and segmentation overlays."""
    # Test Axial slice
    axial_resp = client.get("/api/v1/ct/demo_thoracic_scan/slice?view=axial&slice_index=32&window=lung&overlay_mask=true&show_nodules=true")
    assert axial_resp.status_code == status.HTTP_200_OK
    data = axial_resp.json()
    assert data["view"] == "axial"
    assert data["slice_index"] == 32
    assert data["image_base64"].startswith("data:image/png;base64,")
    assert "nodules_on_slice" in data

    # Test Coronal slice
    coronal_resp = client.get("/api/v1/ct/demo_thoracic_scan/slice?view=coronal&slice_index=64&window=mediastinum")
    assert coronal_resp.status_code == status.HTTP_200_OK
    assert coronal_resp.json()["view"] == "coronal"

    # Test Sagittal slice
    sagittal_resp = client.get("/api/v1/ct/demo_thoracic_scan/slice?view=sagittal&slice_index=64&window=bone")
    assert sagittal_resp.status_code == status.HTTP_200_OK
    assert sagittal_resp.json()["view"] == "sagittal"


def test_ct_nodule_annotations_endpoint(client):
    """Verifies retrieval of mapped LUNA16 nodule annotations."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/annotations")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["success"] is True
    assert data["total_annotations"] == 12
    assert data["inside_volume_count"] == 12
    
    first_nodule = data["annotations"][0]
    assert "coordX_mm" in first_nodule
    assert "voxel_x" in first_nodule
    assert "diameter_mm" in first_nodule
    assert first_nodule["inside_volume"] is True


def test_ct_mesh_stream(client):
    """Verifies 3D Wavefront OBJ mesh generation."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/mesh?structure=lungs&step=2&pad_boundary=true")
    assert response.status_code == status.HTTP_200_OK
    text = response.text
    assert "# LungVision AI 3D Mesh Export" in text
    assert "v " in text  # Contains vertices
    assert "f " in text  # Contains faces


def test_ct_mesh_quality_audit(client):
    """Verifies Trimesh quality evaluation endpoint."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/mesh/quality?structure=left_lung")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["structure"] == "left_lung"
    assert "watertight" in data
    assert "euler_number" in data
    assert "surface_area_mm2" in data
    assert data["voxel_volume_ml"] > 0.0


def test_ct_quantitative_measurements(client):
    """Verifies full-resolution anatomical volume calculations."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/measurements")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["has_segmentation"] is True
    assert data["geometry_validation_status"] == "PASSED"
    assert "lungs" in data
    assert data["lungs"]["left_lung_ml"] > 0.0
    assert data["lungs"]["right_lung_ml"] > 0.0
    assert data["lungs"]["total_lung_volume_ml"] > 0.0
    assert len(data["label_audit"]) >= 3


def test_ct_segmentation_descriptor(client):
    """Verifies structural segmentation descriptor endpoint."""
    response = client.get("/api/v1/ct/demo_thoracic_scan/segmentation")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "exploratory_structural_only"
    assert "disclaimer" in data


def test_ct_not_found(client):
    """Verifies 404 for unknown CT scan ID."""
    response = client.get("/api/v1/ct/non_existent_scan_id/metadata")
    assert response.status_code == status.HTTP_404_NOT_FOUND
