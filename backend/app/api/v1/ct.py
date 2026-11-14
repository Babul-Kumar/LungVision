from typing import Optional
from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import PlainTextResponse
from starlette.concurrency import run_in_threadpool
from backend.app.config import settings
from backend.app.core.security import validate_ct_archive
from backend.app.schemas.ct import (
    CTAnnotationsResponse,
    CTMeshQualityResponse,
    CTMeshResponse,
    CTMeasurementResponse,
    CTMetadataResponse,
    CTSliceResponse,
    CTUploadResponse,
    DRRProjectionResponse
)
from backend.app.services.ct.drr_service import drr_service
from backend.app.services.ct_service import ct_service


router = APIRouter(prefix="/ct", tags=["3D Computed Tomography (CT) Exploration"])


@router.post(
    "/upload",
    response_model=CTUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload CT Scan Volume"
)
async def upload_ct_scan(
    file: UploadFile = File(..., description="CT archive (.zip, .nii.gz, .nii, .dcm, .mhd)")
) -> CTUploadResponse:
    """Ingests a 3D CT volume or archive for structural visualization and analysis."""
    content = await file.read()
    validate_ct_archive(file, content, settings.MAX_CT_SIZE_BYTES)
    filename = file.filename or "ct_upload.zip"
    return await run_in_threadpool(ct_service.register_upload, filename, content)


@router.get(
    "/{scan_id}",
    response_model=CTMetadataResponse,
    summary="Get CT Scan Summary"
)
async def get_ct_scan_summary(scan_id: str) -> CTMetadataResponse:
    """Retrieves spatial properties and configuration for a registered CT scan."""
    return ct_service.get_metadata(scan_id)


@router.get(
    "/{scan_id}/metadata",
    response_model=CTMetadataResponse,
    summary="Get Detailed Volume Metadata"
)
async def get_ct_metadata(scan_id: str) -> CTMetadataResponse:
    """Retrieves dimensional bounds, spacing, origin, direction matrix, labels, and window presets."""
    return ct_service.get_metadata(scan_id)


@router.get(
    "/{scan_id}/slice",
    response_model=CTSliceResponse,
    summary="Extract Orthogonal 2D Slice"
)
async def get_ct_slice(
    scan_id: str,
    view: str = Query("axial", description="Orthogonal view: 'axial', 'coronal', 'sagittal'"),
    slice_index: int = Query(32, ge=0, description="Slice depth index"),
    window: str = Query("lung", description="Window preset: 'lung', 'mediastinum', 'bone', 'abdomen'"),
    window_center: Optional[float] = Query(None, description="Custom HU window center"),
    window_width: Optional[float] = Query(None, description="Custom HU window width"),
    overlay_mask: bool = Query(True, description="Whether to alpha-blend segmentation mask contour"),
    show_nodules: bool = Query(True, description="Whether to render intersecting nodule markers")
) -> CTSliceResponse:
    """Extracts a windowed cross-sectional slice image encoded as base64 PNG."""
    return await run_in_threadpool(
        ct_service.get_slice,
        scan_id=scan_id,
        view=view,
        slice_index=slice_index,
        window=window,
        window_center=window_center,
        window_width=window_width,
        overlay_mask=overlay_mask,
        show_nodules=show_nodules
    )


@router.get(
    "/{scan_id}/annotations",
    response_model=CTAnnotationsResponse,
    summary="Get LUNA16 Nodule Annotations"
)
async def get_ct_annotations(scan_id: str) -> CTAnnotationsResponse:
    """Retrieves all mapped LUNA16 nodule annotations with physical mm and discrete voxel coordinates."""
    return ct_service.get_annotations(scan_id)


@router.get(
    "/{scan_id}/mesh",
    summary="Stream 3D Surface Mesh (OBJ)"
)
async def get_ct_mesh(
    scan_id: str,
    structure: str = Query("lungs", description="Anatomical structure: 'lungs', 'left_lung', 'right_lung', 'airways'"),
    step: int = Query(1, ge=1, le=8, description="Marching cubes downsampling step (1=full, 2=lightweight)"),
    pad_boundary: bool = Query(True, description="Apply 1-voxel padding to ensure watertight closed surfaces"),
    download: bool = Query(False, description="If true, sets attachment header for file download")
):
    """Generates and streams a 3D Wavefront OBJ mesh with physical coordinates for Three.js rendering."""
    obj_data, metadata = await run_in_threadpool(
        ct_service.generate_surface_mesh,
        scan_id=scan_id,
        structure=structure,
        step=step,
        pad_boundary=pad_boundary
    )

    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{scan_id}_{structure}.obj"'

    return PlainTextResponse(content=obj_data, media_type="text/plain", headers=headers)


@router.get(
    "/{scan_id}/mesh/quality",
    response_model=CTMeshQualityResponse,
    summary="Get 3D Mesh Quality Audit Metrics"
)
async def get_mesh_quality(
    scan_id: str,
    structure: str = Query("lungs", description="Target structure to audit")
) -> CTMeshQualityResponse:
    """Evaluates mesh watertightness, Euler number, winding consistency, and mesh vs voxel volume discrepancy."""
    return await run_in_threadpool(ct_service.get_mesh_quality, scan_id=scan_id, structure=structure)


@router.get(
    "/{scan_id}/measurements",
    response_model=CTMeasurementResponse,
    summary="Quantitative Lung Volume Analysis"
)
async def get_ct_measurements(scan_id: str) -> CTMeasurementResponse:
    """Computes exact physical lung volumes in mL, symmetry ratios, and full label distribution audit."""
    return await run_in_threadpool(ct_service.get_measurements, scan_id)


@router.get(
    "/{scan_id}/segmentation",
    summary="Get Structural Segmentation Metadata"
)
async def get_ct_segmentation(scan_id: str):
    """Returns segmentation structural boundaries and available label channels."""
    metadata = ct_service.get_metadata(scan_id)
    return {
        "scan_id": scan_id,
        "segmentation_type": "anatomical_compartment",
        "available_structures": metadata.anatomical_structures,
        "labels_present": metadata.labels_present,
        "status": "exploratory_structural_only",
        "disclaimer": (
            "CT segmentation is structural for 3D exploration. "
            "Not validated for nodule malignancy or clinical staging."
        )
    }


@router.get(
    "/{scan_id}/drr",
    response_model=DRRProjectionResponse,
    summary="Synthesize Digitally Reconstructed Radiograph (DRR)",
    description="Generates a calibrated Frontal (PA) or Lateral (LL) DRR projection using Beer-Lambert ray attenuation physics."
)
async def get_ct_drr(
    scan_id: str,
    view: str = Query("frontal", description="Projection view: 'frontal' (PA) or 'lateral' (LL)"),
    sdd_mm: float = Query(1800.0, ge=500.0, le=3000.0, description="Source-to-Detector Distance in mm"),
    sod_mm: float = Query(1450.0, ge=400.0, le=2500.0, description="Source-to-Object Distance in mm"),
    overlay_silhouette: bool = Query(False, description="Whether to overlay projected 2D lung boundary silhouette"),
    add_noise: bool = Query(False, description="Simulate Poisson quantum photon noise")
) -> DRRProjectionResponse:
    """Synthesizes a 2D radiograph directly from 3D CT volume attenuation line integrals."""
    volume = ct_service.get_volume(scan_id)
    return await run_in_threadpool(
        drr_service.generate_drr,
        volume=volume,
        view=view,
        sdd_mm=sdd_mm,
        sod_mm=sod_mm,
        overlay_silhouette=overlay_silhouette,
        add_noise=add_noise
    )
