"""
LungVision AI — 3D Reconstruction API Router
Router: backend/app/api/v1/reconstruction.py
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, HTTPException, Query, status
from fastapi.responses import PlainTextResponse

from backend.app.schemas.reconstruction import (
    ExportOBJRequest,
    SSMModelInfoResponse,
    SSMReconstructionRequest,
    SSMReconstructionResponse
)
from backend.app.services.ct.ssm_service import ssm_service


router = APIRouter(prefix="/reconstruction", tags=["3D Anatomical Reconstruction"])


@router.get(
    "/ssm/model-info",
    response_model=SSMModelInfoResponse,
    summary="Retrieve Statistical Shape Model Metadata",
    description="Returns architectural specifications, explained variance, principal modes, and landmarks."
)
async def get_ssm_model_info() -> SSMModelInfoResponse:
    """Returns metadata for the calibrated Statistical Shape Model."""
    try:
        info = ssm_service.get_model_info()
        return SSMModelInfoResponse(
            success=True,
            **info,
            disclaimer="SSM Point Distribution Model calibrated from thoracic CT surface extractions for research use only."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve SSM model info: {str(e)}"
        )


@router.post(
    "/ssm/fit",
    response_model=SSMReconstructionResponse,
    summary="Fit 3D Lung Surface from Radiographic Projections",
    description="Optimizes SSM shape coefficients and 7-DoF pose against dual-view projection constraints."
)
async def fit_ssm_from_projections(
    request: SSMReconstructionRequest = Body(...)
) -> SSMReconstructionResponse:
    """Performs non-linear multi-view fitting to synthesize patient-specific 3D lung surfaces."""
    try:
        result = ssm_service.fit_from_projections(
            frontal_aspect_ratio=request.frontal_aspect_ratio,
            lateral_aspect_ratio=request.lateral_aspect_ratio,
            estimated_inspiratory_effort=request.estimated_inspiratory_effort,
            silhouette_scale=request.silhouette_scale,
            target_landmarks_2d=request.target_landmarks_2d,
            n_iterations=request.n_iterations,
            include_mesh_geometry=request.include_mesh_geometry,
        )
        return SSMReconstructionResponse(
            success=result["success"],
            status=result["status"],
            optimization_latency_ms=result["optimization_latency_ms"],
            iterations_computed=result["iterations_computed"],
            optimal_coefficients=result["optimal_coefficients"],
            metrics=result["metrics"],
            volumetry=result["volumetry"],
            surface_mesh=result["surface_mesh"],
            disclaimer=result["disclaimer"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D SSM Reconstruction failed: {str(e)}"
        )


@router.get(
    "/ssm/template-mesh",
    response_class=PlainTextResponse,
    summary="Stream Canonical Template Lung Mesh (OBJ)",
    description="Returns the reference 3D lung boundary mesh in Wavefront OBJ format."
)
async def get_template_mesh():
    """Streams the mean shape template mesh as Wavefront OBJ."""
    try:
        verts = ssm_service.synthesize_shape()
        obj_text = ssm_service.export_reconstructed_obj(verts)
        return PlainTextResponse(
            content=obj_text,
            media_type="text/plain",
            headers={"Content-Disposition": "inline; filename=canonical_lung_mesh.obj"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate template mesh: {str(e)}"
        )


@router.post(
    "/ssm/export-obj",
    response_class=PlainTextResponse,
    summary="Export Patient-Specific Reconstructed OBJ",
    description="Generates and streams a Wavefront OBJ mesh corresponding to the specified shape coefficients."
)
async def export_reconstructed_obj(
    request: ExportOBJRequest = Body(default_factory=ExportOBJRequest)
):
    """Generates a patient-specific OBJ file from deformation coefficients."""
    try:
        coeffs = request.shape_coefficients if request and request.shape_coefficients else []
        verts = ssm_service.synthesize_shape(coeffs)
        obj_text = ssm_service.export_reconstructed_obj(verts)
        return PlainTextResponse(
            content=obj_text,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=lungvision_reconstructed_mesh.obj"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export reconstructed OBJ: {str(e)}"
        )
