"""
LungVision AI — Phase 7 Clinical Prediction & Model Governance Endpoints.

Provides:
- POST /api/v1/predictions: End-to-end multi-label thoracic disease inference with
  clinical quality checking, calibrated thresholding, uncertainty quantification,
  and high-resolution Grad-CAM explainability.
- GET /api/v1/model/health: Operational telemetry, device placement, memory footprint,
  and checkpoint byte-level SHA-256 integrity verification.
- GET /api/v1/model/info: Scientific metadata, architecture specifications,
  operating points, validation metrics, and Brier calibration scores.
"""

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool

from backend.app.config import settings
from backend.app.core.security import validate_image_file
from backend.app.inference.calibration import OperatingPoint
from backend.app.inference.prediction_engine import prediction_engine


router = APIRouter(tags=["Phase 7 Clinical Prediction Engine"])


@router.post(
    "/predictions",
    status_code=status.HTTP_200_OK,
    summary="Phase 7 MultiScaleDenseNet121 Clinical Inference"
)
async def predict_phase7(
    file: UploadFile = File(..., description="Frontal thoracic radiograph (DICOM, PNG, JPEG, WebP)"),
    operating_point: Optional[str] = Form(
        "operating_point_A_f1_optimal",
        description="Operating point: 'operating_point_A_f1_optimal', 'operating_point_B_balanced_youden', or 'operating_point_C_high_sensitivity'"
    ),
    generate_cam: bool = Form(
        False,
        description="Whether to generate high-resolution Grad-CAM visual explainability heatmap overlay"
    ),
    cam_target_disease: Optional[str] = Form(
        None,
        description="Specific target pathology for Grad-CAM (e.g. 'Pneumothorax', 'Atelectasis', 'Cardiomegaly')"
    )
):
    """
    Executes end-to-end clinical inference pipeline:
    1. Pre-inference quality and plausibility screening.
    2. Exact Phase 7 deterministic preprocessing.
    3. MultiScaleDenseNet121 forward pass (8,071,942 parameters).
    4. Calibrated threshold application and margin assessment.
    5. Uncertainty quantification and clinical recommendation.
    6. High-resolution multi-scale Grad-CAM generation (optional).
    """
    content = await file.read()
    validate_image_file(file, content, settings.MAX_IMAGE_SIZE_BYTES)

    # Validate operating point
    op_enum = None
    if operating_point:
        try:
            op_enum = OperatingPoint(operating_point)
        except ValueError:
            valid_opts = [op.value for op in OperatingPoint]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid operating point '{operating_point}'. Choose from: {valid_opts}"
            )

    result = await run_in_threadpool(
        prediction_engine.predict,
        image_input=content,
        operating_point=op_enum,
        generate_cam=generate_cam,
        cam_target_disease=cam_target_disease
    )

    if result.get("status") == "REJECTED":
        return result

    return result


@router.get(
    "/model/health",
    status_code=status.HTTP_200_OK,
    summary="Phase 7 Prediction Engine Health & Integrity"
)
async def get_model_health():
    """Returns runtime health, device placement, parameter counts, and checkpoint SHA-256."""
    return await run_in_threadpool(prediction_engine.get_health)


@router.get(
    "/model/info",
    status_code=status.HTTP_200_OK,
    summary="Phase 7 Model Specifications & Scientific Provenance"
)
async def get_model_info():
    """Returns model architecture, validation AUROC/AUPRC, operating points, and Brier calibration scores."""
    return await run_in_threadpool(prediction_engine.get_model_info)
