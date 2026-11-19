from fastapi import APIRouter, File, Form, Query, UploadFile, status
from starlette.concurrency import run_in_threadpool
from backend.app.config import settings
from backend.app.core.security import validate_image_file
from backend.app.schemas.xray import XRayBinaryResponse, XRayMultiLabelResponse
from backend.app.services.xray_service import xray_service


router = APIRouter(prefix="/xray", tags=["2D Chest X-Ray Inference"])


@router.post(
    "/predict",
    response_model=XRayMultiLabelResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-Label 6-Disease Classification"
)
async def predict_xray_multilabel(
    file: UploadFile = File(..., description="Frontal chest X-ray image (PNG, JPEG, WebP)"),
    model_id: str = Form(
        "densenet121_nih_6disease",
        description="Target model: 'densenet121_nih_6disease' (frozen baseline) or 'lungvision_phase7_msd121_224_direct'"
    )
) -> XRayMultiLabelResponse:
    """Executes multi-label screening across 6 thoracic pathologies using DenseNet121 or Phase 7 MultiScaleDenseNet121.
    
    Pathologies evaluated:
    - Cardiomegaly
    - Edema
    - Consolidation
    - Pneumonia
    - Atelectasis
    - Pneumothorax
    
    Includes calibrated thresholds and experimental detection flags.
    """
    content = await file.read()
    validate_image_file(file, content, settings.MAX_IMAGE_SIZE_BYTES)
    return await run_in_threadpool(xray_service.predict_multilabel, content, model_id=model_id)


@router.post(
    "/predict-binary",
    response_model=XRayBinaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Binary Normal vs Abnormal Classification"
)
async def predict_xray_binary(
    file: UploadFile = File(..., description="Frontal chest X-ray image"),
    model_id: str = Form(
        "original_resnet18",
        description="Target model: 'original_resnet18' or 'blur_augmented_resnet18'"
    )
) -> XRayBinaryResponse:
    """Executes binary Normal/Abnormal classification using the frozen ResNet18 checkpoints."""
    content = await file.read()
    validate_image_file(file, content, settings.MAX_IMAGE_SIZE_BYTES)
    return await run_in_threadpool(xray_service.predict_binary, content, model_id=model_id)

