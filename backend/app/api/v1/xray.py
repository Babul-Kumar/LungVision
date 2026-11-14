from fastapi import APIRouter, File, Form, Query, UploadFile, status
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
    file: UploadFile = File(..., description="Frontal chest X-ray image (PNG, JPEG, WebP)")
) -> XRayMultiLabelResponse:
    """Executes multi-label screening across 6 thoracic pathologies using DenseNet121.
    
    Pathologies evaluated:
    - Cardiomegaly
    - Edema
    - Consolidation
    - Pneumonia
    - Atelectasis
    - Pneumothorax
    
    Includes calibrated NIH validation thresholds and experimental detection flags.
    """
    content = await file.read()
    validate_image_file(file, content, settings.MAX_IMAGE_SIZE_BYTES)
    return xray_service.predict_multilabel(content)


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
    return xray_service.predict_binary(content, model_id=model_id)

