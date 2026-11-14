from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.common import StandardResponse


class DiseasePrediction(BaseModel):
    """Prediction values and calibration flags for an individual pathology."""
    probability: float = Field(..., ge=0.0, le=1.0, description="Sigmoid probability score")
    threshold: float = Field(..., description="Calibrated NIH validation threshold")
    experimental_flag: bool = Field(..., description="True if probability >= threshold")


class ModelInfo(BaseModel):
    """Metadata describing the neural network executing inference."""
    id: str
    architecture: str
    task: str
    training_dataset: str
    checkpoint_epoch: int
    input_dimensions: List[int] = Field(default=[224, 224, 3])


class PredictionSummary(BaseModel):
    """Aggregate findings from multi-label screening."""
    flagged_conditions_count: int
    highest_probability_disease: str
    highest_probability_value: float


class XRayMultiLabelResponse(StandardResponse):
    """Full response schema for 6-disease DenseNet121 inference."""
    model: ModelInfo
    predictions: Dict[str, DiseasePrediction]
    summary: PredictionSummary
    inference_time_ms: float


class BinaryPrediction(BaseModel):
    """Prediction detail for binary Normal/Abnormal screening."""
    abnormal_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_threshold: float
    classification: str = Field(..., description="'Normal' or 'Abnormal'")
    experimental_flag: bool


class XRayBinaryResponse(StandardResponse):
    """Full response schema for binary ResNet18 inference."""
    model: ModelInfo
    prediction: BinaryPrediction
    inference_time_ms: float

