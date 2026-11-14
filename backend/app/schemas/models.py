from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.schemas.common import StandardResponse


class ModelRegistryEntry(BaseModel):
    """Configuration and audit descriptor for a registered neural network."""
    id: str
    name: str
    architecture: str
    task: str
    checkpoint_path: str
    sha256: Optional[str] = None
    input_size: List[int] = [224, 224, 3]
    labels: List[str]
    status: str
    metrics: Dict[str, Any] = {}
    calibration: Dict[str, Any] = {}


class ModelListResponse(StandardResponse):
    """Collection of available models."""
    total_models: int = 0
    models: List[ModelRegistryEntry]


class ResearchMetricItem(BaseModel):
    disease_or_model: str
    dataset: str
    auroc: Optional[float] = None
    auprc: Optional[float] = None
    f1: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    precision: Optional[float] = None
    threshold: Optional[float] = None


class ResearchMetricsResponse(StandardResponse):
    """Consolidated benchmark evaluation metrics."""
    nih_test_metrics: List[Dict[str, Any]]
    chexpert_external_metrics: List[Dict[str, Any]]
    vindr_external_metrics: List[Dict[str, Any]]
    notes: List[str]

