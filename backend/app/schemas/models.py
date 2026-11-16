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
    active_threshold_set: str = "NIH ChestX-ray14 Locked Production Thresholds (Frozen)"
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
    external_validation_metrics: Optional[Dict[str, Any]] = None
    threshold_comparison: Optional[Dict[str, Any]] = None
    manual_tests_audit: Optional[List[Dict[str, Any]]] = None
    expanded_cohort_comparison: Optional[Dict[str, Any]] = None
    threshold_optimization_report: Optional[Dict[str, Any]] = None
    uncertain_label_sensitivity: Optional[Dict[str, Any]] = None
    calibration_analysis: Optional[Dict[str, Any]] = None
    resolution_ablation: Optional[Dict[str, Any]] = None
    phase4_experiment_manifest: Optional[Dict[str, Any]] = None
    resolution_comparison: Optional[Dict[str, Any]] = None
    loss_function_comparison: Optional[Dict[str, Any]] = None
    phase4_classwise_metrics: Optional[Dict[str, Any]] = None
    phase4_calibration_report: Optional[Dict[str, Any]] = None
    phase4_baseline_reconciliation: Optional[Dict[str, Any]] = None
    phase4_prediction_reproduction: Optional[Dict[str, Any]] = None
    phase4_ablation_results: Optional[Dict[str, Any]] = None
    phase4_statistical_audit: Optional[Dict[str, Any]] = None
    phase4_calibration_reaudit: Optional[Dict[str, Any]] = None
    phase4_inference_benchmark: Optional[Dict[str, Any]] = None
    active_threshold_set: str = "NIH ChestX-ray14 Locked Production Thresholds (Frozen)"
    notes: List[str]
