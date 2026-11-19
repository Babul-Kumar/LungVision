"""
LungVision AI — Phase 7 Thresholds & Calibration Profiles.

Stores empirical validation-derived operating points for MultiScaleDenseNet121
(msd121_224_weighted_bce_direct.pth, Val Macro AUROC: 0.819726).

Source artifacts:
- phase7 result/lungvision_phase7_final_evaluation/msd121_224_weighted_bce_direct_thresholds.csv
- phase7 result/lungvision_phase7_final_evaluation/msd121_224_weighted_bce_direct_brier_scores.csv
- phase7 result/lungvision_phase7_final_evaluation/msd121_224_weighted_bce_direct_per_disease_metrics.csv
"""

from enum import Enum
from typing import Dict, List, Optional


DISEASE_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]


class OperatingPoint(str, Enum):
    F1_OPTIMAL = "operating_point_A_f1_optimal"
    BALANCED_YOUDEN = "operating_point_B_balanced_youden"
    HIGH_SENSITIVITY = "operating_point_C_high_sensitivity"


# Primary Operating Point: Validation F1-Optimal (Phase 7 Final Evaluation)
PHASE7_F1_OPTIMAL_THRESHOLDS: Dict[str, float] = {
    "Cardiomegaly": 0.87,
    "Edema": 0.94,
    "Consolidation": 0.79,
    "Pneumonia": 0.55,
    "Atelectasis": 0.67,
    "Pneumothorax": 0.77
}

# Operating Point B: Balanced Standard (Default Unbiased Operating Point)
PHASE7_BALANCED_THRESHOLDS: Dict[str, float] = {
    "Cardiomegaly": 0.50,
    "Edema": 0.50,
    "Consolidation": 0.50,
    "Pneumonia": 0.50,
    "Atelectasis": 0.50,
    "Pneumothorax": 0.50
}

# Operating Point C: High-Sensitivity Screening Point
PHASE7_HIGH_SENSITIVITY_THRESHOLDS: Dict[str, float] = {
    "Cardiomegaly": 0.35,
    "Edema": 0.30,
    "Consolidation": 0.40,
    "Pneumonia": 0.30,
    "Atelectasis": 0.45,
    "Pneumothorax": 0.40
}

# Empirically verified Brier calibration scores on 13,501 validation radiographs
PHASE7_BRIER_SCORES: Dict[str, float] = {
    "Cardiomegaly": 0.078781,
    "Edema": 0.103290,
    "Consolidation": 0.225532,
    "Pneumonia": 0.106051,
    "Atelectasis": 0.188885,
    "Pneumothorax": 0.117270
}

# Per-disease validation metrics (AUROC / AUPRC)
PHASE7_VALIDATION_METRICS: Dict[str, Dict[str, float]] = {
    "Cardiomegaly": {"auroc": 0.8857, "auprc": 0.2320, "sensitivity": 0.6778, "specificity": 0.8965},
    "Edema": {"auroc": 0.8758, "auprc": 0.1446, "sensitivity": 0.7368, "specificity": 0.8554},
    "Consolidation": {"auroc": 0.7945, "auprc": 0.1091, "sensitivity": 0.8153, "specificity": 0.6289},
    "Pneumonia": {"auroc": 0.7591, "auprc": 0.0300, "sensitivity": 0.5252, "specificity": 0.8496},
    "Atelectasis": {"auroc": 0.7936, "auprc": 0.3047, "sensitivity": 0.7539, "specificity": 0.6845},
    "Pneumothorax": {"auroc": 0.8095, "auprc": 0.1529, "sensitivity": 0.5846, "specificity": 0.8452}
}


class CalibrationManager:
    """Provides validated threshold sets and calibration metadata for Phase 7."""

    def __init__(self, default_point: OperatingPoint = OperatingPoint.F1_OPTIMAL):
        self.active_point = default_point
        self._operating_points = {
            OperatingPoint.F1_OPTIMAL: PHASE7_F1_OPTIMAL_THRESHOLDS,
            OperatingPoint.BALANCED_YOUDEN: PHASE7_BALANCED_THRESHOLDS,
            OperatingPoint.HIGH_SENSITIVITY: PHASE7_HIGH_SENSITIVITY_THRESHOLDS
        }

    def get_thresholds(self, point: Optional[OperatingPoint] = None) -> Dict[str, float]:
        """Returns the dictionary of disease-specific thresholds for the requested operating point."""
        key = point or self.active_point
        return self._operating_points.get(key, PHASE7_F1_OPTIMAL_THRESHOLDS).copy()

    def get_threshold(self, disease: str, point: Optional[OperatingPoint] = None) -> float:
        """Returns the threshold for a single disease."""
        thresholds = self.get_thresholds(point)
        return thresholds.get(disease, 0.50)

    def set_active_operating_point(self, point: OperatingPoint):
        """Sets the default operating point for predictions."""
        if point not in self._operating_points:
            raise ValueError(f"Unknown operating point: {point}")
        self.active_point = point

    def get_brier_scores(self) -> Dict[str, float]:
        return PHASE7_BRIER_SCORES.copy()

    def get_validation_metrics(self) -> Dict[str, Dict[str, float]]:
        return PHASE7_VALIDATION_METRICS.copy()


calibration_manager = CalibrationManager()
