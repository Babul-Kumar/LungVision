import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.config import BASE_DIR, settings
from backend.app.core.errors import ResourceNotFoundError
from backend.app.core.logging import logger
from backend.app.schemas.models import (
    ModelListResponse,
    ModelRegistryEntry,
    ResearchMetricsResponse
)
from backend.app.services.model_loader import model_loader


class RegistryService:
    """Manages model registry configuration and historical benchmark research metrics."""

    def __init__(self):
        self.cached_models = self._initialize_model_catalog()

    def _initialize_model_catalog(self) -> Dict[str, ModelRegistryEntry]:
        """Loads model registry definitions from filesystem."""
        catalog: Dict[str, ModelRegistryEntry] = {}

        # 1. DenseNet121 NIH 6-Disease
        catalog["densenet121_nih_6disease"] = ModelRegistryEntry(
            id="densenet121_nih_6disease",
            name="DenseNet121 NIH 6-Disease Multi-Label Classifier",
            architecture="DenseNet121",
            task="Multi-label chest X-ray disease classification",
            checkpoint_path=str(settings.RESOLVED_DENSENET121_PATH),
            sha256="8d3690cd0438587a94afb8fd2ad84c0b30f9c1cc2727ee0eb03662a02ea020af",
            input_size=[224, 224, 3],
            labels=[
                "Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"
            ],
            status="production_frozen",
            metrics={
                "training_dataset": "NIH ChestX-ray14 (112,120 images)",
                "checkpoint_epoch": 4,
                "macro_val_auc": 0.820582,
                "macro_val_auprc": 0.166984,
                "parameters": 6960006
            },
            calibration={
                "Cardiomegaly": 0.950195,
                "Edema": 0.876953,
                "Consolidation": 0.732910,
                "Pneumonia": 0.752930,
                "Atelectasis": 0.749512,
                "Pneumothorax": 0.582520,
                "default_threshold": 0.5000
            }
        )

        # 2. ResNet18 Original Baseline
        catalog["original_resnet18"] = ModelRegistryEntry(
            id="original_resnet18",
            name="Original ResNet18 Baseline Classifier",
            architecture="ResNet18",
            task="Binary Normal vs Abnormal classification",
            checkpoint_path=str(settings.RESOLVED_RESNET18_ORIGINAL_PATH),
            sha256="a293bd9e441b2713e62b7de208611bf7cc97676ea1291ce2818ff3a497528b2f",
            input_size=[224, 224, 3],
            labels=["Normal", "Abnormal"],
            status="production_frozen",
            metrics={
                "training_dataset": "IN-CXR (12,600 images)",
                "checkpoint_epoch": 4,
                "validation_auc": 0.933206,
                "parameters": 11177025
            },
            calibration={
                "selected_threshold": 0.176463,
                "target_sensitivity": 0.90,
                "achieved_sensitivity": 0.90,
                "achieved_specificity": 0.7952,
                "Abnormal": 0.176463,
                "Normal": 0.823537,
                "default_threshold": 0.5000,
                "calibration_strategy": "Validation Set 90% Sensitivity Point"
            }
        )

        # 3. ResNet18 Blur-Augmented Robustness Model
        catalog["blur_augmented_resnet18"] = ModelRegistryEntry(
            id="blur_augmented_resnet18",
            name="Blur-Augmented ResNet18 Robustness Classifier",
            architecture="ResNet18",
            task="Binary Normal vs Abnormal classification",
            checkpoint_path=str(settings.RESOLVED_RESNET18_AUGMENTED_PATH),
            sha256="7d8634454d1ac0dfa2c2dbc8d0b08155512fb52b65c9db5a9171a5b59c7d3f24",
            input_size=[224, 224, 3],
            labels=["Normal", "Abnormal"],
            status="production_frozen",
            metrics={
                "training_dataset": "IN-CXR (Gaussian blur + rotation augmented)",
                "checkpoint_epoch": 1,
                "validation_auc": 0.938037,
                "parameters": 11177025
            },
            calibration={
                "selected_threshold": 0.187923,
                "target_sensitivity": 0.90,
                "achieved_sensitivity": 0.90,
                "achieved_specificity": 0.8222,
                "Abnormal": 0.187923,
                "Normal": 0.812077,
                "default_threshold": 0.5000,
                "calibration_strategy": "Validation Set 90% Sensitivity Point"
            }
        )

        # 4. Statistical Shape Model (SSM) Lung PCA v1
        catalog["ssm_lung_pca_v1"] = ModelRegistryEntry(
            id="ssm_lung_pca_v1",
            name="LungVision Statistical Shape Model (SSM) Lung PCA v1",
            architecture="Point Distribution Model (PDM) / Orthogonal PCA",
            task="3D macroscopic pulmonary boundary reconstruction & morphological parameterization",
            checkpoint_path=str(BASE_DIR / "data" / "3d_reconstruction" / "ssm_lung_pca_v1.json"),
            sha256="e92cf88a3dae8cd893cda93efe33630ccb1f8259dd9e6c7aa9cd9ce8463e65aa",
            input_size=[16, 3],
            labels=["Left Lung", "Right Lung", "Bilateral Lungs"],
            status="production_frozen",
            metrics={
                "training_dataset": "LUNA16 / LIDC-IDRI (430 verified scans, 301 train / 64 val / 65 test)",
                "modes": 20,
                "cumulative_explained_variance_pct": 100.0,
                "held_out_val_assd_mm": 0.072,
                "held_out_val_hd95_mm": 0.168,
                "vertices": 6737,
                "faces": 13478
            },
            calibration={
                "mahalanobis_threshold_sigma": 3.0,
                "confidence_margin_pct": 95.0,
                "calibration_strategy": "Bounded Mahalanobis Distance [-3.0σ, +3.0σ]"
            }
        )

        return catalog

    def get_models(self) -> ModelListResponse:
        """Returns all registered models."""
        models = list(self.cached_models.values())
        return ModelListResponse(total_models=len(models), models=models)

    def get_model_by_id(self, model_id: str) -> ModelRegistryEntry:
        """Retrieves a single model metadata entry."""
        if model_id not in self.cached_models:
            raise ResourceNotFoundError("Model", model_id)
        return self.cached_models[model_id]

    def get_research_metrics(self) -> ResearchMetricsResponse:
        """Aggregates verified historical evaluation metrics directly from saved CSVs."""
        nih_metrics = self._read_csv_dicts(settings.NIH_TEST_METRICS_PATH)
        chexpert_metrics = self._read_csv_dicts(settings.CHEXPERT_METRICS_PATH)
        vindr_metrics = self._read_csv_dicts(settings.VINDR_METRICS_PATH)

        notes = [
            "NIH metrics reflect calibrated test set performance across 6 multi-label diseases.",
            "CheXpert metrics demonstrate cross-hospital domain generalization when transferring locked NIH thresholds.",
            "VinDr metrics represent 15,000-image external evaluation of binary ResNet18 checkpoints.",
            "All metrics represent retrospective benchmark evaluation and do NOT establish clinical diagnostic validity."
        ]

        return ResearchMetricsResponse(
            nih_test_metrics=nih_metrics,
            chexpert_external_metrics=chexpert_metrics,
            vindr_external_metrics=vindr_metrics,
            notes=notes
        )

    @staticmethod
    def _read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
        """Parses a CSV file into a list of row dictionaries."""
        if not path.exists():
            logger.warning(f"Metrics CSV not found at {path}")
            return []
        
        try:
            results = []
            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Clean numeric values
                    clean_row = {}
                    for k, v in row.items():
                        if k is None:
                            continue
                        try:
                            clean_row[k] = float(v)
                        except (ValueError, TypeError):
                            clean_row[k] = v
                    results.append(clean_row)
            return results
        except Exception as e:
            logger.error(f"Error parsing CSV {path}: {e}")
            return []


registry_service = RegistryService()

