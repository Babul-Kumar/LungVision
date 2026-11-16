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
            active_threshold_set="NIH ChestX-ray14 Locked Production Thresholds (Frozen)",
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
                "default_threshold": 0.5000,
                "active_threshold_set": "Operating Point A: NIH ChestX-ray14 Locked Production Thresholds (Frozen)",
                "alternative_operating_points": {
                    "operating_point_A_locked_nih": {
                        "Cardiomegaly": 0.9502,
                        "Edema": 0.8770,
                        "Consolidation": 0.7329,
                        "Pneumonia": 0.7529,
                        "Atelectasis": 0.7495,
                        "Pneumothorax": 0.5825
                    },
                    "operating_point_B_external_youden": {
                        "Cardiomegaly": 0.0298,
                        "Edema": 0.1486,
                        "Consolidation": 0.6336,
                        "Pneumonia": 0.0298,
                        "Atelectasis": 0.3070,
                        "Pneumothorax": 0.1189
                    },
                    "operating_point_C_high_sensitivity": {
                        "Cardiomegaly": 0.0100,
                        "Edema": 0.0100,
                        "Consolidation": 0.0100,
                        "Pneumonia": 0.0100,
                        "Atelectasis": 0.0100,
                        "Pneumothorax": 0.0100
                    }
                }
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
            active_threshold_set="Validation Set 90% Sensitivity Point (Frozen)",
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
            active_threshold_set="Validation Set 90% Sensitivity Point (Frozen)",
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
            active_threshold_set="Mahalanobis ±3.0σ (99.7% Gaussian Mass)",
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

        # 5. MultiScaleDenseNet121 High-Resolution Research Experimental Model
        catalog["densenet121_multiscale_highres"] = ModelRegistryEntry(
            id="densenet121_multiscale_highres",
            name="MultiScaleDenseNet121-FPN High-Resolution Experimental Classifier",
            architecture="MultiScaleDenseNet121-FPN",
            task="Multi-label high-resolution chest radiograph screening (Research Experimental)",
            checkpoint_path=str(settings.RESOLVED_DENSENET121_PATH.parent / "experimental" / "multiscale_densenet121_experimental.pth"),
            sha256="c5884ee4ff2dee85c7b46af6cda496ad0664d92c6e0f468e22a8b12addc30ec4",
            input_size=[512, 512, 3],
            labels=[
                "Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"
            ],
            status="research_experimental",
            active_threshold_set="External Calibration (Operating Point B - Youden J)",
            metrics={
                "input_resolution": "512x512",
                "macro_test_auc": 0.7845,
                "pneumothorax_auc": 0.7690,
                "atelectasis_auc": 0.7850,
                "loss_function": "Focal Loss (gamma=2.0, alpha=0.25)",
                "parameters": 9482118,
                "status": "research_only_not_in_clinical_route"
            },
            calibration={
                "Cardiomegaly": 0.05,
                "Edema": 0.15,
                "Consolidation": 0.60,
                "Pneumonia": 0.05,
                "Atelectasis": 0.30,
                "Pneumothorax": 0.12,
                "default_threshold": 0.50,
                "calibration_strategy": "Platt Temperature Scaling (T=1.42)"
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
        """Aggregates verified historical evaluation metrics directly from saved CSVs and JSON artifacts."""
        nih_metrics = self._read_csv_dicts(settings.NIH_TEST_METRICS_PATH)
        chexpert_metrics = self._read_csv_dicts(settings.CHEXPERT_METRICS_PATH)
        vindr_metrics = self._read_csv_dicts(settings.VINDR_METRICS_PATH)

        # Load modern research evaluation artifacts if generated
        eval_dir = BASE_DIR / "research" / "evaluation"
        ext_metrics = self._load_json(eval_dir / "classwise_metrics.json")
        thresh_comp = self._load_json(eval_dir / "threshold_comparison.json")
        manual_audit_data = self._load_json(eval_dir / "manual_test_reproduction.json")
        manual_audit = manual_audit_data.get("manual_tests_audit") if manual_audit_data else None

        # Phase 3 Deliverables
        cohort_comparison = self._load_json(eval_dir / "external_cohort_comparison.json")
        thresh_opt_report = self._load_json(eval_dir / "threshold_optimization_report.json")
        uncertainty_analysis = self._load_json(eval_dir / "uncertain_label_sensitivity_analysis.json")
        calibration_analysis = self._load_json(eval_dir / "calibration_analysis.json")
        resolution_ablation = self._load_json(eval_dir / "resolution_ablation_results.json")

        # Phase 4 Deliverables
        phase4_manifest = self._load_json(eval_dir / "phase4_experiment_manifest.json")
        resolution_comp = self._load_json(eval_dir / "resolution_comparison.json")
        loss_comp = self._load_json(eval_dir / "loss_function_comparison.json")
        phase4_classwise = self._load_json(eval_dir / "phase4_classwise_metrics.json")
        phase4_calibration = self._load_json(eval_dir / "phase4_calibration_report.json")

        # Phase 4B Audit Deliverables
        phase4_reconciliation = self._load_json(eval_dir / "phase4_baseline_reconciliation.json")
        phase4_reproduction = self._load_json(eval_dir / "phase4_prediction_reproduction.json")
        phase4_ablation = self._load_json(eval_dir / "phase4_ablation_results.json")
        phase4_statistical = self._load_json(eval_dir / "phase4_statistical_audit.json")
        phase4_calib_reaudit = self._load_json(eval_dir / "phase4_calibration_reaudit.json")
        phase4_benchmark = self._load_json(eval_dir / "phase4_inference_benchmark.json")

        notes = [
            "NIH metrics reflect calibrated test set performance across 6 multi-label diseases.",
            "CheXpert metrics demonstrate cross-hospital domain generalization when transferring locked NIH thresholds.",
            "VinDr metrics represent 15,000-image external evaluation of binary ResNet18 checkpoints.",
            "Phase 3 external validation encompasses CheXpert, VinDr-CXR, and MIMIC-CXR compliance audits.",
            "Phase 4 high-resolution multi-scale FPN achieves 0.7690 Pneumothorax AUROC (+25.4% gain over 224x224 baseline).",
            "Phase 4B independently verified baseline reconciliation, controlled 6-model ablations, and Holm-Bonferroni corrections.",
            "Temperature scaling and operating points B and C are research benchmarks; production inference enforces Point A.",
            "All metrics represent retrospective benchmark evaluation and do NOT establish clinical diagnostic validity."
        ]

        return ResearchMetricsResponse(
            nih_test_metrics=nih_metrics,
            chexpert_external_metrics=chexpert_metrics,
            vindr_external_metrics=vindr_metrics,
            external_validation_metrics=ext_metrics,
            threshold_comparison=thresh_comp,
            manual_tests_audit=manual_audit,
            expanded_cohort_comparison=cohort_comparison,
            threshold_optimization_report=thresh_opt_report,
            uncertain_label_sensitivity=uncertainty_analysis,
            calibration_analysis=calibration_analysis,
            resolution_ablation=resolution_ablation,
            phase4_experiment_manifest=phase4_manifest,
            resolution_comparison=resolution_comp,
            loss_function_comparison=loss_comp,
            phase4_classwise_metrics=phase4_classwise,
            phase4_calibration_report=phase4_calibration,
            phase4_baseline_reconciliation=phase4_reconciliation,
            phase4_prediction_reproduction=phase4_reproduction,
            phase4_ablation_results=phase4_ablation,
            phase4_statistical_audit=phase4_statistical,
            phase4_calibration_reaudit=phase4_calib_reaudit,
            phase4_inference_benchmark=phase4_benchmark,
            active_threshold_set="NIH ChestX-ray14 Locked Production Thresholds (Frozen)",
            notes=notes
        )

    @staticmethod
    def _load_json(path: Path) -> Optional[Dict[str, Any]]:
        """Safely loads a JSON file dictionary if present."""
        if not path.exists():
            return None
        try:
            with open(path, mode="r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading JSON from {path}: {e}")
            return None

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
