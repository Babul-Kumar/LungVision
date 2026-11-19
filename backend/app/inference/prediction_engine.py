"""
LungVision AI — Production Prediction Engine (Phase 7).

Manages the singleton model lifecycle, safe checkpoint verification,
quality auditing, exact deterministic preprocessing, continuous inference,
threshold application, uncertainty quantification, and visual explainability.
"""

from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

from backend.app.config import BASE_DIR, settings
from backend.app.core.logging import logger
from backend.app.models.multiscale_densenet import (
    DISEASE_LABELS,
    Phase7MultiScaleDenseNet121
)
from backend.app.inference.calibration import (
    CalibrationManager,
    OperatingPoint,
    calibration_manager
)
from backend.app.inference.gradcam import gradcam_generator
from backend.app.inference.preprocessing import phase7_preprocessor
from backend.app.inference.quality_checker import (
    ImageQualityChecker,
    QualityReport,
    QualityStatus,
    quality_checker
)
from backend.app.inference.uncertainty import (
    ClinicalRecommendation,
    ConfidenceLevel,
    UncertaintyAnalyzer,
    uncertainty_analyzer
)


EXPECTED_PHASE7_SHA256 = "7dc735ee00d9e57ece14cd33f2bdbd1e7fd226adca41cdb692ba0af2a33a9b23"
PRIMARY_CHECKPOINT_PATH = BASE_DIR / "models" / "multiscale_densenet121" / "msd121_224_weighted_bce_direct.pth"
FALLBACK_CHECKPOINT_PATH = BASE_DIR / "research" / "phase7_recovery" / "lungvision_phase7" / "checkpoints" / "msd121_224_weighted_bce_direct.pth"


class PredictionEngine:
    """
    Singleton inference engine for Phase 7 MultiScaleDenseNet121.
    Thread-safe and optimized for production deployment.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[Phase7MultiScaleDenseNet121] = None
        self.checkpoint_path: Optional[Path] = None
        self.checkpoint_sha256: Optional[str] = None
        self.model_loaded_at: Optional[str] = None
        self.total_inferences: int = 0
        self.is_healthy: bool = False
        self._initialize_model()

    def _resolve_checkpoint_path(self) -> Path:
        """Finds the verified Phase 7 checkpoint file."""
        if PRIMARY_CHECKPOINT_PATH.exists():
            return PRIMARY_CHECKPOINT_PATH
        if FALLBACK_CHECKPOINT_PATH.exists():
            return FALLBACK_CHECKPOINT_PATH
        raise FileNotFoundError(
            f"Phase 7 checkpoint not found at {PRIMARY_CHECKPOINT_PATH} or {FALLBACK_CHECKPOINT_PATH}"
        )

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Computes SHA-256 hash of checkpoint file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def _initialize_model(self):
        """Loads and strictly validates the Phase 7 MultiScaleDenseNet121 checkpoint."""
        try:
            logger.info("Initializing Phase 7 Prediction Engine...")
            ckpt_path = self._resolve_checkpoint_path()
            sha256 = self._compute_sha256(ckpt_path)

            if sha256 != EXPECTED_PHASE7_SHA256:
                logger.warning(
                    f"Checkpoint SHA-256 ({sha256}) differs from expected reference ({EXPECTED_PHASE7_SHA256})."
                )

            # Instantiate architecture
            model = Phase7MultiScaleDenseNet121(pretrained=False, dropout=0.25)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert param_count == 8071942, f"Unexpected parameter count: {param_count} != 8071942"

            # Load weights
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt_data["model_state_dict"] if "model_state_dict" in ckpt_data else ckpt_data

            missing, unexpected = model.load_state_dict(state_dict, strict=True)
            if missing or unexpected:
                raise RuntimeError(f"State dict mismatch! Missing: {missing}, Unexpected: {unexpected}")

            # Verify finite parameters
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    raise ValueError(f"Non-finite values detected in parameter: {name}")

            # Move to device and set eval mode
            model.to(self.device)
            model.eval()

            self.model = model
            self.checkpoint_path = ckpt_path
            self.checkpoint_sha256 = sha256
            self.model_loaded_at = datetime.now(timezone.utc).isoformat()
            self.is_healthy = True
            logger.info(
                f"Phase 7 MultiScaleDenseNet121 loaded successfully on {self.device}. "
                f"Parameters: {param_count:,}. SHA256: {sha256[:16]}..."
            )
        except Exception as exc:
            self.is_healthy = False
            logger.error(f"Failed to initialize Phase 7 prediction engine: {exc}", exc_info=True)
            raise

    def predict(
        self,
        image_input: Union[bytes, io.BytesIO, np.ndarray, Image.Image, str, Path],
        operating_point: Optional[OperatingPoint] = None,
        generate_cam: bool = False,
        cam_target_disease: Optional[str] = None
    ) -> Dict:
        """
        Executes end-to-end clinical inference pipeline:
        1. Image Quality Audit.
        2. Exact Preprocessing.
        3. Model Forward Pass.
        4. Calibrated Thresholding.
        5. Uncertainty & Abstention Analysis.
        6. Optional High-Resolution Explainability (Grad-CAM).
        """
        start_time = time.perf_counter()

        # Step 1: Pre-inference Image Quality Audit
        quality_rep = quality_checker.evaluate(image_input)
        if not quality_rep.can_proceed:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                "status": "REJECTED",
                "error": quality_rep.rejection_reason,
                "quality_report": quality_rep.to_dict(),
                "duration_ms": round(duration_ms, 2),
                "model_metadata": self.get_model_info()
            }

        if self.model is None or not self.is_healthy:
            raise RuntimeError("Phase 7 Prediction Engine is uninitialized or in unhealthy state.")

        # Step 2: Exact Preprocessing
        pil_img = phase7_preprocessor.decode_to_pil(image_input)
        input_tensor = phase7_preprocessor.preprocess(pil_img, device=self.device)

        # Step 3: Forward Pass
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        # Step 4: Thresholds & Uncertainty
        active_op = operating_point or calibration_manager.active_point
        thresholds = calibration_manager.get_thresholds(active_op)

        disease_details = []
        findings_map = {}

        for idx, disease in enumerate(DISEASE_LABELS):
            prob = float(probs[idx])
            thresh = float(thresholds.get(disease, 0.50))
            detail = uncertainty_analyzer.assess_disease(disease, prob, thresh)
            disease_details.append(detail)
            findings_map[disease] = detail.to_dict()

        has_quality_warnings = (quality_rep.status == QualityStatus.WARNING)
        uncertainty_sum = uncertainty_analyzer.summarize(disease_details, has_quality_warnings)

        # Step 5: Optional Grad-CAM
        cam_data = None
        if generate_cam:
            target_idx = 0
            if cam_target_disease and cam_target_disease in DISEASE_LABELS:
                target_idx = DISEASE_LABELS.index(cam_target_disease)
            else:
                # Target the disease with highest positive probability, or highest overall
                positives = [d for d in disease_details if d.is_positive]
                if positives:
                    best = max(positives, key=lambda d: d.probability)
                    target_idx = DISEASE_LABELS.index(best.disease)
                else:
                    best = max(disease_details, key=lambda d: d.probability)
                    target_idx = DISEASE_LABELS.index(best.disease)

            cam_result = gradcam_generator.generate(
                model=self.model,
                input_tensor=input_tensor,
                target_idx=target_idx,
                original_image=pil_img
            )
            cam_data = {
                "disease": cam_result["disease"],
                "target_idx": cam_result["target_idx"],
                "overlay_base64": cam_result["overlay_base64"],
                "heatmap_base64": cam_result["heatmap_base64"]
            }

        self.total_inferences += 1
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "COMPLETED",
            "findings": findings_map,
            "uncertainty": uncertainty_sum.to_dict(),
            "quality_report": quality_rep.to_dict(),
            "gradcam": cam_data,
            "duration_ms": round(duration_ms, 2),
            "model_metadata": self.get_model_info(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def get_health(self) -> Dict:
        """Returns runtime operational health metrics for the prediction engine."""
        cuda_info = {}
        if self.device.type == "cuda":
            cuda_info = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2),
                "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2)
            }

        return {
            "status": "healthy" if self.is_healthy else "unhealthy",
            "model_id": "lungvision_phase7_msd121_224_direct",
            "architecture": "MultiScaleDenseNet121",
            "version": "7.0.0",
            "device": str(self.device),
            "cuda_info": cuda_info,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "model_loaded_at": self.model_loaded_at,
            "total_inferences": self.total_inferences
        }

    def get_model_info(self) -> Dict:
        """Returns verified scientific and deployment metadata."""
        return {
            "model_id": "lungvision_phase7_msd121_224_direct",
            "model_name": "LungVision Phase 7 MultiScaleDenseNet121",
            "version": "7.0.0",
            "architecture": "MultiScaleDenseNet121",
            "backbone": "DenseNet121 (Multi-scale P2/P3/P4 Feature Fusion)",
            "input_resolution": [224, 224, 3],
            "preprocessing": "direct",
            "dataset": "NIH ChestX-ray14",
            "total_parameters": 8071942,
            "labels": DISEASE_LABELS,
            "active_operating_point": calibration_manager.active_point.value,
            "thresholds": calibration_manager.get_thresholds(),
            "validation_macro_auroc": 0.819726,
            "brier_scores": calibration_manager.get_brier_scores(),
            "per_disease_validation_metrics": calibration_manager.get_validation_metrics(),
            "intended_use": "Decision-support research and experimental clinical triaging"
        }


# Singleton engine instance
prediction_engine = PredictionEngine()
