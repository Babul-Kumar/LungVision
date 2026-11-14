import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from backend.app.config import settings
from backend.app.core.errors import ModelNotLoadedError, ResourceNotFoundError
from backend.app.core.logging import logger
from backend.app.schemas.xray import (
    BinaryPrediction,
    DiseasePrediction,
    ModelInfo,
    PredictionSummary,
    XRayBinaryResponse,
    XRayMultiLabelResponse
)
from backend.app.services.model_loader import model_loader
from backend.app.services.preprocessing import preprocessor


# Verified 6-disease output ordering matching densenet121 checkpoint
LABELS_6_DISEASE: List[str] = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]

# Verified NIH validation thresholds (from lungVision/nih_validation_thresholds.csv)
DEFAULT_NIH_THRESHOLDS: Dict[str, float] = {
    "Cardiomegaly": 0.9501953125,
    "Edema": 0.876953125,
    "Consolidation": 0.73291015625,
    "Pneumonia": 0.7529296875,
    "Atelectasis": 0.74951171875,
    "Pneumothorax": 0.58251953125
}

# Verified locked thresholds for binary ResNet18 models
RESNET18_THRESHOLDS: Dict[str, float] = {
    "original_resnet18": 0.1764627248048782,
    "blur_augmented_resnet18": 0.187923327088356
}


class XRayService:
    """Handles 2D chest radiograph inference and calibrated thresholding."""

    def __init__(self):
        self.nih_thresholds = self._load_nih_thresholds()

    def _load_nih_thresholds(self) -> Dict[str, float]:
        """Loads thresholds from CSV if available, otherwise uses verified constants."""
        csv_path = settings.NIH_VALIDATION_THRESHOLDS_PATH
        if csv_path.exists():
            try:
                thresholds = {}
                with open(csv_path, mode="r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        disease = row.get("Disease")
                        thresh = row.get("Best_Threshold")
                        if disease and thresh:
                            thresholds[disease] = float(thresh)
                logger.info(f"Loaded {len(thresholds)} calibrated thresholds from {csv_path}")
                return thresholds
            except Exception as e:
                logger.warning(f"Error reading thresholds CSV ({e}), using default verified constants.")
        return DEFAULT_NIH_THRESHOLDS.copy()

    def predict_multilabel(self, image_bytes: bytes) -> XRayMultiLabelResponse:
        """Executes 6-disease multi-label classification on a chest radiograph."""
        start_time = time.perf_counter()

        # 1. Preprocess input image to exact [1, 3, 224, 224] tensor
        input_tensor = preprocessor.preprocess_bytes(image_bytes)
        input_tensor = input_tensor.to(model_loader.device)

        # 2. Retrieve loaded model
        model = model_loader.get_model("densenet121_nih_6disease")

        # 3. Forward pass
        with torch.no_grad():
            logits = model(input_tensor)  # Shape: [1, 6]
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().tolist()

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # 4. Map probabilities and apply thresholds
        predictions_dict: Dict[str, DiseasePrediction] = {}
        flagged_count = 0
        highest_disease = LABELS_6_DISEASE[0]
        highest_prob = -1.0

        for idx, disease in enumerate(LABELS_6_DISEASE):
            prob = float(probabilities[idx])
            threshold = float(self.nih_thresholds.get(disease, DEFAULT_NIH_THRESHOLDS[disease]))
            flag = bool(prob >= threshold)

            if flag:
                flagged_count += 1
            if prob > highest_prob:
                highest_prob = prob
                highest_disease = disease

            predictions_dict[disease] = DiseasePrediction(
                probability=round(prob, 4),
                threshold=round(threshold, 4),
                experimental_flag=flag
            )

        model_meta = model_loader.model_metadata.get("densenet121_nih_6disease", {})

        return XRayMultiLabelResponse(
            model=ModelInfo(
                id="densenet121_nih_6disease",
                architecture="DenseNet121",
                task="Multi-label chest X-ray disease classification",
                training_dataset="NIH ChestX-ray14",
                checkpoint_epoch=model_meta.get("epoch", 4),
                input_dimensions=[224, 224, 3]
            ),
            predictions=predictions_dict,
            summary=PredictionSummary(
                flagged_conditions_count=flagged_count,
                highest_probability_disease=highest_disease,
                highest_probability_value=round(highest_prob, 4)
            ),
            inference_time_ms=round(elapsed_ms, 2)
        )

    def predict_binary(self, image_bytes: bytes, model_id: str = "original_resnet18") -> XRayBinaryResponse:
        """Executes binary Normal vs. Abnormal classification via ResNet18."""
        if model_id not in ("original_resnet18", "blur_augmented_resnet18"):
            raise ResourceNotFoundError("Model", model_id)

        start_time = time.perf_counter()

        input_tensor = preprocessor.preprocess_bytes(image_bytes)
        input_tensor = input_tensor.to(model_loader.device)

        model = model_loader.get_model(model_id)

        with torch.no_grad():
            logits = model(input_tensor)
            prob = float(torch.sigmoid(logits).item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        threshold = RESNET18_THRESHOLDS.get(model_id, 0.1764627)
        flag = bool(prob >= threshold)
        classification = "Abnormal" if flag else "Normal"

        model_meta = model_loader.model_metadata.get(model_id, {})

        return XRayBinaryResponse(
            model=ModelInfo(
                id=model_id,
                architecture="ResNet18",
                task="Binary Normal vs Abnormal classification",
                training_dataset="IN-CXR",
                checkpoint_epoch=model_meta.get("epoch", 4 if model_id == "original_resnet18" else 1),
                input_dimensions=[224, 224, 3]
            ),
            prediction=BinaryPrediction(
                abnormal_probability=round(prob, 4),
                calibrated_threshold=round(threshold, 4),
                classification=classification,
                experimental_flag=flag
            ),
            inference_time_ms=round(elapsed_ms, 2)
        )


xray_service = XRayService()

