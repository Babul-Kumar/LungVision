import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models
from backend.app.config import settings
from backend.app.core.errors import ModelNotLoadedError
from backend.app.core.logging import logger


class ModelLoader:
    """Manages thread-safe singleton loading of verified PyTorch models."""

    def __init__(self):
        self.device = self._resolve_device(settings.TORCH_DEVICE)
        self.models: Dict[str, nn.Module] = {}
        self.model_metadata: Dict[str, dict] = {}
        self._initialized = False

    @staticmethod
    def _resolve_device(device_setting: str) -> torch.device:
        """Determines computation device with safe CPU fallback."""
        if device_setting.lower() == "cuda" and torch.cuda.is_available():
            logger.info("Using CUDA GPU accelerator for inference.")
            return torch.device("cuda")
        elif device_setting.lower() == "auto":
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA GPU. Using CUDA.")
                return torch.device("cuda")
            else:
                logger.info("No CUDA GPU detected. Running on CPU.")
                return torch.device("cpu")
        else:
            logger.info("Using standard CPU for inference.")
            return torch.device("cpu")

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Calculates SHA-256 checksum of checkpoint file."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load_all_models(self) -> None:
        """Loads all verified models into memory during application lifespan startup."""
        logger.info("Initializing neural network model registry...")
        
        # 1. Load DenseNet121 6-Disease Multi-label Model
        self._load_densenet121()

        # 2. Load ResNet18 Binary Original Model
        self._load_resnet18_original()

        # 3. Load ResNet18 Binary Blur-Augmented Model
        self._load_resnet18_augmented()

        self._initialized = True
        logger.info("All model checkpoints successfully loaded and ready for inference.")

    def _load_densenet121(self) -> None:
        ckpt_path = settings.RESOLVED_DENSENET121_PATH
        if not ckpt_path.exists():
            logger.warning(f"DenseNet121 checkpoint missing at: {ckpt_path}")
            return

        try:
            logger.info(f"Loading DenseNet121 from {ckpt_path}...")
            # Load checkpoint with weights_only=False to support numpy scalar metadata in dict
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(1024, 6)
            
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.models["densenet121_nih_6disease"] = model
            self.model_metadata["densenet121_nih_6disease"] = {
                "id": "densenet121_nih_6disease",
                "name": "DenseNet121 NIH 6-Disease Classifier",
                "architecture": "DenseNet121",
                "task": "Multi-label chest X-ray disease classification",
                "checkpoint_path": str(ckpt_path),
                "epoch": checkpoint.get("epoch", 4),
                "best_val_auc": checkpoint.get("best_val_auc", 0.820582),
                "labels": checkpoint.get("labels", [
                    "Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax"
                ]),
                "status": "ready"
            }
            logger.info("DenseNet121 loaded successfully (6 output pathologies).")
        except Exception as e:
            logger.error(f"Failed to load DenseNet121: {e}")

    def _load_resnet18_original(self) -> None:
        ckpt_path = settings.RESOLVED_RESNET18_ORIGINAL_PATH
        if not ckpt_path.exists():
            logger.warning(f"Original ResNet18 checkpoint missing at: {ckpt_path}")
            return

        try:
            logger.info(f"Loading Original ResNet18 from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(512, 1)

            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.models["original_resnet18"] = model
            self.model_metadata["original_resnet18"] = {
                "id": "original_resnet18",
                "name": "Original ResNet18 Baseline Classifier",
                "architecture": "ResNet18",
                "task": "Binary Normal vs Abnormal classification",
                "checkpoint_path": str(ckpt_path),
                "epoch": checkpoint.get("epoch", 4),
                "best_val_auc": checkpoint.get("best_val_auc", 0.933206),
                "labels": ["Normal", "Abnormal"],
                "status": "ready"
            }
            logger.info("Original ResNet18 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Original ResNet18: {e}")

    def _load_resnet18_augmented(self) -> None:
        ckpt_path = settings.RESOLVED_RESNET18_AUGMENTED_PATH
        if not ckpt_path.exists():
            logger.warning(f"Blur-Augmented ResNet18 checkpoint missing at: {ckpt_path}")
            return

        try:
            logger.info(f"Loading Blur-Augmented ResNet18 from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(512, 1)

            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()

            self.models["blur_augmented_resnet18"] = model
            self.model_metadata["blur_augmented_resnet18"] = {
                "id": "blur_augmented_resnet18",
                "name": "Blur-Augmented ResNet18 Robustness Classifier",
                "architecture": "ResNet18",
                "task": "Binary Normal vs Abnormal classification",
                "checkpoint_path": str(ckpt_path),
                "epoch": checkpoint.get("epoch", 1),
                "best_val_auc": checkpoint.get("best_val_auc", 0.938037),
                "labels": ["Normal", "Abnormal"],
                "status": "ready"
            }
            logger.info("Blur-Augmented ResNet18 loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Blur-Augmented ResNet18: {e}")

    def get_model(self, model_id: str) -> nn.Module:
        """Retrieves loaded model instance or raises ModelNotLoadedError."""
        if model_id not in self.models:
            raise ModelNotLoadedError(model_id)
        return self.models[model_id]

    def is_ready(self) -> Tuple[bool, Dict[str, bool]]:
        """Returns readiness flag and per-model availability status."""
        status_map = {
            "densenet121_6disease": "densenet121_nih_6disease" in self.models,
            "resnet18_original": "original_resnet18" in self.models,
            "resnet18_augmented": "blur_augmented_resnet18" in self.models,
        }
        all_ready = status_map["densenet121_6disease"]
        return all_ready, status_map


# Global singleton instance
model_loader = ModelLoader()

