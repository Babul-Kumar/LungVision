import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root is the parent directory of backend/ (i.e. 'computer vision')
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application runtime configuration with environment variable overrides."""
    
    # Application Info
    APP_NAME: str = "LungVision AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ]
    
    # Device Configuration ('auto', 'cpu', 'cuda')
    TORCH_DEVICE: str = "auto"
    
    # Checkpoint Paths (Resolved relative to project root)
    DENSENET121_CHECKPOINT_PATH: Path = BASE_DIR / "models" / "densenet121" / "densenet121_nih_best.pth"
    RESNET18_ORIGINAL_PATH: Path = BASE_DIR / "models" / "resnet18" / "original_resnet18.pth"
    RESNET18_AUGMENTED_PATH: Path = BASE_DIR / "models" / "resnet18" / "blur_augmented_resnet18.pth"
    MULTISCALE_512_PATH: Path = BASE_DIR / "models" / "multiscale_densenet121" / "multiscale_densenet121_fpn_512_v1.pth"

    @property
    def RESOLVED_DENSENET121_PATH(self) -> Path:
        primary = self.DENSENET121_CHECKPOINT_PATH
        if primary.exists():
            return primary
        candidates = [
            BASE_DIR / "lungvision_disease_classification" / "checkpoints" / "densenet121_nih_best.pth",
            BASE_DIR / "checkpoints" / "densenet121_nih_best.pth",
            BASE_DIR / "LungVision_AI_Model_Registry" / "models" / "densenet121_nih_6disease" / "densenet121_nih_best.pth",
            BASE_DIR / "lungVision" / "densenet121_nih_best.pth",
        ]
        for c in candidates:
            if c.exists():
                return c
        return primary

    @property
    def RESOLVED_RESNET18_ORIGINAL_PATH(self) -> Path:
        primary = self.RESNET18_ORIGINAL_PATH
        if primary.exists():
            return primary
        candidates = [
            BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "models" / "original_resnet18" / "model.pth",
            BASE_DIR / "research" / "clinical_trials_and_evidence" / "models" / "original_resnet18" / "model.pth",
            BASE_DIR / "LungVision_AI_Model_Registry" / "models" / "original_resnet18" / "model.pth",
            BASE_DIR / "research" / "model_registry" / "models" / "original_resnet18" / "model.pth",
            BASE_DIR / "checkpoints" / "original_resnet18" / "model.pth",
        ]
        for c in candidates:
            if c.exists():
                return c
        return primary

    @property
    def RESOLVED_RESNET18_AUGMENTED_PATH(self) -> Path:
        primary = self.RESNET18_AUGMENTED_PATH
        if primary.exists():
            return primary
        candidates = [
            BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "models" / "blur_augmented_resnet18" / "model.pth",
            BASE_DIR / "research" / "clinical_trials_and_evidence" / "models" / "blur_augmented_resnet18" / "model.pth",
            BASE_DIR / "LungVision_AI_Model_Registry" / "models" / "blur_augmented_resnet18" / "model.pth",
            BASE_DIR / "research" / "model_registry" / "models" / "blur_augmented_resnet18" / "model.pth",
            BASE_DIR / "checkpoints" / "blur_augmented_resnet18" / "model.pth",
        ]
        for c in candidates:
            if c.exists():
                return c
        return primary

    @property
    def RESOLVED_MULTISCALE_PATH(self) -> Path:
        primary = self.MULTISCALE_512_PATH
        if primary.exists():
            return primary
        candidates = [
            BASE_DIR / "models" / "multiscale_densenet121" / "multiscale_densenet121_experimental.pth",
            BASE_DIR / "backend" / "lungvision_disease_classification" / "checkpoints" / "experimental" / "multiscale_densenet121_fpn_512_v1.pth",
            BASE_DIR / "lungvision_disease_classification" / "checkpoints" / "experimental" / "multiscale_densenet121_fpn_512_v1.pth",
            BASE_DIR / "lungvision_disease_classification" / "checkpoints" / "experimental" / "multiscale_densenet121_experimental.pth",
        ]
        for c in candidates:
            if c.exists():
                return c
        return primary
    
    # Research Metadata Paths
    @property
    def NIH_VALIDATION_THRESHOLDS_PATH(self) -> Path:
        primary = BASE_DIR / "research" / "chexpert_nih_eval" / "nih_validation_thresholds.csv"
        return primary if primary.exists() else (BASE_DIR / "lungVision" / "nih_validation_thresholds.csv")

    @property
    def NIH_TEST_METRICS_PATH(self) -> Path:
        primary = BASE_DIR / "research" / "chexpert_nih_eval" / "nih_test_calibrated_metrics.csv"
        return primary if primary.exists() else (BASE_DIR / "lungVision" / "nih_test_calibrated_metrics.csv")

    @property
    def CHEXPERT_METRICS_PATH(self) -> Path:
        primary = BASE_DIR / "research" / "chexpert_nih_eval" / "chexpert_external_metrics.csv"
        return primary if primary.exists() else (BASE_DIR / "lungVision" / "chexpert_external_metrics.csv")

    @property
    def VINDR_METRICS_PATH(self) -> Path:
        primary = BASE_DIR / "research" / "clinical_trials_and_evidence" / "vindr_external_evaluation" / "vindr_external_metrics.csv"
        return primary if primary.exists() else (BASE_DIR / "LungVision_AI_Final_Research_Bundle" / "vindr_external_evaluation" / "vindr_external_metrics.csv")

    @property
    def MODEL_REGISTRY_JSON_PATH(self) -> Path:
        primary = BASE_DIR / "models" / "model_registry.json"
        if primary.exists():
            return primary
        candidates = [
            BASE_DIR / "research" / "model_registry" / "configuration" / "model_registry.json",
            BASE_DIR / "LungVision_AI_Model_Registry" / "configuration" / "model_registry.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return primary
    
    # File Upload Limits
    MAX_IMAGE_SIZE_BYTES: int = 20 * 1024 * 1024  # 20 MB
    MAX_CT_SIZE_BYTES: int = 150 * 1024 * 1024     # 150 MB
    TEMP_UPLOAD_DIR: Path = BASE_DIR / "backend" / "data" / "uploads"
    
    # Model Image Configuration
    IMAGE_SIZE: int = 224
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

