from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class StandardResponse(BaseModel):
    """Base response model for uniform API success responses."""
    success: bool = True
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    disclaimer: str = (
        "For research and educational use only. Not a medical diagnosis. "
        "Independent verification by a qualified physician is strictly required."
    )


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ReadinessResponse(BaseModel):
    status: str
    device: str
    models_loaded: Dict[str, bool]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

