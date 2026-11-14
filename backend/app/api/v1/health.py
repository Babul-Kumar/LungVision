from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from backend.app.config import settings
from backend.app.schemas.common import HealthResponse, ReadinessResponse
from backend.app.services.model_loader import model_loader


router = APIRouter(tags=["System Health & Diagnostics"])


@router.get("/health", response_model=HealthResponse, summary="Liveness Probe")
async def health_check() -> HealthResponse:
    """Returns 200 OK if the FastAPI process is running."""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness Probe")
async def readiness_check():
    """Verifies that primary neural network models are in memory and ready for inference."""
    is_ready, models_map = model_loader.is_ready()
    
    response = ReadinessResponse(
        status="ready" if is_ready else "degraded",
        device=str(model_loader.device),
        models_loaded=models_map
    )
    
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump()
        )
    return response

