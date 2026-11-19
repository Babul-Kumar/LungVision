from fastapi import APIRouter
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.xray import router as xray_router
from backend.app.api.v1.ct import router as ct_router
from backend.app.api.v1.models import router as models_router
from backend.app.api.v1.research import router as research_router
from backend.app.api.v1.reconstruction import router as reconstruction_router
from backend.app.api.v1.predictions import router as predictions_router


api_router = APIRouter()

# Attach v1 endpoints
api_router.include_router(health_router)
api_router.include_router(xray_router)
api_router.include_router(ct_router)
api_router.include_router(models_router)
api_router.include_router(research_router)
api_router.include_router(reconstruction_router)
api_router.include_router(predictions_router)

