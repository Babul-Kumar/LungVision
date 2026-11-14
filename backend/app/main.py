import sys
from pathlib import Path

# Ensure project root is present on sys.path regardless of execution working directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.router import api_router
from backend.app.config import settings
from backend.app.core.errors import (
    LungVisionException,
    generic_exception_handler,
    lungvision_exception_handler
)
from backend.app.core.logging import logger
from backend.app.services.model_loader import model_loader


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle management."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...")
    
    # Create temp upload directory if not present
    settings.TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load neural network checkpoints into memory once on startup
    try:
        model_loader.load_all_models()
    except Exception as e:
        logger.error(f"Critical error during model startup: {e}")
    
    yield
    
    logger.info("Shutting down LungVision AI application...")


def create_application() -> FastAPI:
    """Factory creating and configuring the FastAPI instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Experimental decision-support API for 2D chest radiograph pathology screening "
            "and 3D computed tomography (CT) volumetric exploration. "
            "For research and educational use only. Not a medical diagnostic device."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register standardized error handlers
    app.add_exception_handler(LungVisionException, lungvision_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Mount v1 API routes
    app.include_router(api_router, prefix=settings.API_V1_STR)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "health": f"{settings.API_V1_STR}/health",
            "ready": f"{settings.API_V1_STR}/ready",
            "models": f"{settings.API_V1_STR}/models",
            "disclaimer": "For research and educational use only. Not for clinical diagnostic use."
        }

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)


