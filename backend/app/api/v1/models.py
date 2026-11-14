from fastapi import APIRouter, status
from backend.app.schemas.models import ModelListResponse, ModelRegistryEntry
from backend.app.services.registry_service import registry_service


router = APIRouter(prefix="/models", tags=["Model Registry & Metadata"])


@router.get(
    "",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    summary="List Registered Neural Networks"
)
async def list_models() -> ModelListResponse:
    """Lists all verified models in the registry, their architectures, checkpoint paths, and metrics."""
    return registry_service.get_models()


@router.get(
    "/{model_id}",
    response_model=ModelRegistryEntry,
    status_code=status.HTTP_200_OK,
    summary="Get Model Details by ID"
)
async def get_model_details(model_id: str) -> ModelRegistryEntry:
    """Retrieves metadata, training datasets, and validation calibration for a specific model."""
    return registry_service.get_model_by_id(model_id)

