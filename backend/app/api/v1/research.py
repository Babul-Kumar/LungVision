from fastapi import APIRouter, status
from backend.app.schemas.models import ResearchMetricsResponse
from backend.app.services.registry_service import registry_service


router = APIRouter(prefix="/research", tags=["Research Evidence & Historical Benchmarks"])


@router.get(
    "/metrics",
    response_model=ResearchMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Historical Benchmark Evaluation Metrics"
)
async def get_research_metrics() -> ResearchMetricsResponse:
    """Returns verified historical research evaluation metrics directly from saved audit artifacts:
    
    1. NIH test set calibrated performance.
    2. CheXpert cross-hospital external generalization.
    3. VinDr-CXR external validation on 15,000 images.
    """
    return registry_service.get_research_metrics()

