from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import Request, status
from fastapi.responses import JSONResponse


class LungVisionException(Exception):
    """Base exception for all LungVision application errors."""
    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_SERVER_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class FileValidationError(LungVisionException):
    """Raised when an uploaded file fails format or security checks."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="FILE_VALIDATION_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class FileTooLargeError(LungVisionException):
    """Raised when an uploaded file exceeds the configured size limit."""
    def __init__(self, max_size_bytes: int, actual_size_bytes: int):
        super().__init__(
            message=f"File exceeds maximum allowed size of {max_size_bytes / (1024*1024):.1f} MB",
            error_code="FILE_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={
                "max_size_bytes": max_size_bytes,
                "actual_size_bytes": actual_size_bytes
            }
        )


class ModelNotLoadedError(LungVisionException):
    """Raised when an inference request is routed to an uninitialized model."""
    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' is not loaded or unavailable",
            error_code="MODEL_NOT_LOADED",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={"model_id": model_id}
        )


class ResourceNotFoundError(LungVisionException):
    """Raised when a requested resource (CT scan, model metadata) does not exist."""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} '{resource_id}' was not found",
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


async def lungvision_exception_handler(request: Request, exc: LungVisionException) -> JSONResponse:
    """Formats custom LungVision exceptions into standardized JSON responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "details": exc.details
            }
        }
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all exception handler protecting against leaked stack traces."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred during processing.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "details": {"exception_type": type(exc).__name__}
            }
        }
    )

