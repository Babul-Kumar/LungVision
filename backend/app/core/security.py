import os
import re
import uuid
from pathlib import Path
from typing import Set, Tuple
from fastapi import UploadFile
from backend.app.core.errors import FileValidationError, FileTooLargeError
from backend.app.core.logging import logger


ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_IMAGE_MIME_TYPES: Set[str] = {"image/png", "image/jpeg", "image/webp"}

ALLOWED_CT_EXTENSIONS: Set[str] = {".zip", ".nii", ".gz", ".mhd", ".dcm"}


def sanitize_filename(filename: str) -> str:
    """Strips directory traversal sequences and unsafe characters."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
    if not clean_name:
        clean_name = f"upload_{uuid.uuid4().hex[:8]}"
    return clean_name


def generate_secure_storage_path(base_dir: Path, original_filename: str) -> Tuple[Path, str]:
    """Generates a non-conflicting UUID-prefixed file path."""
    base_dir.mkdir(parents=True, exist_ok=True)
    clean_name = sanitize_filename(original_filename)
    unique_id = uuid.uuid4().hex[:12]
    secure_filename = f"{unique_id}_{clean_name}"
    return base_dir / secure_filename, unique_id


def validate_image_file(file: UploadFile, content: bytes, max_size_bytes: int) -> None:
    """Validates image payload size, extension, and magic numbers."""
    if len(content) > max_size_bytes:
        raise FileTooLargeError(max_size_bytes=max_size_bytes, actual_size_bytes=len(content))
    
    if len(content) < 16:
        raise FileValidationError(
            message="Uploaded file is empty or too small to be a valid image",
            details={"size_bytes": len(content)}
        )
    
    # Check extension
    filename = file.filename or "unknown.png"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise FileValidationError(
            message=f"Unsupported image extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
            details={"filename": filename, "extension": ext}
        )
    
    # Validate magic bytes
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = content.startswith(b"\xff\xd8\xff")
    is_webp = content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    
    if not (is_png or is_jpeg or is_webp):
        raise FileValidationError(
            message="File header signature does not match expected image type (PNG, JPEG, WebP)",
            details={"filename": filename}
        )


def validate_ct_archive(file: UploadFile, content: bytes, max_size_bytes: int) -> None:
    """Validates CT archive or volume payload with magic byte verification."""
    if len(content) > max_size_bytes:
        raise FileTooLargeError(max_size_bytes=max_size_bytes, actual_size_bytes=len(content))
    
    if len(content) < 32:
        raise FileValidationError(
            message="Uploaded CT volume is empty or invalid",
            details={"size_bytes": len(content)}
        )
    
    filename = file.filename or "unknown.zip"
    lower_name = filename.lower()
    valid_ext = any(lower_name.endswith(ext) for ext in ALLOWED_CT_EXTENSIONS)
    
    if not valid_ext:
        raise FileValidationError(
            message=f"Unsupported CT archive extension. Allowed: {', '.join(sorted(ALLOWED_CT_EXTENSIONS))}",
            details={"filename": filename}
        )

    # Magic byte signature validation
    is_zip = content.startswith(b"PK\x03\x04") or content.startswith(b"PK\x05\x06") or content.startswith(b"PK\x07\x08")
    is_gzip = content.startswith(b"\x1f\x8b")
    is_dicom = (len(content) > 132 and content[128:132] == b"DICM") or content.startswith(b"\x00\x00") or lower_name.endswith(".dcm")
    is_mhd = lower_name.endswith(".mhd") and (b"ObjectType" in content[:256] or b"NDims" in content[:256] or b"DimSize" in content[:256])
    is_nii = lower_name.endswith(".nii") and (len(content) >= 348)

    if lower_name.endswith(".zip") and not is_zip:
        raise FileValidationError(
            message="File extension is .zip but content lacks valid ZIP archive magic bytes",
            details={"filename": filename}
        )
    elif (lower_name.endswith(".gz") or lower_name.endswith(".nii.gz")) and not is_gzip:
        raise FileValidationError(
            message="File extension indicates gzip compression but lacks gzip magic bytes",
            details={"filename": filename}
        )


