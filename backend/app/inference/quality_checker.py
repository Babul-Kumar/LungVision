"""
LungVision AI — Clinical Image Quality & Plausibility Checker.

Enforces clinical quality gates prior to model inference:
1. File format and bitstream decodability.
2. Spatial resolution limits (minimum 128x128).
3. Aspect ratio plausibility for thoracic radiographs.
4. Contrast, dynamic range, and blank/uniform image detection.
5. High-frequency focus / blur assessment via discrete Laplacian variance.
6. Radiographic intensity distribution heuristics.
"""

from dataclasses import dataclass, field
from enum import Enum
import io
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


class QualityStatus(str, Enum):
    PASSED = "PASSED"
    WARNING = "WARNING"
    IMAGE_QUALITY_INSUFFICIENT = "IMAGE_QUALITY_INSUFFICIENT"
    UNABLE_TO_ASSESS = "UNABLE_TO_ASSESS"


@dataclass
class QualityMetrics:
    width: int = 0
    height: int = 0
    aspect_ratio: float = 1.0
    mean_intensity: float = 0.0
    std_intensity: float = 0.0
    min_intensity: int = 0
    max_intensity: int = 255
    blur_variance: float = 0.0
    is_blurry: bool = False
    is_blank: bool = False
    is_overexposed: bool = False
    is_underexposed: bool = False


@dataclass
class QualityReport:
    status: QualityStatus
    can_proceed: bool
    metrics: QualityMetrics
    warnings: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "can_proceed": self.can_proceed,
            "metrics": {
                "width": self.metrics.width,
                "height": self.metrics.height,
                "aspect_ratio": round(self.metrics.aspect_ratio, 3),
                "mean_intensity": round(self.metrics.mean_intensity, 2),
                "std_intensity": round(self.metrics.std_intensity, 2),
                "blur_variance": round(self.metrics.blur_variance, 2),
                "is_blurry": self.metrics.is_blurry,
                "is_blank": self.metrics.is_blank,
                "is_overexposed": self.metrics.is_overexposed,
                "is_underexposed": self.metrics.is_underexposed
            },
            "warnings": self.warnings,
            "rejection_reason": self.rejection_reason
        }


class ImageQualityChecker:
    """Evaluates raw radiographic images against clinical acceptance criteria."""

    MIN_DIMENSION: int = 128
    RECOMMENDED_MIN_DIMENSION: int = 224
    MIN_ASPECT_RATIO: float = 0.33
    MAX_ASPECT_RATIO: float = 3.0
    BLUR_THRESHOLD: float = 10.0
    MIN_CONTRAST_STD: float = 5.0
    MIN_MEAN_INTENSITY: float = 15.0
    MAX_MEAN_INTENSITY: float = 240.0

    @staticmethod
    def _compute_laplacian_variance(gray_arr: np.ndarray) -> float:
        """
        Computes variance of the discrete 3x3 Laplacian operator over the grayscale array.
        Standard focus metric (higher variance indicates sharper edges/focal structures).
        """
        # Discrete Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        h, w = gray_arr.shape
        if h < 3 or w < 3:
            return 0.0

        padded = np.pad(gray_arr.astype(np.float64), pad_width=1, mode="reflect")
        lap = (
            padded[:-2, 1:-1] +
            padded[2:, 1:-1] +
            padded[1:-1, :-2] +
            padded[1:-1, 2:] -
            4.0 * padded[1:-1, 1:-1]
        )
        return float(np.var(lap))

    def evaluate(
        self,
        image_input: Union[bytes, io.BytesIO, np.ndarray, Image.Image, str, Path]
    ) -> QualityReport:
        """Runs comprehensive validation checks on the provided input image."""
        warnings: List[str] = []
        metrics = QualityMetrics()

        # 1. Image loading & decoding check
        try:
            if isinstance(image_input, Image.Image):
                pil_img = image_input.copy()
            elif isinstance(image_input, (str, Path)):
                p = Path(image_input)
                if not p.exists():
                    return QualityReport(
                        status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                        can_proceed=False,
                        metrics=metrics,
                        rejection_reason=f"File not found: {p}"
                    )
                with Image.open(p) as img:
                    pil_img = img.copy()
            elif isinstance(image_input, (bytes, io.BytesIO)):
                stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
                stream.seek(0)
                with Image.open(stream) as img:
                    pil_img = img.copy()
            elif isinstance(image_input, np.ndarray):
                arr = image_input.copy()
                if arr.dtype != np.uint8:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 else np.clip(arr, 0, 255).astype(np.uint8)
                if arr.ndim == 2:
                    pil_img = Image.fromarray(arr, mode="L")
                elif arr.ndim == 3:
                    pil_img = Image.fromarray(arr[:, :, :3], mode="RGB")
                else:
                    return QualityReport(
                        status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                        can_proceed=False,
                        metrics=metrics,
                        rejection_reason=f"Unsupported numpy array dimensions: {arr.ndim}"
                    )
            else:
                return QualityReport(
                    status=QualityStatus.UNABLE_TO_ASSESS,
                    can_proceed=False,
                    metrics=metrics,
                    rejection_reason=f"Unsupported input type: {type(image_input)}"
                )
        except Exception as exc:
            return QualityReport(
                status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                can_proceed=False,
                metrics=metrics,
                rejection_reason=f"Corrupt or unreadable image bitstream: {str(exc)}"
            )

        # 2. Dimensions & aspect ratio
        w, h = pil_img.size
        metrics.width = w
        metrics.height = h
        metrics.aspect_ratio = float(w / max(h, 1))

        if w < self.MIN_DIMENSION or h < self.MIN_DIMENSION:
            return QualityReport(
                status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                can_proceed=False,
                metrics=metrics,
                rejection_reason=f"Resolution ({w}x{h}) is below clinical safety minimum ({self.MIN_DIMENSION}x{self.MIN_DIMENSION})."
            )

        if w < self.RECOMMENDED_MIN_DIMENSION or h < self.RECOMMENDED_MIN_DIMENSION:
            warnings.append(
                f"Sub-optimal resolution ({w}x{h}); recommended minimum is {self.RECOMMENDED_MIN_DIMENSION}x{self.RECOMMENDED_MIN_DIMENSION}."
            )

        if metrics.aspect_ratio < self.MIN_ASPECT_RATIO or metrics.aspect_ratio > self.MAX_ASPECT_RATIO:
            warnings.append(
                f"Unusual thoracic aspect ratio ({metrics.aspect_ratio:.2f}); standard frontal radiographs are typically between 0.6 and 1.6."
            )

        # 3. Grayscale conversion for photometric checks
        gray_img = pil_img.convert("L")
        gray_arr = np.array(gray_img, dtype=np.uint8)

        metrics.mean_intensity = float(np.mean(gray_arr))
        metrics.std_intensity = float(np.std(gray_arr))
        metrics.min_intensity = int(np.min(gray_arr))
        metrics.max_intensity = int(np.max(gray_arr))

        # Check for blank / low-contrast image
        if metrics.std_intensity < self.MIN_CONTRAST_STD:
            metrics.is_blank = True
            return QualityReport(
                status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                can_proceed=False,
                metrics=metrics,
                rejection_reason="Image lacks sufficient radiographic contrast or is completely uniform."
            )

        # Check for extreme over/underexposure
        if metrics.mean_intensity < self.MIN_MEAN_INTENSITY:
            metrics.is_underexposed = True
            return QualityReport(
                status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                can_proceed=False,
                metrics=metrics,
                rejection_reason="Radiograph is severely underexposed (mean intensity < 15/255)."
            )

        if metrics.mean_intensity > self.MAX_MEAN_INTENSITY:
            metrics.is_overexposed = True
            return QualityReport(
                status=QualityStatus.IMAGE_QUALITY_INSUFFICIENT,
                can_proceed=False,
                metrics=metrics,
                rejection_reason="Radiograph is severely overexposed/washed out (mean intensity > 240/255)."
            )

        # 4. Blur assessment via Laplacian variance
        metrics.blur_variance = self._compute_laplacian_variance(gray_arr)
        if metrics.blur_variance < self.BLUR_THRESHOLD:
            metrics.is_blurry = True
            warnings.append(
                f"High-frequency detail is degraded (blur score: {metrics.blur_variance:.1f} < {self.BLUR_THRESHOLD}); diagnostic fine-scale accuracy may be reduced."
            )

        # Determine overall status
        status = QualityStatus.WARNING if len(warnings) > 0 else QualityStatus.PASSED
        return QualityReport(
            status=status,
            can_proceed=True,
            metrics=metrics,
            warnings=warnings
        )


quality_checker = ImageQualityChecker()
