"""
LungVision AI — Phase 7 Exact Preprocessing Pipeline.

Strictly reproduces the deterministic transforms from Phase 7 recovery training:
(notebook/lungvision-ai-phase-7-recovery-run-part2.ipynb, Cell 14).

Preprocessing Modes:
1. "direct": torchvision.transforms.Resize((size, size), interpolation=BILINEAR)
2. "letterbox": Aspect-ratio preserving resize with zero-padding (black border).

Both modes convert inputs to RGB, convert to float32 tensor scaled to [0.0, 1.0],
and apply ImageNet normalization:
  Mean: [0.485, 0.456, 0.406]
  Std:  [0.229, 0.224, 0.225]
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Letterbox:
    """Aspect-ratio preserving resize with symmetric black canvas padding."""

    def __init__(self, size: int, fill: Tuple[int, int, int] = (0, 0, 0)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        w, h = image.size
        scale = min(self.size / w, self.size / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        image = image.resize((nw, nh), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        left = (self.size - nw) // 2
        top = (self.size - nh) // 2
        canvas.paste(image, (left, top))
        return canvas


def build_phase7_transform(size: int = 224, preprocessing: str = "direct") -> transforms.Compose:
    """
    Constructs the exact Phase 7 training & validation transform.
    Directly matches Cell 14 of lungvision-ai-phase-7-recovery-run-part2.ipynb.
    """
    if preprocessing == "direct":
        geometry = transforms.Resize(
            (int(size), int(size)),
            interpolation=transforms.InterpolationMode.BILINEAR
        )
    elif preprocessing == "letterbox":
        geometry = Letterbox(size=int(size))
    else:
        raise ValueError(f"Unknown preprocessing mode: {preprocessing}. Expected 'direct' or 'letterbox'.")

    return transforms.Compose([
        geometry,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])


class Phase7Preprocessor:
    """
    Production-grade preprocessor for Phase 7 MultiScaleDenseNet121.
    Accepts raw bytes, PIL Image, numpy array, or file paths and outputs
    standardized PyTorch tensors matching Phase 7 validation specifications.
    """

    def __init__(self, size: int = 224, preprocessing: str = "direct"):
        self.size = size
        self.preprocessing = preprocessing
        self.transform = build_phase7_transform(size=size, preprocessing=preprocessing)

    def decode_to_pil(self, image_input: Union[bytes, io.BytesIO, np.ndarray, Image.Image, str, Path]) -> Image.Image:
        """Decodes arbitrary image input into a verified RGB PIL Image."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            with Image.open(path) as img:
                return img.convert("RGB")

        if isinstance(image_input, bytes):
            stream = io.BytesIO(image_input)
            with Image.open(stream) as img:
                return img.convert("RGB")

        if isinstance(image_input, io.BytesIO):
            image_input.seek(0)
            with Image.open(image_input) as img:
                return img.convert("RGB")

        if isinstance(image_input, np.ndarray):
            # Check dimensions: [H, W], [H, W, 1], [H, W, 3], or [H, W, 4]
            arr = image_input.copy()
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0 and arr.min() >= 0.0:
                    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr, mode="L").convert("RGB")
            elif arr.ndim == 3:
                if arr.shape[2] == 1:
                    return Image.fromarray(arr[:, :, 0], mode="L").convert("RGB")
                elif arr.shape[2] == 3:
                    return Image.fromarray(arr, mode="RGB")
                elif arr.shape[2] == 4:
                    return Image.fromarray(arr[:, :, :3], mode="RGB")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def preprocess(
        self,
        image_input: Union[bytes, io.BytesIO, np.ndarray, Image.Image, str, Path],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Executes full preprocessing pipeline.
        Returns: [1, 3, size, size] tensor on the requested device.
        """
        pil_img = self.decode_to_pil(image_input)
        tensor = self.transform(pil_img)  # [3, size, size]
        batch_tensor = tensor.unsqueeze(0)  # [1, 3, size, size]
        if device is not None:
            batch_tensor = batch_tensor.to(device)
        return batch_tensor


# Singleton preprocessor for standard 224 direct Phase 7 model
phase7_preprocessor = Phase7Preprocessor(size=224, preprocessing="direct")
