import io
from PIL import Image
import torch
from torchvision import transforms
from backend.app.core.errors import FileValidationError
from backend.app.core.logging import logger


# Exact ImageNet normalization parameters used in training
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class XRayPreprocessor:
    """Preprocesses chest radiographs matching the exact training transformation."""

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def preprocess_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Loads image bytes, normalizes color space to RGB, applies resize and normalization."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Verify and fully load image bytes
            image.load()
        except Exception as e:
            logger.error(f"Image decode failed: {e}")
            raise FileValidationError(
                message="Uploaded image could not be decoded. File may be corrupted.",
                details={"error": str(e)}
            )

        return self.preprocess_pil(image)

    def preprocess_pil(self, image: Image.Image) -> torch.Tensor:
        """Transforms a PIL Image into a standardized 4D Tensor [1, 3, H, W]."""
        # Ensure 3-channel RGB (converting from Grayscale 'L', Palette 'P', or RGBA)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transformation pipeline
        tensor = self.transform(image)

        # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
        return tensor.unsqueeze(0)


# Default singleton preprocessor
preprocessor = XRayPreprocessor(image_size=224)

