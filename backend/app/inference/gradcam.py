"""
LungVision AI — High-Resolution Multi-Scale Grad-CAM Generator.

Leverages the multi-scale feature fusion layer (P2+P3+P4 at 56x56 resolution)
of Phase 7 MultiScaleDenseNet121 to produce localized, high-resolution visual
explainability heatmaps for each target disease.
"""

import base64
import io
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.cm as cm

from backend.app.models.multiscale_densenet import DISEASE_LABELS


class GradCAMGenerator:
    """Generates localized Grad-CAM heatmaps for MultiScaleDenseNet121."""

    def __init__(self, target_layer_name: str = "fuse"):
        self.target_layer_name = target_layer_name

    def generate(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        target_idx: int,
        original_image: Optional[Image.Image] = None
    ) -> Dict:
        """
        Computes Grad-CAM for target_idx on input_tensor.
        Returns:
            {
                "disease": str,
                "target_idx": int,
                "heatmap_normalized": np.ndarray (H, W),
                "overlay_base64": Optional[str],
                "heatmap_base64": Optional[str]
            }
        """
        model.eval()
        target_disease = DISEASE_LABELS[target_idx] if target_idx < len(DISEASE_LABELS) else f"Class_{target_idx}"

        # Target layer for Phase 7 MultiScaleDenseNet121
        target_layer = getattr(model, self.target_layer_name, None)
        if target_layer is None:
            # Fallback for FPN or alternative architectures
            target_layer = getattr(model, "final_norm", None)

        if target_layer is None:
            raise ValueError(f"Could not locate target layer '{self.target_layer_name}' in model.")

        # Ensure gradient tracking on input
        x = input_tensor.clone().detach().requires_grad_(True)

        activations = []
        def forward_hook(module, inp, out):
            activations.append(out)

        hook = target_layer.register_forward_hook(forward_hook)

        # Forward pass with gradients enabled
        with torch.enable_grad():
            logits = model(x)
            score = logits[0, target_idx]
            grads = torch.autograd.grad(score, activations[0], retain_graph=False, create_graph=False)[0]

        hook.remove()

        act = activations[0].detach()  # [1, C, H, W]
        grads = grads.detach()         # [1, C, H, W]

        # Global average pool gradients to get channel importance weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * act).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)

        # Upsample CAM to match target resolution
        target_h, target_w = (original_image.height, original_image.width) if original_image else (224, 224)
        cam = F.interpolate(cam, size=(target_h, target_w), mode="bilinear", align_corners=False)

        cam_arr = cam.squeeze().cpu().numpy()
        cam_min, cam_max = float(cam_arr.min()), float(cam_arr.max())
        if cam_max > cam_min:
            cam_norm = (cam_arr - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_arr)

        overlay_b64 = None
        heatmap_b64 = None

        if original_image is not None:
            # Generate jet colormap (compatible with all matplotlib versions)
            try:
                import matplotlib.pyplot as plt
                colormap = plt.get_cmap("jet")
            except Exception:
                colormap = cm.jet if hasattr(cm, "jet") else cm.viridis
            colored_cam = colormap(cam_norm)[:, :, :3]  # [H, W, 3] in [0, 1]
            colored_cam_uint8 = (colored_cam * 255).astype(np.uint8)

            # Base image in RGB
            orig_rgb = original_image.convert("RGB").resize((target_w, target_h), Image.Resampling.BILINEAR)
            orig_arr = np.array(orig_rgb, dtype=np.float32)

            # Alpha blend (0.55 original + 0.45 CAM heatmap)
            blended = (0.55 * orig_arr + 0.45 * (colored_cam * 255.0)).clip(0, 255).astype(np.uint8)
            overlay_pil = Image.fromarray(blended)

            buf = io.BytesIO()
            overlay_pil.save(buf, format="JPEG", quality=90)
            overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

            cam_pil = Image.fromarray(colored_cam_uint8)
            buf2 = io.BytesIO()
            cam_pil.save(buf2, format="PNG")
            heatmap_b64 = f"data:image/png;base64,{base64.b64encode(buf2.getvalue()).decode('utf-8')}"

        return {
            "disease": target_disease,
            "target_idx": target_idx,
            "heatmap_normalized": cam_norm,
            "overlay_base64": overlay_b64,
            "heatmap_base64": heatmap_b64
        }


gradcam_generator = GradCAMGenerator()
