"""
LungVision AI — Dual-View Neural Shape Predictor Models
Module: backend/app/models/dual_view_shape_predictor.py
Version: 5.0.0

Predicts 3D Statistical Shape Model (SSM) deformation coefficients b in R^M
from paired 2D thoracic radiographs (Frontal PA/AP and Lateral LL).

Implements 4 controlled experimental architectures:
1. Model 1: Frontal-only baseline
2. Model 2: Lateral-only baseline
3. Model 3: Dual-view feature concatenation fusion
4. Model 4: Dual-view cross-attention fusion

Invariants:
- Zero CT spacing / CT dimension / target metadata leakage into features.
- Inputs strictly limited to 2D pixel matrices.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class RadiographBackbone(nn.Module):
    """
    Lightweight convolutional feature backbone for 224x224 thoracic radiographs.
    Extracts both spatial token features (B, 16, 64) for attention and global pooling (B, 256).
    """

    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),  # 7x7
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, feature_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),  # 4x4
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            global_feat: (B, feature_dim)
            spatial_tokens: (B, 16, feature_dim)
        """
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)  # (B, 256, 4, 4)

        b, c, h, w = c4.shape
        spatial_tokens = c4.flatten(2).transpose(1, 2)  # (B, 16, 256)
        global_feat = self.global_pool(c4).flatten(1)  # (B, 256)

        return global_feat, spatial_tokens


class FrontalOnlyShapePredictor(nn.Module):
    """Model 1: Frontal-only baseline shape predictor."""

    def __init__(self, num_modes: int = 20, feature_dim: int = 256):
        super().__init__()
        self.backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_modes),
        )

    def forward(self, frontal: torch.Tensor, lateral: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat, _ = self.backbone(frontal)
        coeffs = self.head(feat)
        return coeffs


class LateralOnlyShapePredictor(nn.Module):
    """Model 2: Lateral-only baseline shape predictor."""

    def __init__(self, num_modes: int = 20, feature_dim: int = 256):
        super().__init__()
        self.backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(128, num_modes),
        )

    def forward(self, frontal: Optional[torch.Tensor], lateral: torch.Tensor) -> torch.Tensor:
        feat, _ = self.backbone(lateral)
        coeffs = self.head(feat)
        return coeffs


class DualViewConcatShapePredictor(nn.Module):
    """Model 3: Dual-view concatenation feature fusion."""

    def __init__(self, num_modes: int = 20, feature_dim: int = 256):
        super().__init__()
        self.frontal_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.lateral_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_modes),
        )

    def forward(self, frontal: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        f_feat, _ = self.frontal_backbone(frontal)
        l_feat, _ = self.lateral_backbone(lateral)
        fused = torch.cat([f_feat, l_feat], dim=-1)
        coeffs = self.fusion_head(fused)
        return coeffs


class DualViewCrossAttentionShapePredictor(nn.Module):
    """
    Model 4: Dual-view cross-attention fusion.
    Cross-attention enables spatial tokens of the Frontal view (coronal plane) to attend
    directly to orthogonal depth features of the Lateral view (sagittal plane), resolving
    anteroposterior and cranial-caudal thoracic boundaries.
    """

    def __init__(self, num_modes: int = 20, feature_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.frontal_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.lateral_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)

        self.cross_attn_f2l = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.cross_attn_l2f = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=0.1
        )
        self.norm_f = nn.LayerNorm(feature_dim)
        self.norm_l = nn.LayerNorm(feature_dim)

        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_modes),
        )

    def forward(self, frontal: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        f_feat, f_tokens = self.frontal_backbone(frontal)  # (B, 256), (B, 16, 256)
        l_feat, l_tokens = self.lateral_backbone(lateral)  # (B, 256), (B, 16, 256)

        # Frontal tokens query Lateral tokens
        f_attn_out, _ = self.cross_attn_f2l(query=f_tokens, key=l_tokens, value=l_tokens)
        f_tokens_fused = self.norm_f(f_tokens + f_attn_out)

        # Lateral tokens query Frontal tokens
        l_attn_out, _ = self.cross_attn_l2f(query=l_tokens, key=f_tokens, value=f_tokens)
        l_tokens_fused = self.norm_l(l_tokens + l_attn_out)

        # Global average pool fused tokens
        f_pooled = f_tokens_fused.mean(dim=1)  # (B, 256)
        l_pooled = l_tokens_fused.mean(dim=1)  # (B, 256)

        fused = torch.cat([f_pooled, l_pooled], dim=-1)  # (B, 512)
        coeffs = self.head(fused)
        return coeffs


class HeteroscedasticDualViewCrossAttentionPredictor(nn.Module):
    """
    Phase 5B: Improved Heteroscedastic Dual-View Cross-Attention Shape Predictor.
    
    Features:
    1. Multi-head bidirectional cross-attention with residual feed-forward refinement.
    2. Heteroscedastic uncertainty head: outputs both mean deformation coefficients b
       and log-variance log(sigma_b^2) for uncertainty-aware shape regression.
    3. Multi-task left/right anatomical residual heads to ensure independent hemithorax
       asymmetry representation.
    """

    def __init__(
        self,
        num_modes: int = 20,
        feature_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_modes = num_modes
        self.frontal_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)
        self.lateral_backbone = RadiographBackbone(in_channels=1, feature_dim=feature_dim)

        self.cross_attn_f2l = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.cross_attn_l2f = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm_f1 = nn.LayerNorm(feature_dim)
        self.norm_l1 = nn.LayerNorm(feature_dim)

        # Residual feed-forward blocks
        self.ffn_f = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.norm_f2 = nn.LayerNorm(feature_dim)
        self.norm_l2 = nn.LayerNorm(feature_dim)

        # Shared fused projection
        self.shared_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 1. Primary shape coefficient mean head
        self.mean_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, num_modes),
        )

        # 2. Heteroscedastic log-variance uncertainty head
        self.logvar_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_modes),
        )

        # 3. Bilateral residual refinement (Left vs Right lung asymmetry)
        self.bilateral_residual_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_modes),
        )

    def forward(
        self,
        frontal: torch.Tensor,
        lateral: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        f_feat, f_tokens = self.frontal_backbone(frontal)
        l_feat, l_tokens = self.lateral_backbone(lateral)

        # Multi-head Cross-Attention
        f_attn_out, _ = self.cross_attn_f2l(query=f_tokens, key=l_tokens, value=l_tokens)
        f_tokens_fused = self.norm_f1(f_tokens + f_attn_out)
        f_tokens_fused = self.norm_f2(f_tokens_fused + self.ffn_f(f_tokens_fused))

        l_attn_out, _ = self.cross_attn_l2f(query=l_tokens, key=f_tokens, value=f_tokens)
        l_tokens_fused = self.norm_l1(l_tokens + l_attn_out)
        l_tokens_fused = self.norm_l2(l_tokens_fused + self.ffn_l(l_tokens_fused))

        f_pooled = f_tokens_fused.mean(dim=1)
        l_pooled = l_tokens_fused.mean(dim=1)

        fused = self.shared_proj(torch.cat([f_pooled, l_pooled], dim=-1))

        # Predict primary mean and bilateral residual
        base_mean = self.mean_head(fused)
        residual = 0.1 * self.bilateral_residual_head(fused)
        pred_coeffs = base_mean + residual

        if return_uncertainty:
            log_var = self.logvar_head(fused)
            # Bound log-variance between -6 and +4 for numerical stability
            log_var = torch.clamp(log_var, min=-6.0, max=4.0)
            return pred_coeffs, log_var
        return pred_coeffs


class HeteroscedasticShapeLoss(nn.Module):
    """
    Phase 5B: Uncertainty-Aware Negative Log-Likelihood with Mahalanobis Regularization.
    L = 0.5 * sum( exp(-s_m) * (b_m - b_m*)^2 + s_m ) + alpha * Mahalanobis(b)
    where s_m = log(sigma_m^2).
    """

    def __init__(self, eigenvalues: torch.Tensor, alpha: float = 0.005):
        super().__init__()
        safe_evals = torch.clamp(eigenvalues, min=1e-3)
        self.register_buffer("inv_eigenvalues", 1.0 / safe_evals)
        self.alpha = float(alpha)

    def forward(
        self,
        pred_coeffs: torch.Tensor,
        target_coeffs: torch.Tensor,
        log_var: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if log_var is None:
            mse = F.mse_loss(pred_coeffs, target_coeffs)
            mahalanobis = torch.mean(torch.sum((pred_coeffs ** 2) * self.inv_eigenvalues, dim=-1))
            return mse + self.alpha * mahalanobis

        # Gaussian Negative Log-Likelihood
        precision = torch.exp(-log_var)
        diff_sq = (pred_coeffs - target_coeffs) ** 2
        nll = 0.5 * torch.mean(torch.sum(precision * diff_sq + log_var, dim=-1))

        # Mahalanobis prior penalty
        mahalanobis = torch.mean(torch.sum((pred_coeffs ** 2) * self.inv_eigenvalues, dim=-1))
        return nll + self.alpha * mahalanobis


class MahalanobisShapeLoss(nn.Module):
    """
    Combined loss for shape parameter regression:
    L = MSE(b, b_target) + alpha * sum(b_m^2 / lambda_m)
    Penalizes deviations from ground-truth coefficients while regularizing
    biologically implausible shape variance via the Mahalanobis prior.
    """

    def __init__(self, eigenvalues: torch.Tensor, alpha: float = 0.005):
        super().__init__()
        # Clamp eigenvalues to avoid division by zero
        safe_evals = torch.clamp(eigenvalues, min=1e-3)
        self.register_buffer("inv_eigenvalues", 1.0 / safe_evals)
        self.alpha = float(alpha)

    def forward(self, pred_coeffs: torch.Tensor, target_coeffs: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred_coeffs, target_coeffs)
        mahalanobis = torch.mean(torch.sum((pred_coeffs ** 2) * self.inv_eigenvalues, dim=-1))
        return mse + self.alpha * mahalanobis


def create_shape_predictor(
    model_type: str = "dual_view_attention",
    num_modes: int = 20,
    feature_dim: int = 256,
) -> nn.Module:
    """Factory creating shape parameter predictor instances."""
    m_type = model_type.lower()
    if m_type in ("frontal_only", "model_1", "frontal"):
        return FrontalOnlyShapePredictor(num_modes=num_modes, feature_dim=feature_dim)
    elif m_type in ("lateral_only", "model_2", "lateral"):
        return LateralOnlyShapePredictor(num_modes=num_modes, feature_dim=feature_dim)
    elif m_type in ("dual_view_concat", "model_3", "concat"):
        return DualViewConcatShapePredictor(num_modes=num_modes, feature_dim=feature_dim)
    elif m_type in ("dual_view_attention", "model_4", "cross_attention", "attention"):
        return DualViewCrossAttentionShapePredictor(num_modes=num_modes, feature_dim=feature_dim)
    elif m_type in ("dual_view_heteroscedastic", "model_5", "dual_view_v2", "heteroscedastic"):
        return HeteroscedasticDualViewCrossAttentionPredictor(num_modes=num_modes, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown shape predictor model type: {model_type}")
