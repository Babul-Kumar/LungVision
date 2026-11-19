"""
LungVision AI — MultiScaleDenseNet121 Architecture Definitions.

Contains:
1. Phase7MultiScaleDenseNet121:
   The experimentally validated Phase 7 recovery architecture (8,071,942 parameters).
   Features DenseNet121 multi-scale intermediate extraction (P2, P3, P4),
   lateral 1x1 projections, multi-scale feature fusion (384 -> 256),
   dual GAP+GMP pooling (512 dims), dropout (0.25), and 6 decoupled linear heads.
   Validated checkpoint: msd121_224_weighted_bce_direct.pth (Val Macro AUROC 0.819726).

2. MultiScaleDenseNet121FPN:
   The earlier Feature Pyramid Network (FPN) research variant with top-down connections,
   coarse/mid/fine decoupled heads, and optional Squeeze-and-Excitation / GAP-only ablations.

3. MultiScaleDenseNet121:
   Unified entrypoint that returns Phase7MultiScaleDenseNet121 by default, or
   MultiScaleDenseNet121FPN when legacy FPN/ablation parameters are specified.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import densenet121, DenseNet121_Weights


DISEASE_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]


# =============================================================================
# 1. PHASE 7 MULTISCALEDENSENET121 (PRODUCTION & EXPERIMENTALLY VALIDATED)
# =============================================================================

class Phase7MultiScaleDenseNet121(nn.Module):
    """
    Phase 7 MultiScaleDenseNet121 architecture.
    
    Source of truth: Phase 7 training & validation recovery run
    (notebook/lungvision-ai-phase-7-recovery-run-part2.ipynb).
    
    Key characteristics:
    - DenseNet121 backbone divided into sequential feature stages.
    - Multi-scale intermediate features extracted at:
        * p2: block1 (256 channels)
        * p3: block2 (512 channels)
        * p4: block3 (1024 channels)
    - Lateral 1x1 projections (256->128, 512->128, 1024->128).
    - Bilinear upsampling of p3, p4 to match p2 spatial resolution.
    - 3-scale concatenation (384 channels) fused via 3x3 Conv + BatchNorm + ReLU (256 channels).
    - Dual pooling: AdaptiveAvgPool2d(1) + AdaptiveMaxPool2d(1) -> 512 channels.
    - Dropout (p=0.25).
    - 6 decoupled disease classification heads (nn.Linear(512, 1)).
    - Exact parameter count: 8,071,942.
    - Output: [batch_size, 6] continuous logits.
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.25, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base = models.densenet121(weights=weights)
        f = base.features

        # DenseNet121 feature stages
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)
        self.block1 = f.denseblock1
        self.trans1 = f.transition1
        self.block2 = f.denseblock2
        self.trans2 = f.transition2
        self.block3 = f.denseblock3
        self.trans3 = f.transition3
        self.block4 = f.denseblock4
        self.final_norm = f.norm5

        # Lateral 1x1 projections: 256, 512, 1024 -> 128
        self.lat2 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(512, 128, kernel_size=1)
        self.lat4 = nn.Conv2d(1024, 128, kernel_size=1)

        # Multi-scale feature fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Dual pooling: GAP + GMP -> 256 + 256 = 512
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(dropout)

        # Decoupled pathology heads (6 independent binary heads)
        self.heads = nn.ModuleList([
            nn.Linear(512, 1) for _ in range(num_classes)
        ])

    def extract_multiscale_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract intermediate multi-scale feature maps (p2, p3, p4)."""
        x = self.stem(x)
        p2 = self.block1(x)
        x = self.trans1(p2)
        p3 = self.block2(x)
        x = self.trans2(p3)
        p4 = self.block3(x)
        return p2, p3, p4

    def forward_with_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and fused feature map (for Grad-CAM)."""
        x = self.stem(x)
        p2 = self.block1(x)
        x = self.trans1(p2)
        p3 = self.block2(x)
        x = self.trans2(p3)
        p4 = self.block3(x)
        x = self.trans3(p4)
        x = self.block4(x)
        x = self.final_norm(x)

        target_size = p2.shape[-2:]
        l2 = self.lat2(p2)
        l3 = F.interpolate(self.lat3(p3), size=target_size, mode="bilinear", align_corners=False)
        l4 = F.interpolate(self.lat4(p4), size=target_size, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([l2, l3, l4], dim=1))
        pooled = torch.cat([self.avg_pool(fused), self.max_pool(fused)], dim=1).flatten(1)
        pooled = self.dropout(pooled)
        logits = torch.cat([head(pooled) for head in self.heads], dim=1)
        return logits, fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing 6 independent logits in standard clinical order."""
        logits, _ = self.forward_with_features(x)
        return logits


# =============================================================================
# 2. LEGACY FPN RESEARCH ARCHITECTURE (PHASE 4B / PHASE 6 ABLATIONS)
# =============================================================================

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention for multi-scale feature maps."""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(in_channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class MultiScaleDenseNet121FPN(nn.Module):
    """Multi-Scale DenseNet121 with Feature Pyramid Network and Decoupled Pathology Heads."""

    def __init__(
        self,
        num_classes: int = 6,
        fpn_channels: int = 256,
        pretrained: bool = False,
        dropout_rate: float = 0.2,
        use_gmp: bool = True,
        use_se: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fpn_channels = fpn_channels
        self.use_gmp = use_gmp
        self.use_se = use_se

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_densenet = densenet121(weights=weights)
        features = base_densenet.features

        self.stage_c2 = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
            features.denseblock1,
            features.transition1,
            features.denseblock2
        )
        self.stage_c3 = nn.Sequential(
            features.transition2,
            features.denseblock3
        )
        self.stage_c4 = nn.Sequential(
            features.transition3,
            features.denseblock4,
            features.norm5
        )

        self.lateral_c4 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        self.smooth_p4 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p3 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )
        self.smooth_p2 = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True)
        )

        if use_se:
            self.att_p2 = ChannelAttention(fpn_channels)
            self.att_p3 = ChannelAttention(fpn_channels)
            self.att_p4 = ChannelAttention(fpn_channels)
        else:
            self.att_p2 = nn.Identity()
            self.att_p3 = nn.Identity()
            self.att_p4 = nn.Identity()

        pool_dim = fpn_channels * 2 if use_gmp else fpn_channels
        self.pool_dim = pool_dim

        self.head_fine = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)
        )
        self.head_mid = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)
        )
        self.head_coarse = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)
        )
        self.head_holistic = nn.Sequential(
            nn.Linear(pool_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        self.specialized_weight = nn.Parameter(torch.tensor(0.6))
        self.holistic_weight = nn.Parameter(torch.tensor(0.4))

    def _pool_features(self, p: torch.Tensor) -> torch.Tensor:
        gap = F.adaptive_avg_pool2d(p, (1, 1)).flatten(1)
        if self.use_gmp:
            gmp = F.adaptive_max_pool2d(p, (1, 1)).flatten(1)
            return torch.cat([gap, gmp], dim=1)
        return gap

    def extract_multiscale_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2 = self.stage_c2(x)
        c3 = self.stage_c3(c2)
        c4 = self.stage_c4(c3)

        m4 = self.lateral_c4(c4)
        m3 = self.lateral_c3(c3) + F.interpolate(m4, size=c3.shape[2:], mode="nearest")
        m2 = self.lateral_c2(c2) + F.interpolate(m3, size=c2.shape[2:], mode="nearest")

        p4 = self.att_p4(self.smooth_p4(m4))
        p3 = self.att_p3(self.smooth_p3(m3))
        p2 = self.att_p2(self.smooth_p2(m2))
        return p2, p3, p4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p2, p3, p4 = self.extract_multiscale_features(x)
        feat_p2 = self._pool_features(p2)
        feat_p3 = self._pool_features(p3)
        feat_p4 = self._pool_features(p4)

        out_coarse = self.head_coarse(feat_p4)
        out_mid = self.head_mid(feat_p3)
        out_fine = self.head_fine(feat_p2)

        out_specialized = torch.cat([out_coarse, out_mid, out_fine], dim=1)
        feat_fused = torch.cat([feat_p2, feat_p3, feat_p4], dim=1)
        out_holistic = self.head_holistic(feat_fused)

        w_spec = torch.sigmoid(self.specialized_weight)
        w_holi = torch.sigmoid(self.holistic_weight)
        w_sum = w_spec + w_holi
        w_spec = w_spec / w_sum
        w_holi = w_holi / w_sum

        logits = (w_spec * out_specialized) + (w_holi * out_holistic)
        return logits


# =============================================================================
# 3. UNIFIED DISPATCHER (PRESERVES 100% RETRO-COMPATIBILITY)
# =============================================================================

def MultiScaleDenseNet121(
    num_classes: int = 6,
    pretrained: bool = False,
    dropout: float = 0.25,
    fpn_channels: Optional[int] = None,
    dropout_rate: Optional[float] = None,
    use_gmp: Optional[bool] = None,
    use_se: Optional[bool] = None,
    variant: Optional[str] = None,
    **kwargs
) -> nn.Module:
    """
    Factory constructor returning either:
    1. Phase7MultiScaleDenseNet121 (default, 8,071,942 parameters, verified Phase 7 model).
    2. MultiScaleDenseNet121FPN (when fpn_channels or FPN ablation flags are explicitly set).
    """
    # If caller specifies FPN parameters or asks for FPN variant
    is_fpn_call = (
        fpn_channels is not None or
        use_gmp is False or
        use_se is False or
        (variant is not None and "fpn" in variant.lower())
    )

    if is_fpn_call:
        eff_fpn_ch = fpn_channels if fpn_channels is not None else 256
        eff_dropout = dropout_rate if dropout_rate is not None else dropout
        eff_gmp = use_gmp if use_gmp is not None else True
        eff_se = use_se if use_se is not None else True
        return MultiScaleDenseNet121FPN(
            num_classes=num_classes,
            fpn_channels=eff_fpn_ch,
            pretrained=pretrained,
            dropout_rate=eff_dropout,
            use_gmp=eff_gmp,
            use_se=eff_se
        )

    # Otherwise return the verified Phase 7 recovery architecture
    eff_dropout = dropout_rate if dropout_rate is not None else dropout
    return Phase7MultiScaleDenseNet121(
        pretrained=pretrained,
        dropout=eff_dropout,
        num_classes=num_classes
    )
