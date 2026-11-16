"""
LungVision AI — High-Resolution Multi-Scale DenseNet121 Architecture.

Implements MultiScaleDenseNet121-FPN for thoracic radiograph screening:
1. Feature extraction from intermediate DenseNet121 blocks:
   - Level 2 (denseblock2): Stride 8 (e.g. 64x64 at 512x512) for sub-millimeter apical lines & atelectasis.
   - Level 3 (denseblock3): Stride 16 (e.g. 32x32 at 512x512) for parenchymal airspace opacification & consolidations.
   - Level 4 (denseblock4/norm5): Stride 32 (e.g. 16x16 at 512x512) for macroscopic cardiothoracic ratio & vascular redistribution.
2. Feature Pyramid Network (FPN) lateral connections with top-down feature aggregation.
3. Dual pooling (Global Average Pooling + Global Max Pooling) preserving localized focal spikes.
4. Decoupled classification heads for fine-scale, mid-scale, and macroscopic conditions.
5. Strict preservation of 6-label clinical ordering:
   [Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax].
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights


DISEASE_LABELS = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax"
]


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


class MultiScaleDenseNet121(nn.Module):
    """
    Multi-Scale DenseNet121 with Feature Pyramid Network and Decoupled Pathology Heads.
    Preserves fine spatial detail for apical pneumothorax and linear atelectasis while
    maintaining macroscopic thoracic context for cardiomegaly and pulmonary edema.

    Supports controlled ablation flags:
    - use_gmp: Whether to use Global Max Pooling alongside GAP (default True).
               If False (Model E), uses Global Average Pooling only.
    - use_se: Whether to apply Squeeze-and-Excitation channel attention (default True).
              If False (Model F), uses identity shortcut without SE recalibration.
    """
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

        # Base DenseNet121 backbone
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        base_densenet = densenet121(weights=weights)
        features = base_densenet.features

        # Partition DenseNet121 into 3 hierarchical stages:
        # Stage 1 + 2: input -> conv0 -> norm0 -> relu0 -> pool0 -> denseblock1 -> transition1 -> denseblock2
        # Output: [B, 512, H/8, W/8]
        self.stage_c2 = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
            features.denseblock1,
            features.transition1,
            features.denseblock2
        )

        # Stage 3: transition2 -> denseblock3
        # Output: [B, 1024, H/16, W/16]
        self.stage_c3 = nn.Sequential(
            features.transition2,
            features.denseblock3
        )

        # Stage 4: transition3 -> denseblock4 -> norm5
        # Output: [B, 1024, H/32, W/32]
        self.stage_c4 = nn.Sequential(
            features.transition3,
            features.denseblock4,
            features.norm5
        )

        # Lateral 1x1 convolutions for FPN projection
        self.lateral_c4 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(1024, fpn_channels, kernel_size=1)
        self.lateral_c2 = nn.Conv2d(512, fpn_channels, kernel_size=1)

        # Anti-aliasing 3x3 convolutions
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

        # Channel attention on multi-scale pyramid levels (or Identity if ablated)
        if use_se:
            self.att_p2 = ChannelAttention(fpn_channels)
            self.att_p3 = ChannelAttention(fpn_channels)
            self.att_p4 = ChannelAttention(fpn_channels)
        else:
            self.att_p2 = nn.Identity()
            self.att_p3 = nn.Identity()
            self.att_p4 = nn.Identity()

        # Pooling dimension: GAP + GMP -> 2 * fpn_channels = 512, or GAP only -> fpn_channels = 256
        pool_dim = fpn_channels * 2 if use_gmp else fpn_channels
        self.pool_dim = pool_dim

        # Decoupled Pathology Heads:
        # 1. Fine-scale head (from P2): Atelectasis (idx 4) & Pneumothorax (idx 5)
        self.head_fine = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)  # [Atelectasis, Pneumothorax]
        )

        # 2. Mid-scale head (from P3): Consolidation (idx 2) & Pneumonia (idx 3)
        self.head_mid = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)  # [Consolidation, Pneumonia]
        )

        # 3. Macroscopic coarse head (from P4): Cardiomegaly (idx 0) & Edema (idx 1)
        self.head_coarse = nn.Sequential(
            nn.Linear(pool_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 2)  # [Cardiomegaly, Edema]
        )

        # 4. Holistic Global Head (from concatenated P2 + P3 + P4 = 3 * pool_dim)
        self.head_holistic = nn.Sequential(
            nn.Linear(pool_dim * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

        # Blend weights between specialized heads and holistic head
        self.specialized_weight = nn.Parameter(torch.tensor(0.6))
        self.holistic_weight = nn.Parameter(torch.tensor(0.4))

    def _pool_features(self, p: torch.Tensor) -> torch.Tensor:
        """Applies pooling: dual GAP + GMP if use_gmp is True; GAP only if False."""
        gap = F.adaptive_avg_pool2d(p, (1, 1)).flatten(1)
        if self.use_gmp:
            gmp = F.adaptive_max_pool2d(p, (1, 1)).flatten(1)
            return torch.cat([gap, gmp], dim=1)
        return gap

    def extract_multiscale_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts multi-scale pyramid feature maps (P2, P3, P4).
        Used for high-resolution visual explainability and feature inspection.
        """
        c2 = self.stage_c2(x)
        c3 = self.stage_c3(c2)
        c4 = self.stage_c4(c3)

        # Top-down FPN aggregation
        m4 = self.lateral_c4(c4)
        m3 = self.lateral_c3(c3) + F.interpolate(m4, size=c3.shape[2:], mode="nearest")
        m2 = self.lateral_c2(c2) + F.interpolate(m3, size=c2.shape[2:], mode="nearest")

        p4 = self.att_p4(self.smooth_p4(m4))
        p3 = self.att_p3(self.smooth_p3(m3))
        p2 = self.att_p2(self.smooth_p2(m2))

        return p2, p3, p4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing 6 disease logits in standard clinical order:
        [Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax].
        """
        p2, p3, p4 = self.extract_multiscale_features(x)

        # Dual pooling per pyramid level
        feat_p2 = self._pool_features(p2)
        feat_p3 = self._pool_features(p3)
        feat_p4 = self._pool_features(p4)

        # Specialized head predictions
        out_coarse = self.head_coarse(feat_p4)  # [Cardiomegaly, Edema]
        out_mid = self.head_mid(feat_p3)        # [Consolidation, Pneumonia]
        out_fine = self.head_fine(feat_p2)      # [Atelectasis, Pneumothorax]

        # Assemble specialized 6-class logits:
        # [Cardiomegaly, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax]
        out_specialized = torch.cat([out_coarse, out_mid, out_fine], dim=1)

        # Holistic multi-scale fusion prediction
        feat_fused = torch.cat([feat_p2, feat_p3, feat_p4], dim=1)
        out_holistic = self.head_holistic(feat_fused)

        # Blended combination
        w_spec = torch.sigmoid(self.specialized_weight)
        w_holi = torch.sigmoid(self.holistic_weight)
        w_sum = w_spec + w_holi
        w_spec = w_spec / w_sum
        w_holi = w_holi / w_sum

        logits = (w_spec * out_specialized) + (w_holi * out_holistic)
        return logits

    def load_baseline_features(self, checkpoint_path: str) -> None:
        """
        Initializes backbone feature weights from the frozen NIH DenseNet121 checkpoint.
        Leaves new FPN and decoupled heads initialized for fine-tuning.
        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        temp_densenet = densenet121(num_classes=6)
        # Load only features keys
        features_only = {k: v for k, v in state_dict.items() if k.startswith("features.")}
        temp_densenet.features.load_state_dict(features_only, strict=False)

        features = temp_densenet.features
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
