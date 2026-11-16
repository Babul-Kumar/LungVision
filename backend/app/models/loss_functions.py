"""
LungVision AI — Advanced Multi-Label Loss Functions.

Implements loss formulations for multi-label chest X-ray classification under
severe class imbalance and sparse positive findings:
1. Binary Cross Entropy with Logits (Standard Baseline).
2. Focal Loss (Lin et al., ICCV 2017) with focusing parameter gamma=2.0 and alpha=0.25.
3. Class-Balanced Loss (Cui et al., CVPR 2019) using effective volume En = (1 - beta^n) / (1 - beta).
4. Asymmetric Loss (Ridnik et al., ICCV 2021) with asymmetric focusing gamma_pos=0, gamma_neg=4, margin=0.05.
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-label Focal Loss with logit inputs.
    Down-weights easy well-classified negative examples, focusing training on hard ambiguous boundaries.
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized logits of shape (B, C).
            targets: Binary ground truth labels of shape (B, C) in {0, 1}.
        """
        probs = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = torch.pow(1.0 - p_t + 1e-8, self.gamma)

        focal_loss = alpha_factor * modulating_factor * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Multi-Label Loss based on the effective number of samples (Cui et al., 2019).
    E_n = (1 - beta^N) / (1 - beta)
    Weights rare pathologies (e.g. Pneumothorax, Pneumonia) inversely proportional to their
    effective sample volume rather than naive inverse frequency, avoiding noise amplification.
    """
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = "focal",
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
    ):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.reduction = reduction
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="none")

        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, np.array(samples_per_class, dtype=np.float64))
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        # Normalize weights so sum equals number of classes
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Shape (B, C)
            targets: Shape (B, C)
        """
        if self.loss_type == "focal":
            base_loss = self.focal_loss(logits, targets)
        else:
            base_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        cb_loss = base_loss * self.weights.unsqueeze(0)

        if self.reduction == "mean":
            return cb_loss.mean()
        elif self.reduction == "sum":
            return cb_loss.sum()
        return cb_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification (Ridnik et al., 2021).
    Decouples focusing on positive vs negative samples:
    Applies asymmetric focusing (gamma_neg > gamma_pos) and probability margin shifting
    to cleanly discard clean, non-pathological background negatives.
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Positive targets
        xs_pos = probs
        # Negative targets with margin shifting
        xs_neg = 1.0 - probs
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Asymmetric focusing
        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))

        if self.gamma_pos > 0:
            loss_pos = loss_pos * torch.pow(1.0 - xs_pos, self.gamma_pos)
        if self.gamma_neg > 0:
            loss_neg = loss_neg * torch.pow(1.0 - xs_neg, self.gamma_neg)

        loss = -1.0 * (loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
