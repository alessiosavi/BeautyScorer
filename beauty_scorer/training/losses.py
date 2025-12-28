"""
Loss functions for beauty score prediction.

Provides various loss functions including:
- Cross entropy with label smoothing
- Focal loss for class imbalance
- Ordinal regression loss for ordered classes
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Label smoothing helps prevent overconfident predictions and
    improves generalization.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        """
        Initialize loss.

        Args:
            smoothing: Label smoothing factor (0 = no smoothing).
            weight: Class weights for imbalanced data.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.

        Args:
            logits: Predicted logits (batch, num_classes).
            targets: Target class indices (batch,).

        Returns:
            Loss value.
        """
        num_classes = logits.size(-1)

        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # Log softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute loss
        loss = -smooth_targets * log_probs

        # Apply class weights
        if self.weight is not None:
            weight = self.weight.to(logits.device)
            loss = loss * weight.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.

    Down-weights well-classified examples and focuses on hard examples.
    Useful when some scores are much more common than others.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        """
        Initialize loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples).
            alpha: Class weights (can be tensor or None).
            reduction: Reduction method.
            label_smoothing: Optional label smoothing.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits (batch, num_classes).
            targets: Target class indices (batch,).

        Returns:
            Loss value.
        """
        num_classes = logits.size(-1)

        # Apply label smoothing to targets
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits)
                smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
        else:
            smooth_targets = F.one_hot(targets, num_classes).float()

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Compute focal weight
        focal_weight = (1 - probs) ** self.gamma

        # Compute loss
        loss = -focal_weight * smooth_targets * log_probs

        # Apply class weights
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = loss * alpha.unsqueeze(0)

        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal regression loss for ordered categories.

    Treats classification as multiple binary classification problems,
    which respects the natural ordering of beauty scores.
    """

    def __init__(
        self,
        num_classes: int = 9,
        reduction: str = "mean",
    ):
        """
        Initialize loss.

        Args:
            num_classes: Number of ordinal classes.
            reduction: Reduction method.
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute ordinal regression loss.

        Args:
            logits: Ordinal logits (batch, num_classes - 1).
            targets: Target class indices (batch,).

        Returns:
            Loss value.
        """
        # Create ordinal targets: for class k, all thresholds < k should be 1
        batch_size = targets.size(0)
        num_thresholds = self.num_classes - 1
        device = logits.device

        ordinal_targets = torch.zeros(batch_size, num_thresholds, device=device)
        for i in range(num_thresholds):
            ordinal_targets[:, i] = (targets > i).float()

        # Binary cross entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, ordinal_targets, reduction="none")
        loss = loss.sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class CrossEntropyLoss(nn.Module):
    """Standard cross entropy loss wrapper with optional class weights."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        """
        Initialize loss.

        Args:
            weight: Class weights.
            label_smoothing: Label smoothing factor.
            reduction: Reduction method.
        """
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss."""
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
            reduction=self.reduction,
        )


def get_loss_function(
    loss_type: Literal["cross_entropy", "focal", "ordinal"] = "cross_entropy",
    num_classes: int = 9,
    label_smoothing: float = 0.1,
    focal_gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Get a loss function by name.

    Args:
        loss_type: Type of loss function.
        num_classes: Number of classes.
        label_smoothing: Label smoothing factor.
        focal_gamma: Gamma for focal loss.
        class_weights: Class weights for imbalanced data.

    Returns:
        Loss function module.
    """
    if loss_type == "cross_entropy":
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(
                smoothing=label_smoothing,
                weight=class_weights,
            )
        return CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    elif loss_type == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=class_weights,
            label_smoothing=label_smoothing,
        )

    elif loss_type == "ordinal":
        return OrdinalRegressionLoss(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
