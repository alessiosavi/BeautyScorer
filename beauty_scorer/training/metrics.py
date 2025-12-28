"""
Metrics for evaluating beauty score predictions.

Provides metrics including accuracy, MAE, RMSE, within-N accuracy,
and confusion matrix computation.
"""

import numpy as np
import torch

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class Accuracy:
    """Accuracy metric tracker."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.correct = 0
        self.total = 0

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """
        Update with batch results.

        Args:
            predictions: Predicted classes.
            targets: True classes.
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.correct += (predictions == targets).sum()
        self.total += len(targets)

    def compute(self) -> float:
        """Compute accuracy."""
        if self.total == 0:
            return 0.0
        return self.correct / self.total


class MAE:
    """Mean Absolute Error metric tracker."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset metric state."""
        self.sum_error = 0.0
        self.total = 0

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
    ) -> None:
        """Update with batch results."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.sum_error += np.abs(predictions - targets).sum()
        self.total += len(targets)

    def compute(self) -> float:
        """Compute MAE."""
        if self.total == 0:
            return 0.0
        return self.sum_error / self.total


class MetricTracker:
    """
    Tracks multiple metrics during training.

    Handles accumulation over batches and epoch-level computation.
    """

    def __init__(self, num_classes: int = 9):
        """
        Initialize tracker.

        Args:
            num_classes: Number of classes for confusion matrix.
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.predictions = []
        self.targets = []
        self.losses = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss: float | None = None,
    ) -> None:
        """
        Update with batch results.

        Args:
            predictions: Predicted classes (batch,).
            targets: True classes (batch,).
            loss: Batch loss value.
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric values.
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        metrics = {}

        # Loss
        if self.losses:
            metrics["loss"] = np.mean(self.losses)

        if len(preds) == 0:
            return metrics

        # Accuracy
        metrics["accuracy"] = (preds == targets).mean()

        # MAE (treating classes as ordinal)
        metrics["mae"] = np.abs(preds - targets).mean()

        # RMSE
        metrics["rmse"] = np.sqrt(((preds - targets) ** 2).mean())

        # Within-N accuracy
        metrics["within_1"] = (np.abs(preds - targets) <= 1).mean()
        metrics["within_2"] = (np.abs(preds - targets) <= 2).mean()

        return metrics

    def get_confusion_matrix(self) -> np.ndarray:
        """
        Get confusion matrix.

        Returns:
            Confusion matrix (num_classes, num_classes).
        """
        preds = np.array(self.predictions)
        targets = np.array(self.targets)

        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for p, t in zip(preds, targets, strict=False):
            if 0 <= p < self.num_classes and 0 <= t < self.num_classes:
                matrix[t, p] += 1

        return matrix

    def get_per_class_accuracy(self) -> dict:
        """
        Get accuracy for each class.

        Returns:
            Dictionary mapping class index to accuracy.
        """
        matrix = self.get_confusion_matrix()
        per_class = {}

        for i in range(self.num_classes):
            total = matrix[i].sum()
            if total > 0:
                per_class[i] = matrix[i, i] / total
            else:
                per_class[i] = 0.0

        return per_class


def compute_metrics(
    predictions: np.ndarray | torch.Tensor | list,
    targets: np.ndarray | torch.Tensor | list,
    num_classes: int = 9,
) -> dict:
    """
    Compute all metrics for predictions vs targets.

    Args:
        predictions: Predicted classes.
        targets: True classes.
        num_classes: Number of classes.

    Returns:
        Dictionary of metric values.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, list):
        predictions = np.array(predictions)

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    elif isinstance(targets, list):
        targets = np.array(targets)

    metrics = {
        "accuracy": (predictions == targets).mean(),
        "mae": np.abs(predictions - targets).mean(),
        "rmse": np.sqrt(((predictions - targets) ** 2).mean()),
        "within_1": (np.abs(predictions - targets) <= 1).mean(),
        "within_2": (np.abs(predictions - targets) <= 2).mean(),
    }

    # Confusion matrix
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, t in zip(predictions, targets, strict=False):
        if 0 <= p < num_classes and 0 <= t < num_classes:
            matrix[int(t), int(p)] += 1
    metrics["confusion_matrix"] = matrix

    return metrics


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """
    Format metrics for logging.

    Args:
        metrics: Dictionary of metric values.
        prefix: Prefix for each metric name.

    Returns:
        Formatted string.
    """
    parts = []
    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        if isinstance(value, float):
            parts.append(f"{prefix}{key}: {value:.4f}")
        else:
            parts.append(f"{prefix}{key}: {value}")
    return " | ".join(parts)
