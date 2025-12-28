"""Training module for BeautyScorer."""

from beauty_scorer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    LRMonitor,
    ModelCheckpoint,
    ProgressBar,
)
from beauty_scorer.training.losses import (
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    OrdinalRegressionLoss,
    get_loss_function,
)
from beauty_scorer.training.metrics import MAE, Accuracy, MetricTracker, compute_metrics
from beauty_scorer.training.optimizers import create_optimizer, create_scheduler
from beauty_scorer.training.trainer import Trainer

__all__ = [
    # Losses
    "CrossEntropyLoss",
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "OrdinalRegressionLoss",
    "get_loss_function",
    # Optimizers
    "create_optimizer",
    "create_scheduler",
    # Metrics
    "compute_metrics",
    "Accuracy",
    "MAE",
    "MetricTracker",
    # Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "ModelCheckpoint",
    "ProgressBar",
    "LRMonitor",
    # Trainer
    "Trainer",
]
