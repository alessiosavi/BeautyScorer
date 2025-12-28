"""
Optimizer and scheduler factories.

Provides functions for creating optimizers with parameter groups
and learning rate schedulers.
"""

from typing import Literal

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from beauty_scorer.config import TrainingConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


def create_optimizer(
    model: BaseBeautyModel,
    config: TrainingConfig | None = None,
    optimizer_type: Literal["adamw", "sgd"] = "adamw",
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    backbone_lr_multiplier: float | None = None,
) -> torch.optim.Optimizer:
    """
    Create optimizer with parameter groups.

    Creates separate parameter groups for backbone and other layers,
    allowing different learning rates.

    Args:
        model: Model to optimize.
        config: Training configuration.
        optimizer_type: Type of optimizer.
        learning_rate: Override learning rate from config.
        weight_decay: Override weight decay from config.
        backbone_lr_multiplier: Override backbone LR multiplier.

    Returns:
        Configured optimizer.
    """
    if config is None:
        config = TrainingConfig()

    lr = learning_rate or config.learning_rate
    wd = weight_decay or config.weight_decay
    backbone_mult = backbone_lr_multiplier or config.backbone_lr_multiplier

    # Get parameter groups from model
    param_groups = model.get_trainable_params()

    # Build optimizer param groups
    optimizer_groups = []
    for group in param_groups:
        lr_scale = group.get("lr_scale", 1.0)
        group_lr = lr * lr_scale

        # Apply backbone multiplier
        if group.get("name") == "backbone":
            group_lr = lr * backbone_mult

        optimizer_groups.append(
            {
                "params": group["params"],
                "lr": group_lr,
                "weight_decay": wd,
            }
        )

        logger.debug(f"Param group '{group.get('name', 'unnamed')}': lr={group_lr:.6f}")

    # Create optimizer
    if optimizer_type == "adamw":
        optimizer = AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-8)
    elif optimizer_type == "sgd":
        optimizer = SGD(optimizer_groups, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger.info(f"Created {optimizer_type} optimizer with {len(optimizer_groups)} param groups")
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig | None = None,
    scheduler_type: str | None = None,
    num_epochs: int | None = None,
    steps_per_epoch: int | None = None,
    warmup_epochs: int = 0,
) -> LRScheduler | ReduceLROnPlateau | None:
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule.
        config: Training configuration.
        scheduler_type: Override scheduler type from config.
        num_epochs: Total number of epochs.
        steps_per_epoch: Number of steps per epoch (for OneCycleLR).
        warmup_epochs: Number of warmup epochs.

    Returns:
        Configured scheduler or None.
    """
    if config is None:
        config = TrainingConfig()

    sched_type = scheduler_type or config.scheduler
    epochs = num_epochs or config.epochs
    warmup = warmup_epochs or config.scheduler_warmup_epochs

    if sched_type == "none":
        return None

    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup,
            eta_min=1e-7,
        )

    elif sched_type == "step":
        scheduler = StepLR(
            optimizer,
            step_size=epochs // 3,
            gamma=0.1,
        )

    elif sched_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        )

    elif sched_type == "onecycle" and steps_per_epoch:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[g["lr"] * 10 for g in optimizer.param_groups],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
        )

    else:
        logger.warning(f"Unknown scheduler type: {sched_type}, using cosine")
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Wrap with warmup if needed
    if warmup > 0:
        scheduler = WarmupScheduler(optimizer, scheduler, warmup_epochs=warmup)

    logger.info(f"Created {sched_type} scheduler with warmup={warmup}")
    return scheduler


class WarmupScheduler:
    """
    Learning rate scheduler wrapper with linear warmup.

    Linearly increases learning rate during warmup epochs,
    then follows the base scheduler.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: LRScheduler,
        warmup_epochs: int = 2,
    ):
        """
        Initialize warmup scheduler.

        Args:
            optimizer: Optimizer.
            scheduler: Base scheduler to use after warmup.
            warmup_epochs: Number of warmup epochs.
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        # Store initial learning rates
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int | None = None, metrics: float | None = None) -> None:
        """
        Update learning rate.

        Args:
            epoch: Current epoch (optional).
            metrics: Metrics value for ReduceLROnPlateau.
        """
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1

        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self.base_lrs[i] * warmup_factor
        else:
            # Use base scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau) and metrics is not None:
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()

    def get_last_lr(self) -> list[float]:
        """Get current learning rates."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Get scheduler state."""
        return {
            "current_epoch": self.current_epoch,
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scheduler state."""
        self.current_epoch = state_dict["current_epoch"]
        self.scheduler.load_state_dict(state_dict["scheduler"])
