"""
Training loop implementation.

Provides a complete training system with:
- Mixed precision training
- Gradient accumulation
- Callbacks support
- Metric tracking
"""

from pathlib import Path

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from beauty_scorer.config import BeautyConfig, TrainingConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.training.callbacks import (
    Callback,
    CallbackList,
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
)
from beauty_scorer.training.losses import get_loss_function
from beauty_scorer.training.metrics import MetricTracker, format_metrics
from beauty_scorer.training.optimizers import create_optimizer, create_scheduler
from beauty_scorer.utils.device import DeviceManager
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training loop for beauty scoring models.

    Features:
    - Automatic mixed precision (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Callback system for customization
    - Metric tracking and logging
    """

    def __init__(
        self,
        model: BaseBeautyModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        config: TrainingConfig | BeautyConfig | None = None,
        device: torch.device | None = None,
        callbacks: list[Callback] | None = None,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            config: Training configuration.
            device: Device to train on.
            callbacks: List of callbacks.
            class_weights: Class weights for loss function.
        """
        # Handle config types
        if isinstance(config, BeautyConfig):
            self.full_config = config
            self.config = config.training
        else:
            self.full_config = None
            self.config = config or TrainingConfig()

        # Device setup
        if device is None:
            self.device_manager = DeviceManager(seed=self.config.seed)
            self.device = self.device_manager.device
        else:
            self.device = device
            self.device_manager = None

        # Model
        self.model = model.to(self.device)

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function
        model_config = model.config if hasattr(model, "config") else None
        self.criterion = get_loss_function(
            loss_type=self.config.loss_function,
            num_classes=model_config.num_classes if model_config else 9,
            label_smoothing=self.config.label_smoothing,
            focal_gamma=self.config.focal_gamma,
            class_weights=class_weights,
        )

        # Optimizer and scheduler
        self.optimizer = create_optimizer(model, self.config)
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config,
            num_epochs=self.config.epochs,
            steps_per_epoch=len(train_loader),
            warmup_epochs=self.config.scheduler_warmup_epochs,
        )

        # AMP with improved NaN handling
        self.use_amp = self.config.use_amp and self.device.type == "cuda"
        # GradScaler automatically handles Inf/NaN gradients by skipping optimizer step
        self.scaler = GradScaler(enabled=self.use_amp) if self.use_amp else None

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # State
        self.current_epoch = 0
        self.global_step = 0
        self.stop_training = False

        # Metrics
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"amp={self.use_amp}, "
            f"grad_accum={self.config.gradient_accumulation_steps}"
        )

    def train_epoch(self) -> dict:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        self.train_metrics.reset()

        accumulation_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            self.callbacks.on_batch_begin(self, batch_idx)

            # Move to device
            photos = batch["photos_tensor"].to(self.device)
            photos_mask = batch["photos_mask"].to(self.device)
            faces = batch["faces_tensor"].to(self.device)
            faces_mask = batch["faces_mask"].to(self.device)
            targets = batch["scores"].to(self.device) - 1  # 0-indexed

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(photos, photos_mask, faces, faces_mask)
                loss = self.criterion(logits, targets)
                loss = loss / accumulation_steps

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Update metrics
            predictions = logits.argmax(dim=-1)
            self.train_metrics.update(
                predictions,
                targets,
                loss.item() * accumulation_steps,
            )

            batch_logs = {"loss": loss.item() * accumulation_steps}
            self.callbacks.on_batch_end(self, batch_idx, batch_logs)

        metrics = self.train_metrics.compute()
        return {f"train_{k}": v for k, v in metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        for batch in self.val_loader:
            # Move to device
            photos = batch["photos_tensor"].to(self.device)
            photos_mask = batch["photos_mask"].to(self.device)
            faces = batch["faces_tensor"].to(self.device)
            faces_mask = batch["faces_mask"].to(self.device)
            targets = batch["scores"].to(self.device) - 1  # 0-indexed

            # Forward pass
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(photos, photos_mask, faces, faces_mask)
                loss = self.criterion(logits, targets)

            # Update metrics
            predictions = logits.argmax(dim=-1)
            self.val_metrics.update(predictions, targets, loss.item())

        metrics = self.val_metrics.compute()
        return {f"val_{k}": v for k, v in metrics.items()}

    def fit(
        self,
        epochs: int | None = None,
        resume_from: str | None = None,
    ) -> dict:
        """
        Run full training loop.

        Args:
            epochs: Number of epochs (overrides config).
            resume_from: Path to checkpoint to resume from.

        Returns:
            Dictionary of final metrics and history.
        """
        num_epochs = epochs or self.config.epochs

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

        # Training callbacks
        self.callbacks.on_train_begin(self)

        try:
            for epoch in range(self.current_epoch, num_epochs):
                if self.stop_training:
                    break

                self.current_epoch = epoch
                self.callbacks.on_epoch_begin(self, epoch)

                # Train and validate
                train_metrics = self.train_epoch()
                val_metrics = self.validate()

                # Update scheduler
                if self.scheduler is not None:
                    if hasattr(self.scheduler, "step"):
                        val_loss = val_metrics.get("val_loss")
                        if hasattr(self.scheduler, "mode"):  # ReduceLROnPlateau
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                # Combine metrics
                epoch_logs = {**train_metrics, **val_metrics}
                epoch_logs["epoch"] = epoch

                # Update history
                self.history["train_loss"].append(train_metrics.get("train_loss", 0))
                self.history["train_acc"].append(train_metrics.get("train_accuracy", 0))
                if val_metrics:
                    self.history["val_loss"].append(val_metrics.get("val_loss", 0))
                    self.history["val_acc"].append(val_metrics.get("val_accuracy", 0))

                # Log metrics
                logger.info(f"Epoch {epoch + 1}/{num_epochs}: {format_metrics(epoch_logs)}")

                self.callbacks.on_epoch_end(self, epoch, epoch_logs)

        finally:
            self.callbacks.on_train_end(self)

        return {
            "final_metrics": epoch_logs,
            "history": self.history,
            "best_val_loss": min(self.history["val_loss"]) if self.history["val_loss"] else None,
        }

    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.current_epoch = checkpoint.get("epoch", 0) + 1

        if "scheduler" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        logger.info(f"Resumed from checkpoint: {path}, epoch {self.current_epoch}")

    def save_checkpoint(self, path: str, metrics: dict | None = None) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": metrics or {},
            "history": self.history,
        }

        if self.scheduler:
            checkpoint["scheduler"] = self.scheduler.state_dict()

        if hasattr(self.model, "config") and self.model.config:
            checkpoint["config"] = self.model.config.model_dump()

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def train_model(
    model: BaseBeautyModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: BeautyConfig | TrainingConfig | None = None,
    output_dir: str = "outputs",
    class_weights: torch.Tensor | None = None,
) -> dict:
    """
    Convenience function to train a model.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        output_dir: Output directory for checkpoints.
        class_weights: Class weights for loss function.

    Returns:
        Training results dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get training config
    if isinstance(config, BeautyConfig):
        train_config = config.training
        patience = train_config.patience
    elif config is not None:
        train_config = config
        patience = config.patience
    else:
        train_config = TrainingConfig()
        patience = train_config.patience

    # Setup callbacks
    callbacks = [
        ProgressBar(),
        EarlyStopping(monitor="val_loss", patience=patience),
        ModelCheckpoint(
            save_dir=output_path,
            monitor="val_loss",
            save_best_only=True,
            save_last=True,
        ),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    # Train
    results = trainer.fit()

    return results
