"""
Training callbacks for monitoring and control.

Provides callbacks for:
- Early stopping
- Model checkpointing
- Progress bars
- Learning rate monitoring
"""

from abc import ABC
from pathlib import Path

import torch
from tqdm.auto import tqdm

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, trainer: "Trainer", batch: int) -> None:
        """Called at the start of each batch."""
        pass

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None:
        """Called at the end of each batch."""
        pass


class CallbackList:
    """Container for multiple callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        """Initialize callback list."""
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: "Trainer") -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer, epoch)

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer, epoch, logs)

    def on_batch_begin(self, trainer: "Trainer", batch: int) -> None:
        for cb in self.callbacks:
            cb.on_batch_begin(trainer, batch)

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(trainer, batch, logs)


class EarlyStopping(Callback):
    """
    Early stopping callback.

    Stops training when a monitored metric stops improving.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best: bool = True,
    ):
        """
        Initialize early stopping.

        Args:
            monitor: Metric to monitor.
            patience: Number of epochs with no improvement to wait.
            min_delta: Minimum change to qualify as improvement.
            mode: 'min' or 'max' for the monitored metric.
            restore_best: Whether to restore best weights when stopping.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Reset state at training start."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch = 0
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Check for improvement."""
        current = logs.get(self.monitor)
        if current is None:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found in logs")
            return

        if self._is_improvement(current):
            self.best_value = current
            self.best_epoch = epoch
            self.wait = 0
            if self.restore_best:
                self.best_weights = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                trainer.stop_training = True
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch}"
                )

    def on_train_end(self, trainer: "Trainer") -> None:
        """Restore best weights if needed."""
        if self.restore_best and self.best_weights is not None:
            trainer.model.load_state_dict(self.best_weights)
            logger.info(f"Restored best model from epoch {self.best_epoch}")

    def _is_improvement(self, current: float) -> bool:
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < self.best_value - self.min_delta
        return current > self.best_value + self.min_delta


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback.

    Saves model checkpoints during training.
    """

    def __init__(
        self,
        save_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        filename: str = "checkpoint_epoch{epoch:02d}_{val_loss:.4f}.pt",
    ):
        """
        Initialize checkpointing.

        Args:
            save_dir: Directory to save checkpoints.
            monitor: Metric to monitor for best model.
            mode: 'min' or 'max' for the monitored metric.
            save_best_only: Only save when metric improves.
            save_last: Always save the last checkpoint.
            filename: Checkpoint filename template.
        """
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.filename = filename

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Create save directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = float("inf") if self.mode == "min" else float("-inf")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Save checkpoint if conditions are met."""
        current = logs.get(self.monitor)

        # Format filename
        filepath = self.save_dir / self.filename.format(
            epoch=epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
        )

        # Check if we should save
        save = False
        if current is not None:
            is_better = (
                current < self.best_value - 1e-6
                if self.mode == "min"
                else current > self.best_value + 1e-6
            )
            if is_better:
                self.best_value = current
                self.best_path = self.save_dir / "best.pt"
                save = True

        if not self.save_best_only or save:
            self._save_checkpoint(trainer, filepath, logs)

        # Save best model separately
        if save and self.best_path:
            self._save_checkpoint(trainer, self.best_path, logs)
            logger.info(f"Saved best model: {self.monitor}={current:.4f}")

    def on_train_end(self, trainer: "Trainer") -> None:
        """Save final checkpoint."""
        if self.save_last:
            filepath = self.save_dir / "last.pt"
            self._save_checkpoint(trainer, filepath, {})
            logger.info(f"Saved last model to {filepath}")

    def _save_checkpoint(
        self,
        trainer: "Trainer",
        filepath: Path,
        logs: dict,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "epoch": trainer.current_epoch,
            "metrics": logs,
        }

        if hasattr(trainer.model, "config") and trainer.model.config:
            checkpoint["config"] = trainer.model.config.model_dump()

        if trainer.scheduler is not None:
            checkpoint["scheduler"] = trainer.scheduler.state_dict()

        torch.save(checkpoint, filepath)


class ProgressBar(Callback):
    """
    Progress bar callback using tqdm.

    Shows training progress with metrics.
    """

    def __init__(self, show_batch_progress: bool = True):
        """
        Initialize progress bar.

        Args:
            show_batch_progress: Show progress for each batch.
        """
        self.show_batch_progress = show_batch_progress
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, trainer: "Trainer") -> None:
        """Create epoch progress bar."""
        self.epoch_pbar = tqdm(
            total=trainer.config.epochs,
            desc="Training",
            position=0,
        )

    def on_train_end(self, trainer: "Trainer") -> None:
        """Close progress bars."""
        if self.epoch_pbar:
            self.epoch_pbar.close()
        if self.batch_pbar:
            self.batch_pbar.close()

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        """Create batch progress bar."""
        if self.show_batch_progress and trainer.train_loader:
            self.batch_pbar = tqdm(
                total=len(trainer.train_loader),
                desc=f"Epoch {epoch + 1}",
                position=1,
                leave=False,
            )

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Update epoch progress."""
        if self.batch_pbar:
            self.batch_pbar.close()

        if self.epoch_pbar:
            # Format metrics for display
            metrics_str = " | ".join(
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in logs.items()
                if not k.startswith("_") and k != "confusion_matrix"
            )
            self.epoch_pbar.set_postfix_str(metrics_str)
            self.epoch_pbar.update(1)

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None:
        """Update batch progress."""
        if self.batch_pbar:
            loss = logs.get("loss", 0)
            self.batch_pbar.set_postfix(loss=f"{loss:.4f}")
            self.batch_pbar.update(1)


class LRMonitor(Callback):
    """
    Learning rate monitoring callback.

    Logs learning rate changes during training.
    """

    def __init__(self, log_every_n_steps: int = 100):
        """
        Initialize LR monitor.

        Args:
            log_every_n_steps: Log frequency.
        """
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0

    def on_batch_end(self, trainer: "Trainer", batch: int, logs: dict) -> None:
        """Log learning rate."""
        self.step_count += 1
        if self.step_count % self.log_every_n_steps == 0:
            lrs = [g["lr"] for g in trainer.optimizer.param_groups]
            logger.debug(f"Step {self.step_count}: LRs = {lrs}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict) -> None:
        """Log learning rate at epoch end."""
        lrs = [g["lr"] for g in trainer.optimizer.param_groups]
        logs["learning_rate"] = lrs[0]  # Main LR
