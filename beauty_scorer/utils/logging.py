"""
Structured logging utilities for BeautyScorer.

Provides consistent logging across the application with support for
file and console output, log levels, and structured formatting.
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path


class LogLevel(str, Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for console."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.name = f"\033[34m{record.name}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str | LogLevel = LogLevel.INFO,
    log_dir: str | Path | None = None,
    log_file: str | None = None,
    use_colors: bool = True,
) -> logging.Logger:
    """
    Set up logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. If None, only console logging.
        log_file: Name of log file. If None, auto-generated with timestamp.
        use_colors: Whether to use colored output for console.

    Returns:
        Root logger instance.
    """
    if isinstance(log_level, LogLevel):
        log_level = log_level.value

    root_logger = logging.getLogger("beauty_scorer")
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if use_colors:
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
    else:
        console_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        console_formatter = logging.Formatter(console_format, datefmt="%H:%M:%S")

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"beauty_scorer_{timestamp}.log"

        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)

        file_format = (
            "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
        )
        file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)
        root_logger.info(f"Logging to file: {file_path}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name, typically __name__.

    Returns:
        Logger instance.
    """
    # Ensure it's under the beauty_scorer namespace
    if not name.startswith("beauty_scorer"):
        name = f"beauty_scorer.{name}"
    return logging.getLogger(name)


class TrainingLogger:
    """
    Logger for training metrics with optional TensorBoard and W&B support.

    This class provides a unified interface for logging training metrics
    to multiple backends.
    """

    def __init__(
        self,
        log_dir: str | Path,
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        config: dict | None = None,
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Directory for TensorBoard logs.
            use_tensorboard: Enable TensorBoard logging.
            use_wandb: Enable Weights & Biases logging.
            wandb_project: W&B project name.
            wandb_entity: W&B entity/team name.
            config: Configuration dict to log with W&B.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        self.tb_writer = None
        self.wandb_run = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.logger.info(f"TensorBoard logging enabled: {self.log_dir / 'tensorboard'}")
            except ImportError:
                self.logger.warning(
                    "TensorBoard not installed. Install with: pip install tensorboard"
                )

        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=wandb_project or "beauty-scorer",
                    entity=wandb_entity,
                    config=config,
                    dir=str(self.log_dir),
                )
                self.logger.info(f"W&B logging enabled: {wandb.run.url}")
            except ImportError:
                self.logger.warning("W&B not installed. Install with: pip install wandb")

    def log_metrics(self, metrics: dict, step: int, prefix: str = "") -> None:
        """
        Log metrics to all enabled backends.

        Args:
            metrics: Dictionary of metric names and values.
            step: Training step or epoch.
            prefix: Prefix for metric names (e.g., "train/", "val/").
        """
        for name, value in metrics.items():
            full_name = f"{prefix}{name}" if prefix else name

            # TensorBoard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(full_name, value, step)

            # W&B
            if self.wandb_run is not None:
                import wandb

                wandb.log({full_name: value}, step=step)

    def log_image(self, name: str, image, step: int) -> None:
        """Log an image to TensorBoard/W&B."""
        if self.tb_writer is not None:
            self.tb_writer.add_image(name, image, step)

        if self.wandb_run is not None:
            import wandb

            wandb.log({name: wandb.Image(image)}, step=step)

    def log_histogram(self, name: str, values, step: int) -> None:
        """Log a histogram to TensorBoard/W&B."""
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(name, values, step)

        if self.wandb_run is not None:
            import wandb

            wandb.log({name: wandb.Histogram(values)}, step=step)

    def close(self) -> None:
        """Close all logging backends."""
        if self.tb_writer is not None:
            self.tb_writer.close()

        if self.wandb_run is not None:
            import wandb

            wandb.finish()
