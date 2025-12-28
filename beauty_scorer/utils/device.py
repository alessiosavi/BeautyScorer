"""
Device detection and management utilities.

Provides automatic device detection, seed setting for reproducibility,
and device information utilities.
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """Information about the current device."""

    device: torch.device
    device_type: str  # "cuda", "mps", "cpu"
    device_name: str
    memory_total: int | None = None  # In bytes
    memory_available: int | None = None  # In bytes
    cuda_version: str | None = None
    cudnn_version: int | None = None


def get_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate device for computation.

    Args:
        device: Device specification. Options:
            - None or "auto": Automatically detect best device
            - "cuda": Use CUDA GPU
            - "cuda:0", "cuda:1", etc.: Use specific GPU
            - "mps": Use Apple Silicon GPU
            - "cpu": Use CPU

    Returns:
        torch.device instance.
    """
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch_device = torch.device(device)
    logger.info(f"Using device: {torch_device}")

    return torch_device


def get_device_info(device: torch.device | None = None) -> DeviceInfo:
    """
    Get detailed information about the device.

    Args:
        device: Device to get info for. If None, uses auto-detected device.

    Returns:
        DeviceInfo dataclass with device details.
    """
    if device is None:
        device = get_device()

    device_type = device.type
    device_name = str(device)

    info = DeviceInfo(
        device=device,
        device_type=device_type,
        device_name=device_name,
    )

    if device_type == "cuda":
        idx = device.index if device.index is not None else 0
        info.device_name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info.memory_total = props.total_memory
        info.memory_available = torch.cuda.mem_get_info(idx)[0]
        info.cuda_version = torch.version.cuda
        info.cudnn_version = torch.backends.cudnn.version()

    elif device_type == "mps":
        info.device_name = "Apple Silicon GPU"

    elif device_type == "cpu":
        info.device_name = "CPU"

    return info


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
        deterministic: If True, use deterministic algorithms (may be slower).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 2.0+ deterministic settings
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Random seed set to {seed}, deterministic={deterministic}")


class DeviceManager:
    """
    Context manager for device operations.

    Handles automatic device detection, memory management, and
    provides utilities for moving data to/from devices.
    """

    def __init__(
        self,
        device: str | None = None,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        """
        Initialize device manager.

        Args:
            device: Device specification (see get_device for options).
            seed: Random seed for reproducibility.
            deterministic: Use deterministic algorithms.
        """
        self.device = get_device(device)
        self.info = get_device_info(self.device)

        if seed is not None:
            set_seed(seed, deterministic)

        self._log_device_info()

    def _log_device_info(self) -> None:
        """Log device information."""
        logger.info(f"Device: {self.info.device_name}")

        if self.info.memory_total is not None:
            total_gb = self.info.memory_total / (1024**3)
            avail_gb = self.info.memory_available / (1024**3) if self.info.memory_available else 0
            logger.info(f"Memory: {avail_gb:.1f} GB available / {total_gb:.1f} GB total")

        if self.info.cuda_version is not None:
            logger.info(f"CUDA: {self.info.cuda_version}, cuDNN: {self.info.cudnn_version}")

    def to_device(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Move tensors to the managed device.

        Args:
            *tensors: Tensors to move.

        Returns:
            Tuple of tensors on the device.
        """
        return tuple(t.to(self.device) if torch.is_tensor(t) else t for t in tensors)

    def empty_cache(self) -> None:
        """Clear device memory cache."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def memory_stats(self) -> dict:
        """
        Get current memory statistics.

        Returns:
            Dictionary with memory stats (allocated, reserved, free).
        """
        stats = {}

        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else 0
            stats["allocated"] = torch.cuda.memory_allocated(idx)
            stats["reserved"] = torch.cuda.memory_reserved(idx)
            stats["free"] = torch.cuda.mem_get_info(idx)[0]

        return stats

    def synchronize(self) -> None:
        """Synchronize device operations."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elif self.device.type == "mps":
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        """Check if using MPS device."""
        return self.device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return self.device.type == "cpu"


def get_optimal_num_workers() -> int:
    """
    Get optimal number of DataLoader workers.

    Returns:
        Recommended number of workers based on CPU count and device.
    """
    cpu_count = os.cpu_count() or 4
    # Use fewer workers on CPU to avoid overhead
    return min(cpu_count, 8)
