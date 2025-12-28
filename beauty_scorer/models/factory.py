"""
Model factory for creating beauty scoring models.

Provides a unified interface for instantiating models
based on configuration.
"""

from typing import Literal

import torch

from beauty_scorer.config import ModelConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.models.lightweight_cnn import LightweightCPUModel
from beauty_scorer.models.mobilenet_transformer import MobileNetTransformerModel
from beauty_scorer.models.vit_arcface import ViTArcFaceModel
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class ModelFactory:
    """
    Factory for creating beauty scoring models.

    Supports registration of custom model architectures.
    """

    _models: dict[str, type[BaseBeautyModel]] = {
        "mobilenet_transformer": MobileNetTransformerModel,
        "vit_arcface": ViTArcFaceModel,
        "lightweight_cpu": LightweightCPUModel,
    }

    @classmethod
    def register(cls, name: str, model_class: type[BaseBeautyModel]) -> None:
        """
        Register a new model architecture.

        Args:
            name: Architecture name.
            model_class: Model class (must inherit from BaseBeautyModel).
        """
        if not issubclass(model_class, BaseBeautyModel):
            raise ValueError("Model class must inherit from BaseBeautyModel")
        cls._models[name] = model_class
        logger.info(f"Registered model architecture: {name}")

    @classmethod
    def create(
        cls,
        config: ModelConfig | None = None,
        architecture: str | None = None,
        **kwargs,
    ) -> BaseBeautyModel:
        """
        Create a model instance.

        Args:
            config: Model configuration.
            architecture: Override architecture from config.
            **kwargs: Additional arguments passed to model constructor.

        Returns:
            Model instance.
        """
        if config is None:
            config = ModelConfig()

        arch = architecture or config.architecture

        if arch not in cls._models:
            available = list(cls._models.keys())
            raise ValueError(f"Unknown architecture: {arch}. Available: {available}")

        # Update config if architecture was overridden
        if architecture and architecture != config.architecture:
            config = config.model_copy(update={"architecture": architecture})

        model_class = cls._models[arch]
        model = model_class(config, **kwargs)

        logger.info(f"Created model: {arch} with {model.count_parameters():,} params")
        return model

    @classmethod
    def list_available(cls) -> list[str]:
        """List available model architectures."""
        return list(cls._models.keys())

    @classmethod
    def get_model_class(cls, name: str) -> type[BaseBeautyModel]:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        return cls._models[name]


def create_model(
    architecture: Literal[
        "mobilenet_transformer", "vit_arcface", "lightweight_cpu"
    ] = "mobilenet_transformer",
    config: ModelConfig | None = None,
    pretrained_path: str | None = None,
    device: torch.device | None = None,
    compile_model: bool = False,
    **kwargs,
) -> BaseBeautyModel:
    """
    Convenience function to create a model.

    Args:
        architecture: Model architecture name.
        config: Model configuration.
        pretrained_path: Path to pretrained weights.
        device: Device to place model on.
        compile_model: Whether to compile with torch.compile.
        **kwargs: Additional config overrides.

    Returns:
        Model instance.
    """
    # Create or update config
    if config is None:
        config = ModelConfig(architecture=architecture, **kwargs)
    elif kwargs:
        config = config.model_copy(update=kwargs)

    # Create model
    model = ModelFactory.create(config, architecture=architecture)

    # Load pretrained weights
    if pretrained_path:
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"Loaded pretrained weights from {pretrained_path}")

    # Move to device
    if device:
        model = model.to(device)

    # Compile model
    if compile_model:
        model.compile_model()

    return model


def load_model(
    path: str,
    device: torch.device | None = None,
    config_override: ModelConfig | None = None,
) -> BaseBeautyModel:
    """
    Load a saved model.

    Args:
        path: Path to model checkpoint.
        device: Device to load model to.
        config_override: Override saved config.

    Returns:
        Loaded model instance.
    """
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)

    # Get config
    if config_override:
        config = config_override
    elif "config" in checkpoint and checkpoint["config"]:
        config = ModelConfig(**checkpoint["config"])
    else:
        # Try to infer architecture from state dict
        logger.warning("No config found in checkpoint, using default config")
        config = ModelConfig()

    # Create model
    model = ModelFactory.create(config)

    # Load state dict
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if device:
        model = model.to(device)

    logger.info(f"Loaded model from {path}")
    return model


def get_model_summary(model: BaseBeautyModel) -> dict:
    """
    Get a summary of model architecture and parameters.

    Args:
        model: Model instance.

    Returns:
        Dictionary with model summary.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count params by component
    param_counts = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        param_counts[name] = count

    return {
        "architecture": model.config.architecture,
        "backbone": model.config.backbone,
        "embed_dim": model.config.embed_dim,
        "num_classes": model.config.num_classes,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "param_counts_by_component": param_counts,
    }
