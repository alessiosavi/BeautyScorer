"""
Configuration management using Pydantic for type-safe settings.

Supports loading from YAML files and environment variables.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    architecture: Literal["mobilenet_transformer", "vit_arcface", "lightweight_cpu"] = Field(
        default="mobilenet_transformer",
        description="Model architecture to use",
    )
    backbone: str = Field(
        default="mobilenet_v3_large",
        description="Backbone network for feature extraction",
    )
    embed_dim: int = Field(default=512, ge=64, le=2048, description="Embedding dimension")
    num_encoder_layers: int = Field(
        default=4, ge=1, le=12, description="Number of transformer encoder layers"
    )
    num_heads: int = Field(default=8, ge=1, le=16, description="Number of attention heads")
    ff_dim: int = Field(default=2048, ge=256, le=8192, description="Feed-forward dimension")
    dropout: float = Field(default=0.2, ge=0.0, le=0.8, description="Dropout rate")
    num_classes: int = Field(default=9, ge=2, description="Number of output classes (scores)")
    unfreeze_backbone_layers: int = Field(
        default=0,
        ge=0,
        description="Number of backbone layers to unfreeze from the end",
    )
    use_positional_encoding: bool = Field(
        default=False,
        description="Whether to use positional encoding in transformers",
    )
    use_cross_attention: bool = Field(
        default=True,
        description="Whether to use cross-attention between photo and face streams",
    )
    use_se_attention: bool = Field(
        default=True,
        description="Whether to use squeeze-and-excitation attention in head",
    )
    use_gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing for memory efficiency",
    )
    pretrained: bool = Field(default=True, description="Use pretrained backbone weights")


class TrainingConfig(BaseModel):
    """Configuration for training process."""

    batch_size: int = Field(default=32, ge=1, le=256, description="Training batch size")
    learning_rate: float = Field(default=1e-4, gt=0, description="Initial learning rate")
    backbone_lr_multiplier: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning rate multiplier for backbone layers",
    )
    weight_decay: float = Field(default=1e-2, ge=0, description="Weight decay for optimizer")
    epochs: int = Field(default=50, ge=1, description="Number of training epochs")
    patience: int = Field(default=5, ge=1, description="Early stopping patience")
    label_smoothing: float = Field(default=0.1, ge=0.0, le=0.5, description="Label smoothing")
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    gradient_accumulation_steps: int = Field(
        default=1, ge=1, description="Gradient accumulation steps"
    )
    max_grad_norm: float = Field(
        default=1.0, gt=0, description="Maximum gradient norm for clipping"
    )
    loss_function: Literal["cross_entropy", "focal", "ordinal"] = Field(
        default="cross_entropy",
        description="Loss function to use",
    )
    focal_gamma: float = Field(default=2.0, ge=0, description="Gamma for focal loss")
    scheduler: Literal["cosine", "step", "plateau", "none"] = Field(
        default="cosine",
        description="Learning rate scheduler",
    )
    scheduler_warmup_epochs: int = Field(
        default=0, ge=0, description="Number of warmup epochs for scheduler"
    )
    val_split: float = Field(default=0.1, ge=0.0, le=0.5, description="Validation split ratio")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for data loading")


class DataConfig(BaseModel):
    """Configuration for data processing."""

    image_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Input image size (height, width)",
    )
    face_size: tuple[int, int] = Field(
        default=(224, 224),
        description="Face crop size (height, width)",
    )
    max_photos: int = Field(default=9, ge=1, le=20, description="Maximum photos per person")
    augmentation: bool = Field(default=True, description="Enable data augmentation")
    augmentation_strength: Literal["light", "medium", "heavy"] = Field(
        default="medium",
        description="Augmentation strength level",
    )
    face_detector: Literal["yolov8", "retinaface", "mtcnn", "opencv"] = Field(
        default="yolov8",
        description="Face detection backend",
    )
    face_expand_percentage: int = Field(
        default=50, ge=0, le=100, description="Face crop expansion percentage"
    )
    face_confidence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum face detection confidence"
    )
    normalize_mean: tuple[float, float, float] = Field(
        default=(0.485, 0.456, 0.406),
        description="Normalization mean (ImageNet default)",
    )
    normalize_std: tuple[float, float, float] = Field(
        default=(0.229, 0.224, 0.225),
        description="Normalization std (ImageNet default)",
    )
    # Sampling configuration
    sample_size: float | int | None = Field(
        default=None,
        description=(
            "Dataset sample size. Float (0-1) = fraction of data, "
            "int >= 1 = absolute count, None = use all data"
        ),
    )
    balance_classes: bool = Field(
        default=False,
        description="Balance class distribution when sampling",
    )
    balance_strategy: Literal["undersample", "sqrt", "proportional"] = Field(
        default="undersample",
        description=(
            "Strategy for balancing: 'undersample' caps each class equally, "
            "'sqrt' uses square root weighting, 'proportional' maintains ratios"
        ),
    )

    @field_validator("image_size", "face_size", mode="before")
    @classmethod
    def convert_to_tuple(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("normalize_mean", "normalize_std", mode="before")
    @classmethod
    def convert_normalize_to_tuple(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v


class PathConfig(BaseModel):
    """Configuration for file paths."""

    dataset_file: str = Field(default="datasets/beauty_dataset.csv", description="Dataset CSV file")
    dataset_base_path: str = Field(default="datasets/images", description="Base path for images")
    output_dir: str = Field(default="outputs", description="Output directory for checkpoints")
    checkpoint_path: str | None = Field(
        default=None,
        description="Path to checkpoint for resuming training",
    )
    export_dir: str = Field(default="exports", description="Directory for exported models")


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    log_dir: str = Field(default="logs", description="Directory for log files")
    use_tensorboard: bool = Field(default=False, description="Enable TensorBoard logging")
    use_wandb: bool = Field(default=False, description="Enable Weights & Biases logging")
    wandb_project: str = Field(default="beauty-scorer", description="W&B project name")
    wandb_entity: str | None = Field(default=None, description="W&B entity/team name")
    log_every_n_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    save_every_n_epochs: int = Field(default=1, ge=1, description="Save checkpoint every N epochs")


class BeautyConfig(BaseModel):
    """Main configuration combining all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    paths: PathConfig = Field(default_factory=PathConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BeautyConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def merge_with(self, overrides: dict) -> "BeautyConfig":
        """Create a new config with overrides applied."""
        current = self.model_dump()
        self._deep_update(current, overrides)
        return BeautyConfig(**current)

    @staticmethod
    def _deep_update(base: dict, updates: dict) -> dict:
        """Recursively update a dictionary."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                BeautyConfig._deep_update(base[key], value)
            else:
                base[key] = value
        return base


def load_config(path: str | Path | None = None, **overrides) -> BeautyConfig:
    """
    Load configuration from file with optional overrides.

    Args:
        path: Path to YAML config file. If None, returns default config.
        **overrides: Key-value pairs to override config values.

    Returns:
        BeautyConfig instance.
    """
    if path is not None:
        config = BeautyConfig.from_yaml(path)
    else:
        config = BeautyConfig()

    if overrides:
        config = config.merge_with(overrides)

    return config
