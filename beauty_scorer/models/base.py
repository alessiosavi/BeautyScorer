"""
Abstract base class for all beauty scoring models.

Defines the common interface that all model implementations must follow.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from beauty_scorer.config import ModelConfig
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class BaseBeautyModel(nn.Module, ABC):
    """
    Abstract base class for all beauty scoring models.

    All model implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize base model.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config or ModelConfig()
        self.num_classes = self.config.num_classes
        self._is_compiled = False

    @abstractmethod
    def forward(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning logits.

        Args:
            photos: Photo tensor (batch, max_photos, C, H, W).
            photos_mask: Boolean mask where True = valid photo (batch, max_photos).
            faces: Face tensor (batch, max_photos, C, H, W).
            faces_mask: Boolean mask where True = valid face (batch, max_photos).

        Returns:
            Logits tensor (batch, num_classes).
        """
        pass

    @abstractmethod
    def get_trainable_params(self) -> list[dict]:
        """
        Get trainable parameters organized by group.

        Returns:
            List of parameter group dictionaries for optimizer.
            Each dict should have 'params' and optionally 'lr_scale'.
        """
        pass

    def predict(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scores and probabilities.

        Args:
            photos: Photo tensor (batch, max_photos, C, H, W).
            photos_mask: Boolean mask (batch, max_photos).
            faces: Face tensor (batch, max_photos, C, H, W).
            faces_mask: Boolean mask (batch, max_photos).

        Returns:
            Tuple of (predicted_scores, probabilities).
            - predicted_scores: (batch,) integer scores (1-indexed)
            - probabilities: (batch, num_classes) probability distribution
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(photos, photos_mask, faces, faces_mask)
            probs = torch.softmax(logits, dim=-1)
            scores = torch.argmax(logits, dim=-1) + 1  # 1-indexed scores

        return scores, probs

    def get_feature_dim(self) -> int:
        """Get the combined feature dimension before classification head."""
        return self.config.embed_dim * 2  # Photo + Face features

    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count model parameters.

        Args:
            trainable_only: Only count trainable parameters.

        Returns:
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def compile_model(self, mode: str = "default") -> None:
        """
        Compile model using torch.compile for faster execution.

        Args:
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune').
        """
        if not self._is_compiled:
            try:
                self.forward = torch.compile(self.forward, mode=mode)
                self._is_compiled = True
                logger.info(f"Model compiled with mode: {mode}")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

    def save(self, path: str) -> None:
        """
        Save model state dict.

        Args:
            path: Path to save the model.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config.model_dump() if self.config else None,
            },
            path,
        )
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        config: ModelConfig | None = None,
        device: torch.device | None = None,
    ) -> "BaseBeautyModel":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            config: Config override. If None, uses saved config.
            device: Device to load model to.

        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)

        # Get config
        if config is None and checkpoint.get("config"):
            config = ModelConfig(**checkpoint["config"])
        elif config is None:
            config = ModelConfig()

        # Create model instance
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])

        if device:
            model = model.to(device)

        logger.info(f"Model loaded from {path}")
        return model

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        if hasattr(self, "backbone"):
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

    def unfreeze_backbone(self, num_layers: int | None = None) -> None:
        """
        Unfreeze backbone parameters.

        Args:
            num_layers: Number of layers from end to unfreeze. If None, unfreeze all.
        """
        if hasattr(self, "backbone"):
            if num_layers is None:
                for param in self.backbone.parameters():
                    param.requires_grad = True
                logger.info("Backbone fully unfrozen")
            else:
                # Unfreeze last N layers
                from beauty_scorer.models.components.backbones import freeze_backbone

                freeze_backbone(self.backbone, num_unfrozen_layers=num_layers)
                logger.info(f"Backbone: last {num_layers} layers unfrozen")

    def get_attention_maps(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> dict | None:
        """
        Get attention maps for visualization (if supported).

        Args:
            photos: Photo tensor.
            photos_mask: Photo mask.
            faces: Face tensor.
            faces_mask: Face mask.

        Returns:
            Dictionary of attention maps, or None if not supported.
        """
        return None

    def __repr__(self) -> str:
        """String representation with parameter count."""
        trainable = self.count_parameters(trainable_only=True)
        total = self.count_parameters(trainable_only=False)
        return (
            f"{self.__class__.__name__}(\n"
            f"  config={self.config.architecture},\n"
            f"  trainable_params={trainable:,},\n"
            f"  total_params={total:,}\n"
            f")"
        )
