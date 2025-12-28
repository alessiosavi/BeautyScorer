"""
Lightweight CPU-optimized model for beauty scoring.

Designed for efficient CPU training and inference:
- EfficientNet-Lite or MobileNetV2 backbone (no SE blocks)
- Simple pooling-based aggregation instead of transformers
- Lightweight attention mechanisms
- Supports INT8 quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from beauty_scorer.config import ModelConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.models.components.backbones import get_backbone
from beauty_scorer.models.components.encoders import LightweightAttention
from beauty_scorer.models.components.heads import ClassificationHead
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class LightweightPooling(nn.Module):
    """
    Lightweight pooling module for sequence aggregation.

    Uses simple mean/max pooling with optional learned weighting,
    much faster than transformer attention on CPU.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pooling_type: str = "mean_max",
    ):
        """
        Initialize pooling module.

        Args:
            embed_dim: Embedding dimension.
            pooling_type: Type of pooling ('mean', 'max', 'mean_max').
        """
        super().__init__()
        self.pooling_type = pooling_type

        if pooling_type == "mean_max":
            # Combine mean and max pooling with learned weights
            self.weight = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.output_dim = embed_dim
        else:
            self.weight = None
            self.output_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply pooling.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Boolean mask where True = valid (batch, seq_len).

        Returns:
            Pooled output (batch, embed_dim).
        """
        if mask is not None:
            # Mask out invalid positions
            mask_expanded = mask.unsqueeze(-1).float()
            x = x * mask_expanded

        if self.pooling_type == "mean":
            if mask is not None:
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                return x.sum(dim=1) / lengths
            return x.mean(dim=1)

        elif self.pooling_type == "max":
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            return x.max(dim=1)[0]

        else:  # mean_max
            if mask is not None:
                lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
                mean_pool = x.sum(dim=1) / lengths
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
                max_pool = x_masked.max(dim=1)[0]
            else:
                mean_pool = x.mean(dim=1)
                max_pool = x.max(dim=1)[0]

            # Learned combination
            weights = F.softmax(self.weight, dim=0)
            return weights[0] * mean_pool + weights[1] * max_pool


class LightweightCPUModel(BaseBeautyModel):
    """
    CPU-optimized lightweight model.

    Design principles:
    - Avoid operations slow on CPU (e.g., full attention)
    - Use depthwise separable convolutions
    - Simple pooling instead of transformers
    - Smaller embedding dimensions
    - Support for quantization

    Best used with:
    - Smaller batch sizes (8-16)
    - Fewer photos per person
    - Lower resolution inputs
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        # Use CPU-friendly backbone (MobileNetV2 has no SE blocks)
        backbone_name = self.config.backbone
        if "mobilenet_v3" in backbone_name:
            # MobileNetV3 has SE blocks which are slower on CPU
            logger.warning(
                "MobileNetV3 has SE blocks which are slower on CPU. "
                "Consider using mobilenet_v2 or efficientnet_lite0."
            )

        self.backbone, self.feature_dim = get_backbone(
            backbone_name,
            pretrained=self.config.pretrained,
            pool=True,
        )

        # Keep backbone frozen for CPU efficiency
        for param in self.backbone.parameters():
            param.requires_grad = False

        embed_dim = self.config.embed_dim

        # Simple linear projection
        self.photo_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.config.dropout),
        )

        # Lightweight pooling (no transformers)
        self.photo_pool = LightweightPooling(embed_dim, pooling_type="mean_max")
        self.face_pool = LightweightPooling(embed_dim, pooling_type="mean_max")

        # Optional lightweight attention (single-head, much faster than transformer)
        if self.config.num_encoder_layers > 0:
            self.photo_attn = LightweightAttention(embed_dim, dropout=self.config.dropout)
            self.face_attn = LightweightAttention(embed_dim, dropout=self.config.dropout)
            self.use_attention = True
        else:
            self.photo_attn = None
            self.face_attn = None
            self.use_attention = False

        # Simple classification head
        head_input_dim = embed_dim * 2
        self.head = ClassificationHead(
            input_dim=head_input_dim,
            hidden_dim=embed_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

        # For quantization support
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        logger.info(
            f"LightweightCPUModel initialized: "
            f"backbone={backbone_name}, "
            f"embed_dim={embed_dim}, "
            f"use_attention={self.use_attention}, "
            f"params={self.count_parameters():,}"
        )

    def _extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: (batch, max_images, C, H, W)

        Returns:
            Features (batch, max_images, feature_dim)
        """
        batch_size, max_images, c, h, w = images.shape

        # Flatten
        images_flat = images.view(batch_size * max_images, c, h, w)

        # Extract features (no checkpointing for CPU)
        features_flat = self.backbone(images_flat)

        # Reshape
        features = features_flat.view(batch_size, max_images, -1)

        return features

    def forward(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            photos: (batch, max_photos, C, H, W)
            photos_mask: (batch, max_photos)
            faces: (batch, max_photos, C, H, W)
            faces_mask: (batch, max_photos)

        Returns:
            Logits (batch, num_classes)
        """
        # Extract features
        photo_features = self._extract_features(photos)
        face_features = self._extract_features(faces)

        # Project
        photo_emb = self.photo_proj(photo_features)
        face_emb = self.face_proj(face_features)

        # Aggregate with attention or pooling
        if self.use_attention:
            photo_vec = self.photo_attn(photo_emb, photos_mask)
            face_vec = self.face_attn(face_emb, faces_mask)
        else:
            photo_vec = self.photo_pool(photo_emb, photos_mask)
            face_vec = self.face_pool(face_emb, faces_mask)

        # Classify
        combined = torch.cat([photo_vec, face_vec], dim=1)
        logits = self.head(combined)

        return logits

    def forward_quantized(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for quantized model.

        Args:
            photos, photos_mask, faces, faces_mask: Same as forward.

        Returns:
            Logits (batch, num_classes)
        """
        photos = self.quant(photos)
        faces = self.quant(faces)

        logits = self.forward(photos, photos_mask, faces, faces_mask)

        logits = self.dequant(logits)
        return logits

    def get_trainable_params(self) -> list[dict]:
        """Get trainable parameters."""
        params = []

        # All trainable params (backbone is frozen)
        trainable = [p for p in self.parameters() if p.requires_grad]
        params.append(
            {
                "params": trainable,
                "lr_scale": 1.0,
                "name": "all",
            }
        )

        return params

    def prepare_for_quantization(self) -> None:
        """Prepare model for quantization-aware training or post-training quantization."""
        self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        torch.ao.quantization.prepare(self, inplace=True)
        logger.info("Model prepared for quantization")

    def convert_to_quantized(self) -> None:
        """Convert prepared model to quantized version."""
        torch.ao.quantization.convert(self, inplace=True)
        logger.info("Model converted to quantized")

    @classmethod
    def from_quantized(cls, path: str) -> "LightweightCPUModel":
        """
        Load a quantized model.

        Args:
            path: Path to quantized model.

        Returns:
            Quantized model instance.
        """
        model = torch.jit.load(path)
        return model
