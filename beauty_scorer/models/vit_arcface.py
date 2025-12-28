"""
Vision Transformer + ArcFace architecture for beauty scoring.

State-of-the-art architecture using:
- DINOv2/ViT backbone for photo features
- Option for face-specific backbone
- Attention-based pooling
- Multi-scale feature aggregation
"""

import torch
import torch.nn as nn

from beauty_scorer.config import ModelConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.models.components.backbones import freeze_backbone, get_backbone
from beauty_scorer.models.components.encoders import (
    AttentionPooling,
    CrossAttentionFusion,
    PositionalEncoding,
)
from beauty_scorer.models.components.heads import SEAttentionHead
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class ViTArcFaceModel(BaseBeautyModel):
    """
    Vision Transformer + ArcFace architecture.

    This model uses:
    - DINOv2 or ViT backbone for photos (excellent self-supervised features)
    - Optional separate backbone for faces (can use face-specific model)
    - Attention-based pooling instead of CLS token
    - Cross-attention fusion between streams
    - Multi-layer feature aggregation

    Best used with:
    - Higher resolution inputs (384x384)
    - Heavy augmentation
    - Longer training
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        # Photo backbone (ViT/DINOv2)
        self.photo_backbone, photo_feature_dim = get_backbone(
            self.config.backbone,
            pretrained=self.config.pretrained,
            pool=False,  # ViT returns CLS features directly
        )

        # Face backbone (can be same or different)
        # For now, use same backbone; could be extended to use face-specific model
        self.face_backbone, face_feature_dim = get_backbone(
            self.config.backbone,
            pretrained=self.config.pretrained,
            pool=False,
        )
        self.share_backbone = True  # Can be made configurable

        # Freeze/unfreeze backbones
        if self.config.unfreeze_backbone_layers == 0:
            for param in self.photo_backbone.parameters():
                param.requires_grad = False
            for param in self.face_backbone.parameters():
                param.requires_grad = False
        else:
            freeze_backbone(
                self.photo_backbone,
                num_unfrozen_layers=self.config.unfreeze_backbone_layers,
            )
            freeze_backbone(
                self.face_backbone,
                num_unfrozen_layers=self.config.unfreeze_backbone_layers,
            )

        embed_dim = self.config.embed_dim

        # Feature projection (ViT features to embed_dim)
        self.photo_proj = nn.Sequential(
            nn.Linear(photo_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(face_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Positional encoding for sequence
        if self.config.use_positional_encoding:
            self.photo_pos_enc = PositionalEncoding(
                embed_dim, dropout=self.config.dropout, learnable=True
            )
            self.face_pos_enc = PositionalEncoding(
                embed_dim, dropout=self.config.dropout, learnable=True
            )
        else:
            self.photo_pos_enc = None
            self.face_pos_enc = None

        # Self-attention layers for sequence processing
        self.photo_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.ff_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.config.num_encoder_layers // 2,
        )
        self.face_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=self.config.num_heads,
                dim_feedforward=self.config.ff_dim,
                dropout=self.config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=self.config.num_encoder_layers // 2,
        )

        # Cross-attention fusion
        if self.config.use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        else:
            self.cross_attention = None

        # Attention pooling
        self.photo_pool = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=self.config.num_heads,
        )
        self.face_pool = AttentionPooling(
            embed_dim=embed_dim,
            num_heads=self.config.num_heads,
        )

        # Classification head with SE attention
        head_input_dim = embed_dim * 2
        self.head = SEAttentionHead(
            input_dim=head_input_dim,
            hidden_dim=embed_dim,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )

        # Learnable embeddings for missing faces/photos
        # When a sample has no valid faces, use this learned embedding instead
        # This allows the model to learn a meaningful representation for "no face detected"
        self.no_face_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.no_photo_embedding = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.no_face_embedding, std=0.02)
        nn.init.normal_(self.no_photo_embedding, std=0.02)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"ViTArcFaceModel initialized: "
            f"backbone={self.config.backbone}, "
            f"embed_dim={embed_dim}, "
            f"params={self.count_parameters():,}"
        )

    def _init_weights(self):
        """Initialize weights."""
        for module in [self.photo_proj, self.face_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _extract_vit_features(
        self,
        images: torch.Tensor,
        backbone: nn.Module,
    ) -> torch.Tensor:
        """
        Extract features from images using ViT backbone.

        Args:
            images: (batch, max_images, C, H, W)
            backbone: ViT backbone module.

        Returns:
            Features (batch, max_images, feature_dim)
        """
        batch_size, max_images, c, h, w = images.shape

        # Flatten batch and sequence
        images_flat = images.view(batch_size * max_images, c, h, w)

        # Extract features
        if self.config.use_gradient_checkpointing and self.training:
            features_flat = torch.utils.checkpoint.checkpoint(
                backbone, images_flat, use_reentrant=False
            )
        else:
            features_flat = backbone(images_flat)

        # Reshape back
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
        # Extract ViT features
        photo_features = self._extract_vit_features(photos, self.photo_backbone)

        if self.share_backbone:
            face_features = self._extract_vit_features(faces, self.photo_backbone)
        else:
            face_features = self._extract_vit_features(faces, self.face_backbone)

        # Project to embedding dimension
        photo_emb = self.photo_proj(photo_features)
        face_emb = self.face_proj(face_features)

        # Add positional encoding
        if self.photo_pos_enc is not None:
            photo_emb = self.photo_pos_enc(photo_emb)
        if self.face_pos_enc is not None:
            face_emb = self.face_pos_enc(face_emb)

        # Self-attention with empty mask handling
        # Check if any sample has all tokens masked (would cause NaN)
        photos_has_valid = photos_mask.any(dim=1).all()
        faces_has_valid = faces_mask.any(dim=1).all()

        if photos_has_valid:
            photo_padding_mask = ~photos_mask
            photo_emb = self.photo_self_attn(photo_emb, src_key_padding_mask=photo_padding_mask)
        # else: skip self-attention, keep photo_emb as-is (from projection)

        if faces_has_valid:
            face_padding_mask = ~faces_mask
            face_emb = self.face_self_attn(face_emb, src_key_padding_mask=face_padding_mask)
        # else: skip self-attention, keep face_emb as-is (from projection)

        # Cross-attention
        if self.cross_attention is not None:
            photo_emb, face_emb = self.cross_attention(photo_emb, face_emb, photos_mask, faces_mask)

        # Attention pooling
        photo_vec = self.photo_pool(photo_emb, photos_mask)
        face_vec = self.face_pool(face_emb, faces_mask)

        # Replace invalid vectors with learned embeddings
        # This allows model to handle images without faces as valid information
        batch_size = photos.size(0)
        has_photos = photos_mask.any(dim=1)  # (batch,)
        has_faces = faces_mask.any(dim=1)  # (batch,)

        # Use learned embedding for samples without valid photos
        if not has_photos.all():
            no_photo_expanded = self.no_photo_embedding.unsqueeze(0).expand(batch_size, -1)
            photo_vec = torch.where(
                has_photos.unsqueeze(-1),
                photo_vec,
                no_photo_expanded,
            )

        # Use learned embedding for samples without valid faces
        if not has_faces.all():
            no_face_expanded = self.no_face_embedding.unsqueeze(0).expand(batch_size, -1)
            face_vec = torch.where(
                has_faces.unsqueeze(-1),
                face_vec,
                no_face_expanded,
            )

        # Classify
        combined = torch.cat([photo_vec, face_vec], dim=1)
        logits = self.head(combined)

        return logits

    def get_trainable_params(self) -> list[dict]:
        """Get trainable parameters organized by group."""
        params = []

        # Backbone parameters (very low LR)
        backbone_params = [p for p in self.photo_backbone.parameters() if p.requires_grad]
        backbone_params += [p for p in self.face_backbone.parameters() if p.requires_grad]
        if backbone_params:
            params.append(
                {
                    "params": backbone_params,
                    "lr_scale": 0.01,  # Very low LR for pretrained ViT
                    "name": "backbone",
                }
            )

        # Projection and attention parameters
        other_params = (
            list(self.photo_proj.parameters())
            + list(self.face_proj.parameters())
            + list(self.photo_self_attn.parameters())
            + list(self.face_self_attn.parameters())
            + list(self.photo_pool.parameters())
            + list(self.face_pool.parameters())
        )
        if self.cross_attention is not None:
            other_params += list(self.cross_attention.parameters())
        if self.photo_pos_enc is not None:
            other_params += list(self.photo_pos_enc.parameters())
        if self.face_pos_enc is not None:
            other_params += list(self.face_pos_enc.parameters())

        params.append(
            {
                "params": other_params,
                "lr_scale": 1.0,
                "name": "encoder",
            }
        )

        # Head parameters
        params.append(
            {
                "params": list(self.head.parameters()),
                "lr_scale": 1.0,
                "name": "head",
            }
        )

        return params
