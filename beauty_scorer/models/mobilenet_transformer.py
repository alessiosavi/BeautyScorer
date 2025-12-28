"""
MobileNet + Transformer architecture for beauty scoring.

This is the improved version of the original architecture with:
- Configurable backbone unfreezing
- Optional positional encodings
- Cross-attention fusion between photo and face streams
- Squeeze-and-Excitation attention in the head
- Support for gradient checkpointing
"""

import torch
import torch.nn as nn

from beauty_scorer.config import ModelConfig
from beauty_scorer.models.base import BaseBeautyModel
from beauty_scorer.models.components.backbones import freeze_backbone, get_backbone
from beauty_scorer.models.components.encoders import CrossAttentionFusion, TransformerEncoder
from beauty_scorer.models.components.heads import ClassificationHead, SEAttentionHead
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class MobileNetTransformerModel(BaseBeautyModel):
    """
    Improved MobileNet + Transformer architecture.

    Architecture:
    1. MobileNetV3 backbone extracts features from each photo/face
    2. Projection layers map features to embedding dimension
    3. Transformer encoders aggregate variable-length sequences
    4. Optional cross-attention fuses photo and face information
    5. SE-attention head produces final classification

    Features:
    - Configurable backbone unfreezing for fine-tuning
    - Optional positional encoding (permutation-invariant by default)
    - Cross-attention between photo and face streams
    - Squeeze-and-Excitation attention in classification head
    - Gradient checkpointing for memory efficiency
    """

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize model.

        Args:
            config: Model configuration.
        """
        super().__init__(config)

        # Get backbone
        self.backbone, self.feature_dim = get_backbone(
            self.config.backbone,
            pretrained=self.config.pretrained,
            pool=True,
        )

        # Freeze/unfreeze backbone
        if self.config.unfreeze_backbone_layers == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            freeze_backbone(
                self.backbone,
                num_unfrozen_layers=self.config.unfreeze_backbone_layers,
            )

        # Projection layers
        embed_dim = self.config.embed_dim
        self.photo_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.face_proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        # Transformer encoders
        self.photo_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            dropout=self.config.dropout,
            use_positional_encoding=self.config.use_positional_encoding,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        self.face_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=self.config.num_encoder_layers,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            dropout=self.config.dropout,
            use_positional_encoding=self.config.use_positional_encoding,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )

        # Cross-attention fusion (optional)
        if self.config.use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
            )
        else:
            self.cross_attention = None

        # Classification head
        head_input_dim = embed_dim * 2  # Photo + Face features
        if self.config.use_se_attention:
            self.head = SEAttentionHead(
                input_dim=head_input_dim,
                hidden_dim=embed_dim,
                num_classes=self.config.num_classes,
                dropout=self.config.dropout,
            )
        else:
            self.head = ClassificationHead(
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
            f"MobileNetTransformerModel initialized: "
            f"backbone={self.config.backbone}, "
            f"embed_dim={embed_dim}, "
            f"layers={self.config.num_encoder_layers}, "
            f"params={self.count_parameters():,}"
        )

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for module in [self.photo_proj, self.face_proj]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _extract_features(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract features from images using backbone.

        Args:
            images: (batch, max_images, C, H, W)
            mask: (batch, max_images)

        Returns:
            Features (batch, max_images, feature_dim)
        """
        batch_size, max_images, c, h, w = images.shape

        # Flatten batch and sequence dimensions
        images_flat = images.view(batch_size * max_images, c, h, w)

        # Extract features
        if self.config.use_gradient_checkpointing and self.training:
            features_flat = torch.utils.checkpoint.checkpoint(
                self.backbone, images_flat, use_reentrant=False
            )
        else:
            features_flat = self.backbone(images_flat)

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
            photos_mask: (batch, max_photos) - True = valid
            faces: (batch, max_photos, C, H, W)
            faces_mask: (batch, max_photos) - True = valid

        Returns:
            Logits (batch, num_classes)
        """
        # Extract features
        photo_features = self._extract_features(photos, photos_mask)
        face_features = self._extract_features(faces, faces_mask)

        # Project to embedding dimension
        photo_emb = self.photo_proj(photo_features)
        face_emb = self.face_proj(face_features)

        # Apply cross-attention if enabled
        if self.cross_attention is not None:
            photo_emb, face_emb = self.cross_attention(photo_emb, face_emb, photos_mask, faces_mask)

        # Encode sequences
        photo_vec = self.photo_encoder(photo_emb, photos_mask)
        face_vec = self.face_encoder(face_emb, faces_mask)

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

        # Concatenate and classify
        combined = torch.cat([photo_vec, face_vec], dim=1)
        logits = self.head(combined)

        return logits

    def get_trainable_params(self) -> list[dict]:
        """
        Get trainable parameters organized by group.

        Returns backbone params with lower learning rate.
        """
        params = []

        # Backbone parameters (with lower LR)
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            params.append(
                {
                    "params": backbone_params,
                    "lr_scale": 0.1,  # Lower LR for backbone
                    "name": "backbone",
                }
            )

        # Projection and encoder parameters
        encoder_params = list(self.photo_proj.parameters()) + list(self.face_proj.parameters())
        encoder_params += list(self.photo_encoder.parameters()) + list(
            self.face_encoder.parameters()
        )
        if self.cross_attention is not None:
            encoder_params += list(self.cross_attention.parameters())
        params.append(
            {
                "params": encoder_params,
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

    def get_intermediate_features(
        self,
        photos: torch.Tensor,
        photos_mask: torch.Tensor,
        faces: torch.Tensor,
        faces_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Get intermediate features for analysis.

        Args:
            photos: Photo tensor.
            photos_mask: Photo mask.
            faces: Face tensor.
            faces_mask: Face mask.

        Returns:
            Dictionary of intermediate features.
        """
        # Extract features
        photo_features = self._extract_features(photos, photos_mask)
        face_features = self._extract_features(faces, faces_mask)

        # Project
        photo_emb = self.photo_proj(photo_features)
        face_emb = self.face_proj(face_features)

        # Encode
        photo_vec = self.photo_encoder(photo_emb, photos_mask)
        face_vec = self.face_encoder(face_emb, faces_mask)

        return {
            "photo_backbone_features": photo_features,
            "face_backbone_features": face_features,
            "photo_embeddings": photo_emb,
            "face_embeddings": face_emb,
            "photo_encoded": photo_vec,
            "face_encoded": face_vec,
        }
