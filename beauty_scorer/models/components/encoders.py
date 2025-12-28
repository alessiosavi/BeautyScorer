"""
Transformer encoders and attention modules for sequence aggregation.

Provides transformer encoders for aggregating variable-length photo/face
sequences, cross-attention fusion, and attention pooling mechanisms.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class PositionalEncoding(nn.Module):
    """
    Learnable or sinusoidal positional encoding.

    For permutation-invariant design, this can be disabled.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 100,
        dropout: float = 0.1,
        learnable: bool = True,
    ):
        """
        Initialize positional encoding.

        Args:
            embed_dim: Embedding dimension.
            max_len: Maximum sequence length.
            dropout: Dropout rate.
            learnable: Whether to use learnable encodings.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learnable = learnable

        if learnable:
            self.pe = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        else:
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).

        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for sequence aggregation.

    Uses CLS token to aggregate variable-length sequences into
    a fixed-size representation.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize transformer encoder.

        Args:
            embed_dim: Embedding dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            ff_dim: Feed-forward dimension.
            dropout: Dropout rate.
            use_positional_encoding: Whether to use positional encoding.
            use_gradient_checkpointing: Enable gradient checkpointing.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Positional encoding
        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        else:
            self.pos_encoding = None

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Boolean mask where True = valid token (batch, seq_len).

        Returns:
            Aggregated representation (batch, embed_dim).
        """
        batch_size = x.size(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # Create padding mask (True = ignore)
        if mask is not None:
            # Add False for CLS token (always attend)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=mask.device)
            padding_mask = torch.cat([cls_mask, ~mask], dim=1)
        else:
            padding_mask = None

        # Apply transformer encoder
        if self.use_gradient_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.encoder, x, src_key_padding_mask=padding_mask, use_reentrant=False
            )
        else:
            x = self.encoder(x, src_key_padding_mask=padding_mask)

        # Extract CLS token
        cls_output = x[:, 0]
        cls_output = self.norm(cls_output)

        return cls_output


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention module for fusing photo and face streams.

    Allows the photo stream to attend to face features and vice versa
    before final fusion.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize cross-attention fusion.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        # Photo attends to faces
        self.photo_to_face_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Face attends to photos
        self.face_to_photo_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms
        self.norm_photo = nn.LayerNorm(embed_dim)
        self.norm_face = nn.LayerNorm(embed_dim)

        # Feed-forward
        self.ff_photo = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_face = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.norm_photo_ff = nn.LayerNorm(embed_dim)
        self.norm_face_ff = nn.LayerNorm(embed_dim)

    def forward(
        self,
        photo_features: torch.Tensor,
        face_features: torch.Tensor,
        photo_mask: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention fusion.

        Args:
            photo_features: Photo features (batch, seq_len, embed_dim).
            face_features: Face features (batch, seq_len, embed_dim).
            photo_mask: Photo mask where True = valid (batch, seq_len).
            face_mask: Face mask where True = valid (batch, seq_len).

        Returns:
            Tuple of (enhanced_photo_features, enhanced_face_features).
        """
        # Convert masks to key_padding_mask format (True = ignore)
        photo_key_mask = ~photo_mask if photo_mask is not None else None
        face_key_mask = ~face_mask if face_mask is not None else None

        # Check for samples with no valid faces (would cause NaN in attention)
        # For these samples, skip cross-attention and use identity
        if face_mask is not None:
            has_valid_faces = face_mask.any(dim=1)  # (batch,)
            all_have_faces = has_valid_faces.all()
        else:
            all_have_faces = True

        if face_mask is not None:
            has_valid_photos = photo_mask.any(dim=1) if photo_mask is not None else None
            all_have_photos = has_valid_photos.all() if has_valid_photos is not None else True
        else:
            all_have_photos = True

        # Photo attends to faces (skip if any sample has no valid faces)
        if all_have_faces:
            photo_attn_out, _ = self.photo_to_face_attn(
                query=photo_features,
                key=face_features,
                value=face_features,
                key_padding_mask=face_key_mask,
            )
            photo_features = self.norm_photo(photo_features + photo_attn_out)
        else:
            # Skip attention for samples without faces, apply norm only
            photo_features = self.norm_photo(photo_features)
        photo_features = self.norm_photo_ff(photo_features + self.ff_photo(photo_features))

        # Face attends to photos (skip if any sample has no valid photos)
        if all_have_photos:
            face_attn_out, _ = self.face_to_photo_attn(
                query=face_features,
                key=photo_features,
                value=photo_features,
                key_padding_mask=photo_key_mask,
            )
            face_features = self.norm_face(face_features + face_attn_out)
        else:
            # Skip attention for samples without photos, apply norm only
            face_features = self.norm_face(face_features)
        face_features = self.norm_face_ff(face_features + self.ff_face(face_features))

        return photo_features, face_features


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence aggregation.

    Alternative to CLS token - learns to weight different positions.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
    ):
        """
        Initialize attention pooling.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
        """
        super().__init__()

        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply attention pooling.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Boolean mask where True = valid token (batch, seq_len).

        Returns:
            Pooled representation (batch, embed_dim).
        """
        batch_size = x.size(0)
        embed_dim = x.size(-1)

        # Check for samples with no valid tokens
        if mask is not None:
            has_valid = mask.any(dim=1)  # (batch,)
            all_valid = has_valid.all()
            any_invalid = (~has_valid).any()
        else:
            all_valid = True
            any_invalid = False

        # Fast path: all samples have valid tokens
        if all_valid:
            query = self.query.expand(batch_size, -1, -1)
            key_mask = ~mask if mask is not None else None

            output, _ = self.attention(
                query=query,
                key=x,
                value=x,
                key_padding_mask=key_mask,
            )
            return self.norm(output.squeeze(1))

        # Slow path: handle per-sample invalid masks
        # Use same dtype as input to handle AMP correctly
        output = torch.zeros(batch_size, embed_dim, device=x.device, dtype=x.dtype)

        # Process valid samples with attention
        if has_valid.any():
            valid_idx = has_valid.nonzero(as_tuple=True)[0]
            x_valid = x[valid_idx]
            mask_valid = mask[valid_idx]

            query = self.query.expand(len(valid_idx), -1, -1)
            key_mask = ~mask_valid

            attn_out, _ = self.attention(
                query=query,
                key=x_valid,
                value=x_valid,
                key_padding_mask=key_mask,
            )
            # Cast to output dtype for AMP compatibility (attention may output float16)
            output[valid_idx] = attn_out.squeeze(1).to(output.dtype)

        # Process invalid samples with mean pooling
        if any_invalid:
            invalid_idx = (~has_valid).nonzero(as_tuple=True)[0]
            output[invalid_idx] = x[invalid_idx].mean(dim=1).to(output.dtype)

        return self.norm(output)


class LightweightAttention(nn.Module):
    """
    Lightweight single-head attention for CPU efficiency.

    Uses simple dot-product attention without multi-head overhead.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Initialize lightweight attention.

        Args:
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.scale = embed_dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply lightweight attention and pool.

        Args:
            x: Input tensor (batch, seq_len, embed_dim).
            mask: Boolean mask where True = valid (batch, seq_len).

        Returns:
            Pooled representation (batch, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape
        device = x.device
        dtype = x.dtype

        # Check which samples have valid tokens
        if mask is not None:
            valid_counts = mask.sum(dim=1)  # (batch,)
            has_valid = valid_counts > 0  # (batch,)
            all_valid = has_valid.all()
            any_invalid = (~has_valid).any()
        else:
            all_valid = True
            any_invalid = False

        # Fast path: all samples have valid tokens
        if all_valid:
            q = self.query(x.mean(dim=1, keepdim=True))
            k = self.key(x)
            v = self.value(x)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            output = torch.matmul(attn, v).squeeze(1)
            return self.norm(output)

        # Slow path: handle per-sample invalid masks
        # Use same dtype as input to handle AMP correctly
        output = torch.zeros(batch_size, embed_dim, device=device, dtype=dtype)

        # Process valid samples with attention
        if has_valid.any():
            valid_idx = has_valid.nonzero(as_tuple=True)[0]
            x_valid = x[valid_idx]
            mask_valid = mask[valid_idx] if mask is not None else None

            q = self.query(x_valid.mean(dim=1, keepdim=True))
            k = self.key(x_valid)
            v = self.value(x_valid)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if mask_valid is not None:
                scores = scores.masked_fill(~mask_valid.unsqueeze(1), float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            attn_output = torch.matmul(attn, v).squeeze(1)
            # Cast to output dtype for AMP compatibility (attention may output float16)
            output[valid_idx] = attn_output.to(output.dtype)

        # Process invalid samples with mean pooling (over all positions)
        if any_invalid:
            invalid_idx = (~has_valid).nonzero(as_tuple=True)[0]
            # Use mean of input embeddings (which are zeros for invalid, but this is safe)
            output[invalid_idx] = x[invalid_idx].mean(dim=1).to(output.dtype)

        return self.norm(output)
