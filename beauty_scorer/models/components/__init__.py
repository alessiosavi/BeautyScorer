"""Model components for BeautyScorer."""

from beauty_scorer.models.components.backbones import (
    BackboneRegistry,
    get_backbone,
    list_available_backbones,
)
from beauty_scorer.models.components.encoders import (
    AttentionPooling,
    CrossAttentionFusion,
    PositionalEncoding,
    TransformerEncoder,
)
from beauty_scorer.models.components.heads import (
    ClassificationHead,
    RegressionHead,
    SEAttentionHead,
)

__all__ = [
    # Encoders
    "TransformerEncoder",
    "CrossAttentionFusion",
    "PositionalEncoding",
    "AttentionPooling",
    # Heads
    "ClassificationHead",
    "SEAttentionHead",
    "RegressionHead",
    # Backbones
    "BackboneRegistry",
    "get_backbone",
    "list_available_backbones",
]
