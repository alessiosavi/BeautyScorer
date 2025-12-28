"""
BeautyScorer: Deep learning model for predicting beauty scores from photos.

This package provides:
- Multiple model architectures (MobileNet+Transformer, ViT+ArcFace, Lightweight CPU)
- Complete training pipeline with callbacks and metrics
- High-level inference API
- Model export utilities (ONNX, TorchScript)
"""

__version__ = "1.0.0"
__author__ = "BeautyScorer Team"

from beauty_scorer.config import BeautyConfig, DataConfig, ModelConfig, TrainingConfig, load_config
from beauty_scorer.inference.predictor import BeautyPredictor
from beauty_scorer.models.factory import ModelFactory, create_model
from beauty_scorer.training.trainer import Trainer

__all__ = [
    # Config
    "BeautyConfig",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "load_config",
    # Models
    "ModelFactory",
    "create_model",
    # Training
    "Trainer",
    # Inference
    "BeautyPredictor",
]
