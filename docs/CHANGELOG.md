# Changelog

All notable changes to BeautyScorer are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New experimental features in development

## [1.0.0] - 2025-12-28

### Added

- **Complete project rewrite** with professional modular architecture
- **Three model architectures**:
  - `MobileNetTransformerModel`: Balanced accuracy/speed (default)
  - `ViTArcFaceModel`: State-of-the-art accuracy with DINOv2/ViT backbone
  - `LightweightCPUModel`: Optimized for CPU training and inference
- **Pydantic configuration system** with type validation
- **YAML configuration files**:
  - `default.yaml`: GPU training with MobileNet+Transformer
  - `cpu_training.yaml`: CPU-optimized lightweight model
  - `advanced_model.yaml`: ViT+ArcFace for maximum accuracy
- **Model components**:
  - Transformer encoders with CLS token aggregation
  - Cross-attention fusion between photo and face streams
  - Squeeze-and-Excitation attention heads
  - Attention pooling for ViT models
  - Backbone registry with 10+ pretrained options
- **Training features**:
  - Automatic Mixed Precision (AMP) training
  - Gradient accumulation for larger effective batch sizes
  - Gradient clipping for training stability
  - Multiple loss functions: CrossEntropy, Focal, Ordinal
  - Learning rate schedulers: Cosine, Step, Plateau, OneCycle
  - Warmup epochs support
  - Early stopping with best model restoration
  - Model checkpointing
- **Callbacks system**:
  - `EarlyStopping`: Stop when validation metric plateaus
  - `ModelCheckpoint`: Save best and last models
  - `ProgressBar`: tqdm-based progress display
  - `LRMonitor`: Learning rate logging
- **Metrics**:
  - Accuracy, MAE, RMSE
  - Within-1 and Within-2 accuracy
  - Confusion matrix
  - Per-class accuracy
- **Inference module**:
  - `BeautyPredictor`: High-level prediction API
  - Batch prediction support
  - Detailed prediction explanations
- **Model export**:
  - ONNX export for cross-platform deployment
  - TorchScript export for PyTorch deployment
  - INT8 quantization for edge devices
  - `ONNXPredictor` for inference without PyTorch
- **Data pipeline**:
  - Albumentations-based augmentation (light/medium/heavy)
  - Multiple face detection backends (YOLOv8, RetinaFace, MTCNN, OpenCV)
  - Automatic class weight computation
  - Stratified train/validation split
- **CLI scripts**:
  - `train.py`: Training with config and CLI overrides
  - `evaluate.py`: Model evaluation with metrics and visualizations
  - `infer.py`: Single/batch inference from command line
  - `export_model.py`: Export to various formats
- **Utilities**:
  - Structured logging with TensorBoard and W&B support
  - Device management with automatic detection
  - Visualization for training curves and confusion matrices
  - Reproducibility via seed setting
- **Documentation**:
  - Comprehensive API reference
  - Architecture deep dive
  - Configuration guide
  - Quickstart tutorial
  - Development guide
- **Testing**:
  - Unit tests for models, data, inference
  - Integration tests
  - Debug scripts for development

### Changed

- **Complete architecture redesign** from notebook-style to modular package
- **Configuration**: From hardcoded values to Pydantic + YAML
- **Logging**: From print statements to structured logging
- **Data loading**: From manual to Albumentations pipeline
- **Training loop**: From notebook to Trainer class with callbacks

### Fixed

- Hardcoded paths replaced with configuration
- Global state removed
- Memory leaks in data loading
- Missing type hints added throughout
- Proper train/validation split (was training-only before)

### Removed

- Jupyter notebook (replaced with Python scripts)
- TensorFlow dependency (was unused)
- Global device/config variables

## [0.0.1-RC1] - 2025-12-27

### Fixed

- Small bugs in method signature
- Add example for inference

## [0.0.1] - 2025-12-26

### Added

- Initial implementation with MobileNetV3 + Transformer
- Basic training loop
- Inference utilities
- Documentation for model architecture

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | 2025-12-28 | Complete rewrite, 3 architectures, production-ready |
| 0.0.1-RC1 | 2025-12-27 | Bug fixes, inference example |
| 0.0.1 | 2025-12-26 | Initial release |

## Migration Guide

### From 0.0.1 to 1.0.0

The 1.0.0 release is a complete rewrite. Migration steps:

1. **Update imports**:

```python
# Old
from model import BeautyScoreModel
import utils

# New
from beauty_scorer import create_model, BeautyPredictor
from beauty_scorer.config import load_config
```

1. **Update model creation**:

```python
# Old
CONF = utils.load_conf("conf.yaml")[0]
model = BeautyScoreModel(conf=CONF)

# New
config = load_config("configs/default.yaml")
model = create_model(config=config.model)
```

1. **Update inference**:

```python
# Old
score, probs = score_person(person_id, model, basepath)

# New
predictor = BeautyPredictor("model.pt")
result = predictor.predict_person(photo_paths)
score = result["score"]
probs = result["probabilities"]
```

1. **Update training**:

```python
# Old (notebook-style)
train(model, raw_dl)

# New
from beauty_scorer.training import Trainer
trainer = Trainer(model, train_loader, val_loader, config=config)
results = trainer.fit()
```

1. **Update configuration**:

```yaml
# Old (conf.yaml)
C: 3
H: 224
W: 224
N_MAX: 9
BATCH_SIZE: 32

# New (configs/default.yaml)
model:
  num_classes: 9
data:
  image_size: [224, 224]
  max_photos: 9
training:
  batch_size: 32
```

### Checkpoint Compatibility

Old checkpoints can be loaded with some modifications:

```python
# Load old checkpoint
old_state = torch.load("old_model.pt", weights_only=True)

# Create new model
config = ModelConfig(
    architecture="mobilenet_transformer",
    embed_dim=512,
    num_encoder_layers=4
)
model = create_model(config=config)

# Load state dict (may need key mapping)
model.load_state_dict(old_state, strict=False)
```
