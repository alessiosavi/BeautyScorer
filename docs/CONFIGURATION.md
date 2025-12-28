# Configuration Guide

Complete reference for all configuration options in BeautyScorer.

## Overview

BeautyScorer uses Pydantic for type-safe configuration with YAML file support. All settings are organized into logical groups:

- **Model**: Architecture and model parameters
- **Training**: Training process settings
- **Data**: Data processing and augmentation
- **Paths**: File and directory paths
- **Logging**: Logging and experiment tracking

## Loading Configuration

### From YAML File

```python
from beauty_scorer.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Access settings
print(config.model.architecture)
print(config.training.batch_size)
```

### With Overrides

```python
# Override specific values
config = load_config("configs/default.yaml")
config = config.merge_with({
    "training": {"epochs": 100},
    "model": {"embed_dim": 256}
})
```

### Programmatic Creation

```python
from beauty_scorer.config import BeautyConfig, ModelConfig, TrainingConfig

config = BeautyConfig(
    model=ModelConfig(architecture="vit_arcface", embed_dim=768),
    training=TrainingConfig(epochs=100, batch_size=16)
)
```

## Configuration Sections

### Model Configuration

Controls the neural network architecture.

```yaml
model:
  # Architecture selection
  architecture: mobilenet_transformer  # mobilenet_transformer, vit_arcface, lightweight_cpu

  # Backbone network
  backbone: mobilenet_v3_large  # See backbone options below
  pretrained: true              # Use pretrained weights

  # Embedding dimensions
  embed_dim: 512                # Main embedding dimension (64-2048)
  ff_dim: 2048                  # Feed-forward dimension (256-8192)

  # Transformer settings
  num_encoder_layers: 4         # Transformer layers (1-12)
  num_heads: 8                  # Attention heads (1-16)
  dropout: 0.2                  # Dropout rate (0.0-0.8)

  # Output
  num_classes: 9                # Number of score classes

  # Fine-tuning
  unfreeze_backbone_layers: 0   # Layers to unfreeze from end (0 = all frozen)

  # Optional features
  use_positional_encoding: false  # Add position info (breaks permutation invariance)
  use_cross_attention: true       # Cross-attention between photo/face streams
  use_se_attention: true          # SE attention in classification head
  use_gradient_checkpointing: false  # Trade compute for memory
```

#### Available Backbones

| Backbone | Family | Feature Dim | Notes |
|----------|--------|-------------|-------|
| `mobilenet_v3_large` | MobileNet | 960 | Default, good balance |
| `mobilenet_v3_small` | MobileNet | 576 | Faster, less accurate |
| `mobilenet_v2` | MobileNet | 1280 | No SE blocks, CPU-friendly |
| `efficientnet_b0` | EfficientNet | 1280 | Efficient and accurate |
| `efficientnet_lite0` | EfficientNet | 1280 | Mobile-optimized |
| `resnet50` | ResNet | 2048 | Classic architecture |
| `vit_b_16` | ViT | 768 | Vision Transformer |
| `vit_b_32` | ViT | 768 | Faster ViT |
| `dinov2_vits14` | DINOv2 | 384 | SOTA self-supervised |
| `dinov2_vitb14` | DINOv2 | 768 | Larger DINOv2 |

### Training Configuration

Controls the training process.

```yaml
training:
  # Basic settings
  batch_size: 32                # Samples per batch (1-256)
  epochs: 50                    # Total training epochs
  seed: 42                      # Random seed for reproducibility

  # Learning rate
  learning_rate: 0.0001         # Initial learning rate
  backbone_lr_multiplier: 0.1   # LR multiplier for backbone (0.0-1.0)
  weight_decay: 0.01            # L2 regularization

  # Early stopping
  patience: 5                   # Epochs without improvement before stopping

  # Loss function
  loss_function: cross_entropy  # cross_entropy, focal, ordinal
  label_smoothing: 0.1          # Label smoothing factor (0.0-0.5)
  focal_gamma: 2.0              # Gamma for focal loss

  # Learning rate scheduler
  scheduler: cosine             # cosine, step, plateau, onecycle, none
  scheduler_warmup_epochs: 2    # Warmup epochs (0+)

  # Optimization
  use_amp: true                 # Automatic Mixed Precision
  gradient_accumulation_steps: 1  # Effective batch = batch_size × this
  max_grad_norm: 1.0            # Gradient clipping

  # Data split
  val_split: 0.1                # Validation set ratio (0.0-0.5)

  # Data loading
  num_workers: 4                # DataLoader workers
  pin_memory: true              # Pin memory for GPU transfer
```

#### Loss Function Options

| Loss | When to Use |
|------|-------------|
| `cross_entropy` | Default, works well for balanced data |
| `focal` | Class imbalance, focuses on hard examples |
| `ordinal` | Ordinal regression, respects score ordering |

#### Scheduler Options

| Scheduler | Behavior |
|-----------|----------|
| `cosine` | Smooth cosine annealing to min LR |
| `step` | Step decay at fixed intervals |
| `plateau` | Reduce LR when metric plateaus |
| `onecycle` | One-cycle policy (warm up then down) |
| `none` | Constant learning rate |

### Data Configuration

Controls data processing and augmentation.

```yaml
data:
  # Input sizes
  image_size: [224, 224]        # Photo size [height, width]
  face_size: [224, 224]         # Face crop size

  # Sequence settings
  max_photos: 9                 # Maximum photos per person (1-20)

  # Augmentation
  augmentation: true            # Enable augmentation
  augmentation_strength: medium # light, medium, heavy

  # Face detection
  face_detector: yolov8         # yolov8, retinaface, mtcnn, opencv
  face_expand_percentage: 50    # Crop expansion (0-100)
  face_confidence_threshold: 0.3  # Detection confidence (0.0-1.0)

  # Normalization (ImageNet defaults)
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

#### Augmentation Strength Levels

| Level | Transforms |
|-------|------------|
| `light` | HorizontalFlip, slight ColorJitter |
| `medium` | + RandomResizedCrop, GaussianBlur, GaussNoise |
| `heavy` | + ShiftScaleRotate, MotionBlur, CoarseDropout |

#### Face Detector Options

| Detector | Speed | Accuracy | Notes |
|----------|-------|----------|-------|
| `yolov8` | Fast | High | Default, good balance |
| `retinaface` | Medium | Highest | Best accuracy |
| `mtcnn` | Slow | High | Classic choice |
| `opencv` | Very Fast | Low | Fallback, no dependencies |

### Path Configuration

Controls file and directory locations.

```yaml
paths:
  # Dataset
  dataset_file: datasets/beauty_dataset.csv  # CSV with folder_name, score
  dataset_base_path: datasets/images         # Directory with image folders

  # Output
  output_dir: outputs                         # Training outputs
  export_dir: exports                         # Exported models

  # Resume training
  checkpoint_path: null                       # Path to resume checkpoint
```

### Logging Configuration

Controls logging and experiment tracking.

```yaml
logging:
  # Console logging
  log_level: INFO               # DEBUG, INFO, WARNING, ERROR
  log_dir: logs                 # Directory for log files

  # Experiment tracking
  use_tensorboard: true         # Enable TensorBoard
  use_wandb: false              # Enable Weights & Biases
  wandb_project: beauty-scorer  # W&B project name
  wandb_entity: null            # W&B team/entity

  # Frequency
  log_every_n_steps: 10         # Log metrics every N steps
  save_every_n_epochs: 1        # Save checkpoint every N epochs
```

## Pre-made Configurations

### default.yaml

Balanced configuration for GPU training:

```yaml
model:
  architecture: mobilenet_transformer
  embed_dim: 512
  num_encoder_layers: 4
  unfreeze_backbone_layers: 4

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  use_amp: true

data:
  image_size: [224, 224]
  augmentation_strength: medium
```

### cpu_training.yaml

Optimized for CPU training:

```yaml
model:
  architecture: lightweight_cpu
  backbone: mobilenet_v2
  embed_dim: 256
  num_encoder_layers: 0
  use_cross_attention: false
  use_se_attention: false

training:
  batch_size: 8
  use_amp: false
  gradient_accumulation_steps: 4
  num_workers: 2

data:
  face_size: [112, 112]
  max_photos: 6
  augmentation_strength: light
  face_detector: opencv
```

### advanced_model.yaml

Maximum accuracy configuration:

```yaml
model:
  architecture: vit_arcface
  backbone: dinov2_vits14
  embed_dim: 768
  num_encoder_layers: 6
  use_gradient_checkpointing: true

training:
  batch_size: 16
  learning_rate: 0.00005
  backbone_lr_multiplier: 0.01
  epochs: 100
  patience: 10
  loss_function: focal
  scheduler_warmup_epochs: 5
  gradient_accumulation_steps: 2

data:
  image_size: [384, 384]
  augmentation_strength: heavy
  face_detector: retinaface
```

## CLI Overrides

Override any config value from command line:

```bash
# Override epochs and batch size
python scripts/train.py --config configs/default.yaml \
    --epochs 100 \
    --batch-size 16

# Override model architecture
python scripts/train.py --config configs/default.yaml \
    --model lightweight_cpu

# Override learning rate
python scripts/train.py --config configs/default.yaml \
    --lr 0.00005
```

## Environment Variables

Some settings can be overridden via environment variables:

```bash
# CUDA device
export CUDA_VISIBLE_DEVICES=0,1

# Number of threads
export OMP_NUM_THREADS=4

# Disable AMP
export BEAUTY_SCORER_NO_AMP=1
```

## Validation

Configuration is validated on load:

```python
from beauty_scorer.config import ModelConfig
from pydantic import ValidationError

try:
    config = ModelConfig(embed_dim=10)  # Too small
except ValidationError as e:
    print(e)  # embed_dim must be >= 64
```

## Saving Configuration

Save your configuration for reproducibility:

```python
config = load_config("configs/default.yaml")
config = config.merge_with({"training": {"epochs": 100}})
config.to_yaml("outputs/my_experiment/config.yaml")
```
