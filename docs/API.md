# API Reference

Complete API documentation for BeautyScorer.

## Table of Contents

- [Configuration](#configuration)
- [Models](#models)
- [Training](#training)
- [Inference](#inference)
- [Data](#data)
- [Utilities](#utilities)

---

## Configuration

### `beauty_scorer.config`

#### `BeautyConfig`

Main configuration class combining all sub-configurations.

```python
from beauty_scorer.config import BeautyConfig, load_config

# Load from YAML
config = load_config("configs/default.yaml")

# Create with defaults
config = BeautyConfig()

# Access sub-configs
print(config.model.architecture)
print(config.training.batch_size)
print(config.data.image_size)

# Save to YAML
config.to_yaml("my_config.yaml")

# Merge with overrides
new_config = config.merge_with({"training": {"epochs": 100}})
```

#### `ModelConfig`

Model architecture configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `architecture` | str | `"mobilenet_transformer"` | Model architecture |
| `backbone` | str | `"mobilenet_v3_large"` | Backbone network |
| `embed_dim` | int | `512` | Embedding dimension |
| `num_encoder_layers` | int | `4` | Transformer layers |
| `num_heads` | int | `8` | Attention heads |
| `ff_dim` | int | `2048` | Feed-forward dimension |
| `dropout` | float | `0.2` | Dropout rate |
| `num_classes` | int | `9` | Number of score classes |
| `unfreeze_backbone_layers` | int | `0` | Layers to unfreeze |
| `use_positional_encoding` | bool | `False` | Add positional encoding |
| `use_cross_attention` | bool | `True` | Cross-attention fusion |
| `use_se_attention` | bool | `True` | SE attention in head |
| `use_gradient_checkpointing` | bool | `False` | Memory optimization |
| `pretrained` | bool | `True` | Use pretrained weights |

#### `TrainingConfig`

Training process configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | `32` | Training batch size |
| `learning_rate` | float | `1e-4` | Initial learning rate |
| `backbone_lr_multiplier` | float | `0.1` | Backbone LR multiplier |
| `weight_decay` | float | `1e-2` | Weight decay |
| `epochs` | int | `50` | Number of epochs |
| `patience` | int | `5` | Early stopping patience |
| `label_smoothing` | float | `0.1` | Label smoothing |
| `use_amp` | bool | `True` | Mixed precision training |
| `gradient_accumulation_steps` | int | `1` | Gradient accumulation |
| `max_grad_norm` | float | `1.0` | Gradient clipping |
| `loss_function` | str | `"cross_entropy"` | Loss function |
| `scheduler` | str | `"cosine"` | LR scheduler |
| `val_split` | float | `0.1` | Validation split ratio |
| `seed` | int | `42` | Random seed |
| `num_workers` | int | `4` | Data loading workers |

#### `DataConfig`

Data processing configuration.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_size` | tuple | `(224, 224)` | Input image size |
| `face_size` | tuple | `(224, 224)` | Face crop size |
| `max_photos` | int | `9` | Max photos per person |
| `augmentation` | bool | `True` | Enable augmentation |
| `augmentation_strength` | str | `"medium"` | Augmentation level |
| `face_detector` | str | `"yolov8"` | Face detection backend |
| `face_expand_percentage` | int | `50` | Face crop expansion |
| `face_confidence_threshold` | float | `0.3` | Detection threshold |

---

## Models

### `beauty_scorer.models`

#### `create_model`

Factory function to create models.

```python
from beauty_scorer.models import create_model

# Create with architecture name
model = create_model("mobilenet_transformer")

# Create with config
from beauty_scorer.config import ModelConfig
config = ModelConfig(embed_dim=256, num_encoder_layers=2)
model = create_model(config=config)

# Load pretrained weights
model = create_model(
    "mobilenet_transformer",
    pretrained_path="checkpoint.pt",
    device="cuda"
)
```

#### `BaseBeautyModel`

Abstract base class for all models.

```python
class BaseBeautyModel(nn.Module):
    def forward(
        self,
        photos: torch.Tensor,      # (batch, max_photos, C, H, W)
        photos_mask: torch.Tensor, # (batch, max_photos)
        faces: torch.Tensor,       # (batch, max_photos, C, H, W)
        faces_mask: torch.Tensor   # (batch, max_photos)
    ) -> torch.Tensor:             # (batch, num_classes)
        """Forward pass returning logits."""
        pass

    def predict(
        self,
        photos, photos_mask, faces, faces_mask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_scores, probabilities)."""
        pass

    def get_trainable_params(self) -> list[dict]:
        """Get parameter groups for optimizer."""
        pass

    def count_parameters(self, trainable_only=True) -> int:
        """Count model parameters."""
        pass

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        pass

    @classmethod
    def load(cls, path: str, device=None) -> "BaseBeautyModel":
        """Load model from checkpoint."""
        pass
```

#### `MobileNetTransformerModel`

Default architecture with MobileNetV3 + Transformer.

```python
from beauty_scorer.models import MobileNetTransformerModel
from beauty_scorer.config import ModelConfig

config = ModelConfig(
    architecture="mobilenet_transformer",
    backbone="mobilenet_v3_large",
    embed_dim=512,
    num_encoder_layers=4,
    use_cross_attention=True,
    use_se_attention=True
)

model = MobileNetTransformerModel(config)
```

#### `ViTArcFaceModel`

Advanced architecture with Vision Transformer.

```python
from beauty_scorer.models import ViTArcFaceModel
from beauty_scorer.config import ModelConfig

config = ModelConfig(
    architecture="vit_arcface",
    backbone="dinov2_vits14",
    embed_dim=768,
    num_encoder_layers=6
)

model = ViTArcFaceModel(config)
```

#### `LightweightCPUModel`

CPU-optimized lightweight model.

```python
from beauty_scorer.models import LightweightCPUModel
from beauty_scorer.config import ModelConfig

config = ModelConfig(
    architecture="lightweight_cpu",
    backbone="mobilenet_v2",
    embed_dim=256,
    num_encoder_layers=1  # No transformer for speed
)

model = LightweightCPUModel(config)
```

---

## Training

### `beauty_scorer.training`

#### `Trainer`

Main training class with full training loop.

```python
from beauty_scorer.training import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
    callbacks=[...],
    class_weights=weights
)

# Train
results = trainer.fit(epochs=50)

# Resume from checkpoint
results = trainer.fit(resume_from="checkpoint.pt")

# Access history
print(trainer.history)  # {"train_loss": [...], "val_loss": [...], ...}
```

#### Callbacks

```python
from beauty_scorer.training import (
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    LRMonitor
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best=True
    ),
    ModelCheckpoint(
        save_dir="checkpoints",
        monitor="val_loss",
        save_best_only=True
    ),
    ProgressBar(show_batch_progress=True),
    LRMonitor(log_every_n_steps=100)
]
```

#### Loss Functions

```python
from beauty_scorer.training import get_loss_function

# Cross entropy with label smoothing
loss_fn = get_loss_function(
    "cross_entropy",
    label_smoothing=0.1,
    class_weights=weights
)

# Focal loss for class imbalance
loss_fn = get_loss_function(
    "focal",
    focal_gamma=2.0,
    class_weights=weights
)

# Ordinal regression
loss_fn = get_loss_function("ordinal", num_classes=9)
```

#### Metrics

```python
from beauty_scorer.training import compute_metrics, MetricTracker

# One-shot computation
metrics = compute_metrics(predictions, targets, num_classes=9)
# Returns: accuracy, mae, rmse, within_1, within_2, confusion_matrix

# Incremental tracking
tracker = MetricTracker(num_classes=9)
for batch in loader:
    tracker.update(preds, targets, loss)
metrics = tracker.compute()
confusion = tracker.get_confusion_matrix()
```

---

## Inference

### `beauty_scorer.inference`

#### `BeautyPredictor`

High-level inference API.

```python
from beauty_scorer import BeautyPredictor

# Initialize
predictor = BeautyPredictor(
    model_path="model.pt",
    device="auto"  # auto, cuda, cpu, mps
)

# Predict from paths
result = predictor.predict_person(["photo1.jpg", "photo2.jpg"])
# Returns:
# {
#     "score": 7,
#     "probabilities": {1: 0.01, 2: 0.02, ..., 9: 0.05},
#     "confidence": 0.45
# }

# Predict from numpy arrays
import numpy as np
images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)]
result = predictor.predict_from_images(images)

# Batch prediction
results = predictor.predict_batch([
    ["person1_1.jpg", "person1_2.jpg"],
    ["person2_1.jpg", "person2_2.jpg"]
])

# Get top-k predictions
top_k = predictor.get_top_predictions(photos, top_k=3)
# Returns: [(7, 0.45), (6, 0.30), (8, 0.15)]

# Detailed explanation
explanation = predictor.explain_prediction(photos)
```

#### Model Export

```python
from beauty_scorer.inference import (
    export_to_onnx,
    export_to_torchscript,
    export_to_quantized,
    get_model_size
)

# ONNX export
export_to_onnx(
    model,
    "model.onnx",
    max_photos=9,
    image_size=(224, 224),
    opset_version=17
)

# TorchScript export
export_to_torchscript(model, "model.pt")

# Quantized export
export_to_quantized(model, "model_int8.pt")

# Get model size info
info = get_model_size(model)
print(f"Size: {info['saved_size_mb']:.1f} MB")
print(f"Params: {info['total_params']:,}")
```

#### ONNX Inference

```python
from beauty_scorer.inference.export import ONNXPredictor

predictor = ONNXPredictor(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

logits = predictor.predict(photos, photos_mask, faces, faces_mask)
```

---

## Data

### `beauty_scorer.data`

#### Dataset Loading

```python
from beauty_scorer.data import (
    load_dataset_from_csv,
    train_val_split,
    create_data_loaders,
    compute_class_weights
)

# Load dataset
data = load_dataset_from_csv(
    csv_path="dataset.csv",
    base_path="images/"
)

# Split
train_data, val_data = train_val_split(
    data,
    val_ratio=0.1,
    stratify=True,
    seed=42
)

# Create loaders
train_loader, val_loader = create_data_loaders(
    train_data,
    val_data,
    batch_size=32,
    num_workers=4
)

# Class weights
weights = compute_class_weights(train_data, num_classes=9)
```

#### BeautyDataset

```python
from beauty_scorer.data import BeautyDataset
from torch.utils.data import DataLoader

dataset = BeautyDataset(
    data,
    config=data_config,
    is_training=True,
    cache_faces=False
)

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=BeautyDataset.collate_fn
)
```

#### Transforms

```python
from beauty_scorer.data import (
    get_train_transforms,
    get_val_transforms,
    get_inference_transforms
)

# Training with augmentation
train_transform = get_train_transforms(
    image_size=(224, 224),
    augmentation_strength="medium"  # light, medium, heavy
)

# Validation (no augmentation)
val_transform = get_val_transforms(image_size=(224, 224))
```

#### Face Extraction

```python
from beauty_scorer.data import FaceExtractor, extract_face

# Single extraction
face = extract_face(
    image,
    backend="yolov8",  # yolov8, retinaface, mtcnn, opencv
    expand_percentage=50,
    confidence_threshold=0.3
)

# Batch extraction
extractor = FaceExtractor(backend="yolov8")
faces = extractor.extract_batch(images)
```

---

## Utilities

### `beauty_scorer.utils`

#### Device Management

```python
from beauty_scorer.utils import (
    get_device,
    get_device_info,
    set_seed,
    DeviceManager
)

# Get best device
device = get_device("auto")  # auto, cuda, cpu, mps

# Device info
info = get_device_info(device)
print(f"Name: {info.device_name}")
print(f"Memory: {info.memory_total / 1e9:.1f} GB")

# Set seed for reproducibility
set_seed(42, deterministic=True)

# Context manager
dm = DeviceManager(device="auto", seed=42)
photos, faces = dm.to_device(photos, faces)
```

#### Logging

```python
from beauty_scorer.utils import setup_logging, get_logger

# Setup
setup_logging(
    log_level="INFO",
    log_dir="logs"
)

# Get logger
logger = get_logger(__name__)
logger.info("Training started")
```

#### Visualization

```python
from beauty_scorer.utils import (
    plot_training_history,
    plot_confusion_matrix,
    visualize_predictions,
    show_image
)

# Training curves
plot_training_history(
    {"train_loss": [...], "val_loss": [...]},
    save_path="history.png"
)

# Confusion matrix
plot_confusion_matrix(
    matrix,
    class_names=["1", "2", ..., "9"],
    normalize=True,
    save_path="confusion.png"
)

# Prediction visualization
visualize_predictions(
    images,
    predictions,
    true_labels,
    save_path="predictions.png"
)
```
