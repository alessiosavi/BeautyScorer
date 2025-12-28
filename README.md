# BeautyScorer

A professional deep learning framework for predicting beauty scores (1-9) from photos using transformer-based multi-image aggregation.

## Features

- **Three Model Architectures**: MobileNet+Transformer (balanced), ViT+ArcFace (SOTA), Lightweight CPU (efficient)
- **Permutation Invariant**: Photo order doesn't affect predictions
- **Variable-Length Inputs**: Handle 1-9 photos per person with automatic masking
- **Modern Training Pipeline**: AMP, gradient accumulation, callbacks, early stopping
- **Production Ready**: ONNX/TorchScript export, quantization support
- **Type-Safe Configuration**: Pydantic-based config with YAML support

## Installation

```bash
# Clone the repository
git clone https://github.com/alessiosavi/BeautyScorer.git
cd BeautyScorer

# Install in development mode
pip install -e .

# Or install with all extras
pip install -e ".[all]"
```

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+ (optional, for GPU training)

## Quick Start

### Training

```bash
# Train with default configuration (MobileNet+Transformer)
python scripts/train.py --config configs/default.yaml

# Train CPU-optimized model
python scripts/train.py --config configs/cpu_training.yaml

# Train advanced ViT model
python scripts/train.py --config configs/advanced_model.yaml
```

### Inference

```bash
# Predict from photos
python scripts/infer.py --model outputs/checkpoints/best.pt --photos photo1.jpg photo2.jpg

# Predict from folder
python scripts/infer.py --model outputs/checkpoints/best.pt --folder /path/to/photos/
```

### Python API

```python
from beauty_scorer import BeautyPredictor, create_model, Trainer
from beauty_scorer.config import load_config

# Inference
predictor = BeautyPredictor("model.pt")
result = predictor.predict_person(["photo1.jpg", "photo2.jpg"])
print(f"Score: {result['score']}, Confidence: {result['confidence']:.1%}")

# Training
config = load_config("configs/default.yaml")
model = create_model(config=config.model)
trainer = Trainer(model, train_loader, val_loader, config=config)
results = trainer.fit()
```

## Model Architectures

### 1. MobileNet + Transformer (Default)

Balanced architecture for most use cases:

- MobileNetV3-Large backbone with optional fine-tuning
- Transformer encoders for sequence aggregation
- Cross-attention fusion between photo and face streams
- SE-attention in classification head

```yaml
model:
  architecture: mobilenet_transformer
  embed_dim: 512
  num_encoder_layers: 4
```

### 2. ViT + ArcFace (Advanced)

State-of-the-art for maximum accuracy:

- DINOv2 or ViT backbone with self-supervised features
- Attention-based pooling
- Higher resolution support (384x384)

```yaml
model:
  architecture: vit_arcface
  backbone: dinov2_vits14
  embed_dim: 768
```

### 3. Lightweight CPU (Efficient)

Optimized for CPU training and inference:

- MobileNetV2/EfficientNet-Lite backbone (no SE blocks)
- Simple pooling instead of transformers
- INT8 quantization support

```yaml
model:
  architecture: lightweight_cpu
  backbone: mobilenet_v2
  embed_dim: 256
```

## Model Comparison

| Model | Params | Speed (GPU) | Speed (CPU) | Best For |
|-------|--------|-------------|-------------|----------|
| MobileNet+Transformer | ~15M | Fast | Medium | General use |
| ViT+ArcFace | ~90M | Medium | Slow | Maximum accuracy |
| Lightweight CPU | ~5M | Very Fast | Fast | Edge deployment |

## Configuration

All settings are controlled via YAML files with Pydantic validation:

```yaml
model:
  architecture: mobilenet_transformer
  backbone: mobilenet_v3_large
  embed_dim: 512
  num_encoder_layers: 4
  num_heads: 8
  dropout: 0.2
  num_classes: 9
  use_cross_attention: true
  use_se_attention: true

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  patience: 5
  label_smoothing: 0.1
  use_amp: true

data:
  image_size: [224, 224]
  max_photos: 9
  augmentation: true
  face_detector: yolov8
```

## Project Structure

```text
beauty_scorer/
├── pyproject.toml          # Package configuration
├── configs/                # YAML configuration files
│   ├── default.yaml
│   ├── cpu_training.yaml
│   └── advanced_model.yaml
├── beauty_scorer/          # Main package
│   ├── config.py           # Pydantic configuration
│   ├── data/               # Dataset and preprocessing
│   ├── models/             # Model architectures
│   │   ├── components/     # Encoders, heads, backbones
│   │   ├── mobilenet_transformer.py
│   │   ├── vit_arcface.py
│   │   └── lightweight_cnn.py
│   ├── training/           # Training pipeline
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── callbacks.py
│   ├── inference/          # Inference and export
│   └── utils/              # Utilities
├── scripts/                # CLI scripts
└── tests/                  # Unit tests
```

## Dataset Format

The dataset CSV should have columns:

- `folder_name`: Directory name containing person's photos
- `score`: Beauty score (1-9)

```csv
folder_name,score
person_001,7
person_002,5
person_003,8
```

Images are loaded from: `{base_path}/{folder_name}/*.jpg`

## Model Export

```bash
# Export to ONNX
python scripts/export_model.py --model outputs/best.pt --format onnx

# Export to TorchScript
python scripts/export_model.py --model outputs/best.pt --format torchscript

# Export quantized model
python scripts/export_model.py --model outputs/best.pt --format quantized
```

## Evaluation

```bash
python scripts/evaluate.py \
  --model outputs/best.pt \
  --data datasets/test.csv \
  --data-path datasets/images \
  --output eval_results
```

Outputs:

- `metrics.json`: Accuracy, MAE, RMSE, within-1/2 accuracy
- `confusion_matrix.png`: Visualization
- `predictions.json`: Per-sample predictions

## Key Features Explained

### Permutation Invariance

The model treats photos as an unordered set - shuffling photos produces the same result:

```python
model.eval()
output1 = model(photos, mask, faces, face_mask)
output2 = model(photos[:, perm], mask[:, perm], faces[:, perm], face_mask[:, perm])
assert torch.allclose(output1, output2)  # True!
```

### Cross-Attention Fusion

Photo and face streams attend to each other before final aggregation, allowing the model to learn correlations between full-body appearance and facial features.

### Class Weights

Automatically computed inverse-frequency weights handle imbalanced score distributions:

```python
from beauty_scorer.data.dataset import compute_class_weights
weights = compute_class_weights(train_data, num_classes=9)
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ --cov=beauty_scorer --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run `pytest` and `ruff check`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{beautyscorer2025,
  title = {BeautyScorer: Transformer-based Beauty Score Prediction},
  year = {2025},
  url = {https://github.com/alessiosvi/BeautyScorer}
}
```

## Acknowledgments

- Built with PyTorch, timm, and Albumentations
- Face detection powered by DeepFace
- Inspired by transformer-based set aggregation methods
