# Quickstart Guide

Get started with BeautyScorer in 5 minutes.

## Installation

### From Source

```bash
git clone https://github.com/alessiosavi/BeautyScorer.git
cd BeautyScorer
pip install -e .
```

### With Optional Dependencies

```bash
# All dependencies (dev, tensorboard, wandb, onnx)
pip install -e ".[all]"

# Just development tools
pip install -e ".[dev]"
```

## Inference (Using Pre-trained Model)

### Command Line

```bash
# Single person from photo list
python scripts/infer.py --model model.pt --photos photo1.jpg photo2.jpg photo3.jpg

# Single person from folder
python scripts/infer.py --model model.pt --folder /path/to/person/photos/

# With verbose output
python scripts/infer.py --model model.pt --folder ./photos/ --verbose --top-k 5
```

### Python API

```python
from beauty_scorer import BeautyPredictor

# Initialize predictor
predictor = BeautyPredictor("model.pt", device="auto")

# Predict from file paths
result = predictor.predict_person([
    "photos/person1_front.jpg",
    "photos/person1_side.jpg",
    "photos/person1_full.jpg"
])

print(f"Predicted Score: {result['score']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Top 3 probabilities:")
for score, prob in sorted(result['probabilities'].items(),
                          key=lambda x: x[1], reverse=True)[:3]:
    print(f"  Score {score}: {prob:.1%}")
```

## Training Your Own Model

### 1. Prepare Your Dataset

Create a CSV file with two columns:

```csv
folder_name,score
person_001,7
person_002,5
person_003,8
```

Organize images:

```text
datasets/
├── beauty_dataset.csv
└── images/
    ├── person_001/
    │   ├── photo1.jpg
    │   ├── photo2.jpg
    │   └── photo3.jpg
    ├── person_002/
    │   └── photo1.jpg
    └── person_003/
        ├── photo1.jpg
        └── photo2.jpg
```

### 2. Configure Training

Edit `configs/default.yaml` or create a custom config:

```yaml
model:
  architecture: mobilenet_transformer
  embed_dim: 512
  num_classes: 9

training:
  batch_size: 32
  learning_rate: 0.0001
  epochs: 50
  patience: 5

data:
  image_size: [224, 224]
  max_photos: 9
  augmentation: true

paths:
  dataset_file: datasets/beauty_dataset.csv
  dataset_base_path: datasets/images
  output_dir: outputs
```

### 3. Train

```bash
# With default config
python scripts/train.py --config configs/default.yaml

# Override settings
python scripts/train.py --config configs/default.yaml \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.00005
```

### 4. Monitor Training

Training creates:

```text
outputs/
├── checkpoints/
│   ├── best.pt          # Best model (lowest val loss)
│   ├── last.pt          # Latest checkpoint
│   └── checkpoint_epoch*.pt
├── config.yaml          # Saved configuration
└── logs/                # Training logs
```

View TensorBoard logs:

```bash
tensorboard --logdir outputs/logs
```

## Evaluation

```bash
python scripts/evaluate.py \
    --model outputs/checkpoints/best.pt \
    --data datasets/test.csv \
    --data-path datasets/images \
    --output eval_results \
    --save-predictions
```

This generates:

- `eval_results/metrics.json` - Accuracy, MAE, RMSE
- `eval_results/confusion_matrix.png` - Visualization
- `eval_results/predictions.json` - Per-sample predictions

## Model Export

### ONNX (Cross-platform)

```bash
python scripts/export_model.py --model best.pt --format onnx
```

### TorchScript (PyTorch deployment)

```bash
python scripts/export_model.py --model best.pt --format torchscript
```

### Quantized INT8 (Edge deployment)

```bash
python scripts/export_model.py --model best.pt --format quantized
```

## Choosing a Model Architecture

| Use Case | Architecture | Config |
|----------|--------------|--------|
| General purpose | `mobilenet_transformer` | `default.yaml` |
| Maximum accuracy | `vit_arcface` | `advanced_model.yaml` |
| CPU/Edge devices | `lightweight_cpu` | `cpu_training.yaml` |

```bash
# Train lightweight model for CPU
python scripts/train.py --config configs/cpu_training.yaml

# Train advanced model for best accuracy
python scripts/train.py --config configs/advanced_model.yaml
```

## Next Steps

- [API Reference](API.md) - Complete API documentation
- [Architecture](ARCHITECTURE.md) - Understanding the models
- [Configuration](CONFIGURATION.md) - All configuration options
- [Development Guide](DEVELOPMENT.md) - Contributing
