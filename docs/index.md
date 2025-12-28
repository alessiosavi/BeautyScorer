# BeautyScorer Documentation

Welcome to BeautyScorer, a professional deep learning framework for predicting beauty scores from photos using transformer-based multi-image aggregation.

## Overview

BeautyScorer predicts beauty scores (1-9) for individuals based on multiple photos and face crops using deep learning. The model treats photos as an unordered set, making it robust to photo order.

## Key Features

- **Three Model Architectures**: Choose the right model for your use case
  - MobileNet+Transformer: Balanced accuracy and speed
  - ViT+ArcFace: State-of-the-art accuracy
  - Lightweight CPU: Fast inference on CPU/edge devices

- **Permutation Invariant**: Photo order doesn't affect predictions

- **Production Ready**: Export to ONNX, TorchScript, or quantized INT8

- **Modern Training**: AMP, gradient accumulation, early stopping, callbacks

## Quick Links

- [Quickstart Guide](QUICKSTART.md) - Get started in 5 minutes
- [API Reference](API.md) - Complete API documentation
- [Architecture](ARCHITECTURE.md) - Technical deep dive
- [Configuration](CONFIGURATION.md) - All configuration options
- [Development Guide](DEVELOPMENT.md) - Contributing and development
- [Changelog](CHANGELOG.md) - Version history

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from beauty_scorer import BeautyPredictor

# Load model and predict
predictor = BeautyPredictor("model.pt")
result = predictor.predict_person(["photo1.jpg", "photo2.jpg"])
print(f"Score: {result['score']}, Confidence: {result['confidence']:.1%}")
```

## License

MIT License - see [LICENSE](../LICENSE) for details.
