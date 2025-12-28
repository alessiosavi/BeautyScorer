# Development Guide

Guide for contributing to and developing BeautyScorer.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) CUDA 11.8+ for GPU support

### Installation

```bash
# Clone repository
git clone https://github.com/alessiosavi/BeautyScorer.git
cd BeautyScorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check code style
ruff check beauty_scorer/

# Type checking
mypy beauty_scorer/
```

## Project Structure

```text
beauty_scorer/
├── beauty_scorer/          # Main package
│   ├── __init__.py         # Package exports
│   ├── config.py           # Pydantic configuration
│   ├── data/               # Data loading and processing
│   │   ├── dataset.py      # Dataset classes
│   │   ├── transforms.py   # Data augmentation
│   │   └── preprocessing.py # Face extraction
│   ├── models/             # Neural network models
│   │   ├── base.py         # Abstract base class
│   │   ├── factory.py      # Model factory
│   │   ├── mobilenet_transformer.py
│   │   ├── vit_arcface.py
│   │   ├── lightweight_cnn.py
│   │   └── components/     # Reusable components
│   │       ├── encoders.py # Transformers, attention
│   │       ├── heads.py    # Classification heads
│   │       └── backbones.py # Feature extractors
│   ├── training/           # Training pipeline
│   │   ├── trainer.py      # Main trainer class
│   │   ├── losses.py       # Loss functions
│   │   ├── optimizers.py   # Optimizer factories
│   │   ├── metrics.py      # Evaluation metrics
│   │   └── callbacks.py    # Training callbacks
│   ├── inference/          # Inference and export
│   │   ├── predictor.py    # High-level API
│   │   └── export.py       # Model export
│   └── utils/              # Utilities
│       ├── logging.py      # Logging setup
│       ├── device.py       # Device management
│       └── visualization.py # Plotting
├── configs/                # YAML configurations
├── scripts/                # CLI scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
├── pyproject.toml          # Package configuration
└── README.md
```

## Coding Standards

### Style Guide

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use double quotes for strings
- Type hints for all public functions

```python
# Good
def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 9,
) -> dict[str, float]:
    """Compute evaluation metrics."""
    ...

# Bad
def compute_metrics(predictions, targets, num_classes=9):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def train_epoch(self, epoch: int) -> dict[str, float]:
    """
    Train for one epoch.

    Args:
        epoch: Current epoch number.

    Returns:
        Dictionary of training metrics including:
        - loss: Average training loss
        - accuracy: Training accuracy

    Raises:
        RuntimeError: If model is not in training mode.
    """
```

### Type Hints

All public functions should have type hints:

```python
from typing import Optional
import torch

def forward(
    self,
    photos: torch.Tensor,
    photos_mask: torch.Tensor,
    faces: torch.Tensor,
    faces_mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run tests matching pattern
pytest tests/ -k "test_forward" -v

# Run with coverage
pytest tests/ --cov=beauty_scorer --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow" -v
```

### Writing Tests

```python
import pytest
import torch

class TestMobileNetTransformer:
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        config = ModelConfig(embed_dim=256)
        return MobileNetTransformerModel(config)

    def test_forward_shape(self, model):
        """Test output shape."""
        batch_size = 2
        photos = torch.randn(batch_size, 4, 3, 224, 224)
        photos_mask = torch.ones(batch_size, 4, dtype=torch.bool)
        faces = torch.randn(batch_size, 4, 3, 224, 224)
        faces_mask = torch.ones(batch_size, 4, dtype=torch.bool)

        output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, 9)

    @pytest.mark.slow
    def test_training_step(self, model):
        """Test one training step."""
        # Long-running test
        ...
```

### Test Categories

- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Slow tests**: Mark with `@pytest.mark.slow`

## Adding New Features

### Adding a New Model Architecture

1. Create model file in `beauty_scorer/models/`:

```python
# beauty_scorer/models/new_model.py
from beauty_scorer.models.base import BaseBeautyModel

class NewModel(BaseBeautyModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize layers

    def forward(self, photos, photos_mask, faces, faces_mask):
        # Implement forward pass
        pass

    def get_trainable_params(self):
        # Return parameter groups
        pass
```

2. Register in factory:

```python
# beauty_scorer/models/factory.py
from beauty_scorer.models.new_model import NewModel

class ModelFactory:
    _models = {
        "mobilenet_transformer": MobileNetTransformerModel,
        "new_model": NewModel,  # Add here
        ...
    }
```

3. Add to config:

```python
# beauty_scorer/config.py
class ModelConfig(BaseModel):
    architecture: Literal[
        "mobilenet_transformer",
        "vit_arcface",
        "lightweight_cpu",
        "new_model"  # Add here
    ]
```

4. Write tests:

```python
# tests/test_models.py
class TestNewModel:
    def test_forward(self):
        ...
```

### Adding a New Backbone

1. Register in backbone registry:

```python
# beauty_scorer/models/components/backbones.py

@BackboneRegistry.register(
    "new_backbone",
    feature_dim=512,
    input_size=224,
    family="custom",
    description="My new backbone"
)
def _new_backbone(pretrained=True):
    # Return backbone module
    model = ...
    return model.features
```

### Adding a New Loss Function

1. Implement in losses.py:

```python
# beauty_scorer/training/losses.py

class NewLoss(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, logits, targets):
        ...
```

2. Register in factory:

```python
def get_loss_function(loss_type, ...):
    if loss_type == "new_loss":
        return NewLoss(...)
```

## Debugging

### Debug Scripts

Use the provided debug scripts:

```bash
# Quick sanity check
python debug_quick.py

# Interactive debugging
python debug_notebook.py
```

### Common Issues

**CUDA Out of Memory**:

```python
# Reduce batch size
config.training.batch_size = 8

# Enable gradient checkpointing
config.model.use_gradient_checkpointing = True

# Use gradient accumulation
config.training.gradient_accumulation_steps = 4
```

**Slow Training**:

```python
# Enable AMP
config.training.use_amp = True

# Increase workers
config.training.num_workers = 8

# Pin memory
config.training.pin_memory = True
```

**Poor Convergence**:

```python
# Lower learning rate
config.training.learning_rate = 1e-5

# Add warmup
config.training.scheduler_warmup_epochs = 3

# Try different loss
config.training.loss_function = "focal"
```

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** following coding standards
4. **Add tests** for new functionality
5. **Run tests**: `pytest tests/ -v`
6. **Run linter**: `ruff check beauty_scorer/`
7. **Commit**: `git commit -m "Add my feature"`
8. **Push**: `git push origin feature/my-feature`
9. **Create PR** with description of changes

### PR Checklist

- [ ] Tests pass (`pytest tests/`)
- [ ] Code style OK (`ruff check beauty_scorer/`)
- [ ] Type hints added for public functions
- [ ] Docstrings added for new functions
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated

## Release Process

1. Update version in `pyproject.toml`
2. Update `docs/CHANGELOG.md`
3. Create git tag: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. GitHub Actions will build and publish

## Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions
- **Email**: <alessiosavibtc@gmail.com>
