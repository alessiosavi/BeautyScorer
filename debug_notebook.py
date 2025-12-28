#!/usr/bin/env python3
"""
Interactive debugging notebook for BeautyScorer.

This script provides an interactive environment for debugging and exploring
BeautyScorer components. Run it in an interactive Python session or use
as a starting point for Jupyter notebooks.

Usage:
    python -i debug_notebook.py   # Interactive mode
    python debug_notebook.py      # Run all demos
    ipython -i debug_notebook.py  # IPython interactive mode
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure we're using the local package
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# Setup and Imports
# =============================================================================

print("Loading BeautyScorer components...")

from beauty_scorer.config import BeautyConfig, ModelConfig, TrainingConfig
from beauty_scorer.inference.export import get_model_size
from beauty_scorer.models.factory import ModelFactory, create_model
from beauty_scorer.training.callbacks import EarlyStopping, ModelCheckpoint
from beauty_scorer.training.losses import (
    CrossEntropyLoss,
    FocalLoss,
    OrdinalRegressionLoss,
    get_loss_function,
)
from beauty_scorer.training.metrics import MetricTracker, compute_metrics
from beauty_scorer.training.optimizers import create_optimizer
from beauty_scorer.utils.device import get_device, set_seed

print("All imports successful!")

# Set seed for reproducibility
set_seed(42)

# Get device
DEVICE = get_device()
print(f"Using device: {DEVICE}")


# =============================================================================
# Helper Functions
# =============================================================================


def create_dummy_batch(
    batch_size: int = 2,
    max_photos: int = 4,
    num_classes: int = 9,
    image_size: int = 224,
) -> dict:
    """Create a dummy batch for testing."""
    return {
        "photos": torch.randn(batch_size, max_photos, 3, image_size, image_size),
        "photos_mask": torch.ones(batch_size, max_photos, dtype=torch.bool),
        "faces": torch.randn(batch_size, max_photos, 3, image_size, image_size),
        "faces_mask": torch.ones(batch_size, max_photos, dtype=torch.bool),
        "targets": torch.randint(0, num_classes, (batch_size,)),
    }


def print_model_summary(model: nn.Module) -> None:
    """Print model summary with parameter counts."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nModel Summary:")
    print(f"  Architecture: {model.__class__.__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Size (MB): {total_params * 4 / 1024 / 1024:.2f}")


def benchmark_model(
    model: nn.Module,
    batch_size: int = 1,
    max_photos: int = 4,
    num_iterations: int = 10,
    warmup: int = 3,
) -> dict:
    """Benchmark model inference speed."""
    import time

    model.eval()
    batch = create_dummy_batch(batch_size, max_photos)

    # Move to device
    photos = batch["photos"].to(DEVICE)
    photos_mask = batch["photos_mask"].to(DEVICE)
    faces = batch["faces"].to(DEVICE)
    faces_mask = batch["faces_mask"].to(DEVICE)
    model = model.to(DEVICE)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(photos, photos_mask, faces, faces_mask)

    # Benchmark
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(photos, photos_mask, faces, faces_mask)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "fps": 1.0 / np.mean(times),
    }


# =============================================================================
# Demo Functions
# =============================================================================


def demo_config():
    """Demonstrate configuration system."""
    print("\n" + "=" * 60)
    print(" Configuration System Demo")
    print("=" * 60)

    # Default config
    print("\n1. Default Configuration:")
    config = BeautyConfig()
    print(f"   Model architecture: {config.model.architecture}")
    print(f"   Embed dim: {config.model.embed_dim}")
    print(f"   Batch size: {config.training.batch_size}")
    print(f"   Image size: {config.data.image_size}")

    # Custom config
    print("\n2. Custom Configuration:")
    custom_config = BeautyConfig(
        model=ModelConfig(
            architecture="lightweight_cpu",
            embed_dim=128,
            num_encoder_layers=1,
        ),
        training=TrainingConfig(
            batch_size=16,
            learning_rate=1e-4,
        ),
    )
    print(f"   Model architecture: {custom_config.model.architecture}")
    print(f"   Embed dim: {custom_config.model.embed_dim}")
    print(f"   Batch size: {custom_config.training.batch_size}")

    # Merge configs
    print("\n3. Config Merging:")
    base = BeautyConfig()
    merged = base.merge_with(
        {
            "model": {"embed_dim": 256},
            "training": {"epochs": 100},
        }
    )
    print(f"   Base embed_dim: {base.model.embed_dim}")
    print(f"   Merged embed_dim: {merged.model.embed_dim}")
    print(f"   Merged epochs: {merged.training.epochs}")

    return config


def demo_models():
    """Demonstrate model creation and inference."""
    print("\n" + "=" * 60)
    print(" Model Creation Demo")
    print("=" * 60)

    # List available architectures
    print("\n1. Available Architectures:")
    for arch in ModelFactory.list_available():
        print(f"   - {arch}")

    # Create each architecture
    print("\n2. Creating Models:")

    models = {}
    configs = {
        "lightweight_cpu": ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=64,
            num_encoder_layers=1,
        ),
        "mobilenet_transformer": ModelConfig(
            architecture="mobilenet_transformer",
            backbone="mobilenet_v2",
            embed_dim=128,
            num_encoder_layers=1,
        ),
        "vit_arcface": ModelConfig(
            architecture="vit_arcface",
            backbone="mobilenet_v2",
            embed_dim=128,
            num_encoder_layers=1,
        ),
    }

    for name, config in configs.items():
        model = create_model(config=config)
        models[name] = model
        print_model_summary(model)

    # Test forward pass
    print("\n3. Forward Pass Test:")
    batch = create_dummy_batch()

    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(
                batch["photos"],
                batch["photos_mask"],
                batch["faces"],
                batch["faces_mask"],
            )
        print(f"   {name}: output shape = {output.shape}")

    return models


def demo_training_step():
    """Demonstrate a single training step."""
    print("\n" + "=" * 60)
    print(" Training Step Demo")
    print("=" * 60)

    # Create model and optimizer
    config = ModelConfig(
        architecture="lightweight_cpu",
        backbone="mobilenet_v2",
        embed_dim=64,
        num_encoder_layers=1,
    )
    model = create_model(config=config)
    model.to(DEVICE)

    optimizer = create_optimizer(model)
    criterion = get_loss_function("cross_entropy")

    # Create batch
    batch = create_dummy_batch()
    photos = batch["photos"].to(DEVICE)
    photos_mask = batch["photos_mask"].to(DEVICE)
    faces = batch["faces"].to(DEVICE)
    faces_mask = batch["faces_mask"].to(DEVICE)
    targets = batch["targets"].to(DEVICE)

    # Training step
    print("\n1. Single Training Step:")
    model.train()
    optimizer.zero_grad()

    logits = model(photos, photos_mask, faces, faces_mask)
    loss = criterion(logits, targets)

    print(f"   Logits shape: {logits.shape}")
    print(f"   Loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()

    # Check predictions
    print("\n2. Predictions:")
    predictions = logits.argmax(dim=-1)
    print(f"   Predicted classes: {predictions.tolist()}")
    print(f"   Target classes: {targets.tolist()}")

    return model, optimizer


def demo_metrics():
    """Demonstrate metrics computation."""
    print("\n" + "=" * 60)
    print(" Metrics Demo")
    print("=" * 60)

    # Create sample predictions
    predictions = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    targets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 8])  # One off by 1

    # Compute metrics
    print("\n1. compute_metrics():")
    metrics = compute_metrics(predictions, targets, num_classes=9)
    for name, value in metrics.items():
        if name != "confusion_matrix":
            print(f"   {name}: {value:.4f}")

    # Metric tracker
    print("\n2. MetricTracker (batched):")
    tracker = MetricTracker(num_classes=9)

    for i in range(5):
        preds = torch.randint(0, 9, (10,))
        targs = torch.randint(0, 9, (10,))
        tracker.update(preds, targs, loss=0.5 - i * 0.05)

    tracked_metrics = tracker.compute()
    for name, value in tracked_metrics.items():
        print(f"   {name}: {value:.4f}")

    return metrics


def demo_losses():
    """Demonstrate different loss functions."""
    print("\n" + "=" * 60)
    print(" Loss Functions Demo")
    print("=" * 60)

    batch_size = 4
    num_classes = 9

    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    print("\n1. Standard Cross-Entropy:")
    ce_loss = CrossEntropyLoss()
    print(f"   Loss: {ce_loss(logits, targets).item():.4f}")

    print("\n2. Focal Loss (gamma=2.0):")
    focal_loss = FocalLoss(gamma=2.0)
    print(f"   Loss: {focal_loss(logits, targets).item():.4f}")

    print("\n3. Ordinal Regression Loss:")
    ordinal_logits = torch.randn(batch_size, num_classes - 1)
    ordinal_loss = OrdinalRegressionLoss(num_classes=num_classes)
    print(f"   Loss: {ordinal_loss(ordinal_logits, targets).item():.4f}")


def demo_callbacks():
    """Demonstrate training callbacks."""
    print("\n" + "=" * 60)
    print(" Callbacks Demo")
    print("=" * 60)

    # Mock trainer
    class MockTrainer:
        stop_training = False
        model = nn.Linear(10, 10)
        optimizer = None
        scheduler = None
        current_epoch = 0

    trainer = MockTrainer()

    # Early stopping
    print("\n1. EarlyStopping:")
    es = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    es.on_train_begin(trainer)

    losses = [1.0, 0.9, 0.95, 0.96, 0.97]  # Improvement then plateau
    for epoch, loss in enumerate(losses):
        es.on_epoch_end(trainer, epoch, {"val_loss": loss})
        print(f"   Epoch {epoch}: val_loss={loss:.2f}, stop={trainer.stop_training}")
        if trainer.stop_training:
            break

    # Model checkpoint
    print("\n2. ModelCheckpoint:")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = ModelCheckpoint(
            save_dir=tmpdir,
            monitor="val_loss",
            save_best_only=True,
        )

        trainer.stop_training = False
        trainer.model = nn.Linear(10, 10)
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters())

        checkpoint.on_train_begin(trainer)

        losses = [1.0, 0.9, 0.8, 0.85]
        for epoch, loss in enumerate(losses):
            trainer.current_epoch = epoch
            checkpoint.on_epoch_end(trainer, epoch, {"val_loss": loss})
            best_path = Path(tmpdir) / "best.pt"
            print(f"   Epoch {epoch}: val_loss={loss:.2f}, saved={best_path.exists()}")


def demo_export():
    """Demonstrate model export."""
    print("\n" + "=" * 60)
    print(" Model Export Demo")
    print("=" * 60)

    # Create lightweight model
    config = ModelConfig(
        architecture="lightweight_cpu",
        backbone="mobilenet_v2",
        embed_dim=64,
        num_encoder_layers=1,
    )
    model = create_model(config=config)

    # Model size
    print("\n1. Model Size Info:")
    size_info = get_model_size(model)
    for key, value in size_info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value:,}")

    # # Export to TorchScript
    # print("\n2. TorchScript Export:")
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     export_path = Path(tmpdir) / "model.pt"
    #     export_to_torchscript(model, export_path, max_photos=4)

    #     # Load and verify
    #     loaded = torch.jit.load(str(export_path))
    #     batch = create_dummy_batch()

    #     with torch.no_grad():
    #         output = loaded(
    #             batch["photos"],
    #             batch["photos_mask"],
    #             batch["faces"],
    #             batch["faces_mask"],
    #         )
    #     print(f"   Export path: {export_path}")
    #     print(f"   Output shape: {output.shape}")
    #     print(f"   File size: {export_path.stat().st_size / 1024 / 1024:.2f} MB")


def demo_benchmark():
    """Benchmark different architectures."""
    print("\n" + "=" * 60)
    print(" Benchmark Demo")
    print("=" * 60)

    configs = {
        "lightweight_cpu": ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=64,
            num_encoder_layers=1,
        ),
        "mobilenet_transformer": ModelConfig(
            architecture="mobilenet_transformer",
            backbone="mobilenet_v2",
            embed_dim=128,
            num_encoder_layers=1,
        ),
    }

    print(f"\nBenchmarking on {DEVICE}...")
    print(f"{'Architecture':<25} {'Mean (ms)':<12} {'Std (ms)':<12} {'FPS':<10}")
    print("-" * 60)

    for name, config in configs.items():
        model = create_model(config=config)
        results = benchmark_model(model, num_iterations=5, warmup=2)
        print(
            f"{name:<25} {results['mean_ms']:<12.2f} {results['std_ms']:<12.2f} {results['fps']:<10.2f}"
        )


def run_all_demos():
    """Run all demo functions."""
    demo_config()
    demo_models()
    demo_training_step()
    demo_metrics()
    demo_losses()
    demo_callbacks()
    demo_export()
    demo_benchmark()

    print("\n" + "=" * 60)
    print(" All Demos Complete!")
    print("=" * 60)
    print("\nYou can now explore interactively. Try:")
    print("  - model = create_model(config=ModelConfig(...))")
    print("  - batch = create_dummy_batch()")
    print("  - output = model(batch['photos'], ...)")
    print("")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BeautyScorer Debug Notebook")
    parser.add_argument(
        "--demo",
        choices=[
            "all",
            "config",
            "models",
            "training",
            "metrics",
            "losses",
            "callbacks",
            "export",
            "benchmark",
        ],
        default="all",
        help="Which demo to run",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run interactively (drop to Python prompt after)",
    )

    args = parser.parse_args()

    demos = {
        "config": demo_config,
        "models": demo_models,
        "training": demo_training_step,
        "metrics": demo_metrics,
        "losses": demo_losses,
        "callbacks": demo_callbacks,
        "export": demo_export,
        "benchmark": demo_benchmark,
        "all": run_all_demos,
    }

    demos[args.demo]()

    if args.interactive:
        print("\nEntering interactive mode...")
        import code

        code.interact(local=globals())
