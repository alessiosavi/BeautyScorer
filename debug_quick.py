#!/usr/bin/env python3
"""
Quick sanity check script for BeautyScorer.

Run this script to verify that all components are working correctly.
This performs fast checks without requiring GPU or large datasets.

Usage:
    python debug_quick.py
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import torch


def print_header(title: str) -> None:
    """Print section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}")


def print_status(name: str, passed: bool, error: str = "") -> None:
    """Print test status."""
    status = "[PASS]" if passed else "[FAIL]"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  {color}{status}{reset} {name}")
    if error:
        print(f"         Error: {error}")


def check_imports() -> dict[str, bool]:
    """Check all imports work correctly."""
    print_header("Checking Imports")
    results = {}

    imports = [
        ("beauty_scorer", "from beauty_scorer import create_model, BeautyPredictor"),
        ("config", "from beauty_scorer.config import BeautyConfig, ModelConfig, load_config"),
        ("data.dataset", "from beauty_scorer.data.dataset import BeautyDataset"),
        ("data.transforms", "from beauty_scorer.data.transforms import get_train_transforms"),
        ("models.factory", "from beauty_scorer.models.factory import create_model, ModelFactory"),
        ("models.base", "from beauty_scorer.models.base import BaseBeautyModel"),
        ("training.trainer", "from beauty_scorer.training.trainer import Trainer"),
        ("training.losses", "from beauty_scorer.training.losses import get_loss_function"),
        ("training.metrics", "from beauty_scorer.training.metrics import compute_metrics"),
        ("training.callbacks", "from beauty_scorer.training.callbacks import EarlyStopping"),
        ("inference.predictor", "from beauty_scorer.inference.predictor import BeautyPredictor"),
        ("inference.export", "from beauty_scorer.inference.export import export_to_torchscript"),
        ("utils.logging", "from beauty_scorer.utils.logging import setup_logging"),
        ("utils.device", "from beauty_scorer.utils.device import get_device"),
    ]

    for name, import_stmt in imports:
        try:
            exec(import_stmt)
            results[name] = True
            print_status(name, True)
        except Exception as e:
            results[name] = False
            print_status(name, False, str(e))

    return results


def check_config() -> dict[str, bool]:
    """Check configuration system."""
    print_header("Checking Configuration")
    results = {}

    from beauty_scorer.config import BeautyConfig, ModelConfig

    # Test default config
    try:
        config = BeautyConfig()
        assert config.model.architecture == "mobilenet_transformer"
        results["default_config"] = True
        print_status("Default config creation", True)
    except Exception as e:
        results["default_config"] = False
        print_status("Default config creation", False, str(e))

    # Test custom config
    try:
        config = ModelConfig(
            architecture="lightweight_cpu",
            embed_dim=128,
            num_encoder_layers=1,
        )
        assert config.embed_dim == 128
        results["custom_config"] = True
        print_status("Custom config creation", True)
    except Exception as e:
        results["custom_config"] = False
        print_status("Custom config creation", False, str(e))

    # Test validation
    try:
        caught_error = False
        try:
            ModelConfig(embed_dim=10)  # Too small
        except ValueError:
            caught_error = True
        assert caught_error, "Should have raised ValueError"
        results["config_validation"] = True
        print_status("Config validation", True)
    except Exception as e:
        results["config_validation"] = False
        print_status("Config validation", False, str(e))

    # Test YAML loading
    try:
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            from beauty_scorer.config import load_config

            config = load_config(str(config_path))
            results["yaml_loading"] = True
            print_status("YAML config loading", True)
        else:
            results["yaml_loading"] = True
            print_status("YAML config loading", True, "(skipped - no config file)")
    except Exception as e:
        results["yaml_loading"] = False
        print_status("YAML config loading", False, str(e))

    return results


def check_models() -> dict[str, bool]:
    """Check model creation and forward pass."""
    print_header("Checking Models")
    results = {}

    from beauty_scorer.config import ModelConfig
    from beauty_scorer.models.factory import create_model

    architectures = [
        ("lightweight_cpu", {"embed_dim": 64, "num_encoder_layers": 1}),
        ("mobilenet_transformer", {"embed_dim": 128, "num_encoder_layers": 1}),
        ("vit_arcface", {"embed_dim": 128, "num_encoder_layers": 1}),
    ]

    batch_size = 2
    max_photos = 4

    for arch_name, overrides in architectures:
        try:
            config = ModelConfig(
                architecture=arch_name,
                backbone="mobilenet_v2",
                **overrides,
            )
            model = create_model(config=config)
            model.eval()

            # Create dummy input
            photos = torch.randn(batch_size, max_photos, 3, 224, 224)
            photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            faces = torch.randn(batch_size, max_photos, 3, 224, 224)
            faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

            # Forward pass
            with torch.no_grad():
                output = model(photos, photos_mask, faces, faces_mask)

            assert output.shape == (batch_size, 9), f"Expected (2, 9), got {output.shape}"
            assert not torch.isnan(output).any(), "Output contains NaN"

            results[arch_name] = True
            print_status(f"Model: {arch_name}", True)
        except Exception as e:
            results[arch_name] = False
            print_status(f"Model: {arch_name}", False, str(e))

    return results


def check_losses() -> dict[str, bool]:
    """Check loss functions."""
    print_header("Checking Loss Functions")
    results = {}

    from beauty_scorer.training.losses import get_loss_function

    losses = ["cross_entropy", "focal", "ordinal"]

    batch_size = 4
    num_classes = 9

    for loss_name in losses:
        try:
            loss_fn = get_loss_function(loss_name, num_classes=num_classes)

            if loss_name == "ordinal":
                logits = torch.randn(batch_size, num_classes - 1)
            else:
                logits = torch.randn(batch_size, num_classes)

            targets = torch.randint(0, num_classes, (batch_size,))

            loss = loss_fn(logits, targets)

            assert loss.ndim == 0, "Loss should be scalar"
            assert loss.item() > 0, "Loss should be positive"
            assert not torch.isnan(loss), "Loss contains NaN"

            results[loss_name] = True
            print_status(f"Loss: {loss_name}", True)
        except Exception as e:
            results[loss_name] = False
            print_status(f"Loss: {loss_name}", False, str(e))

    return results


def check_metrics() -> dict[str, bool]:
    """Check metrics computation."""
    print_header("Checking Metrics")
    results = {}

    from beauty_scorer.training.metrics import MetricTracker, compute_metrics

    # Test compute_metrics
    try:
        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 2, 4, 4])

        metrics = compute_metrics(predictions, targets, num_classes=9)

        assert "accuracy" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert metrics["accuracy"] >= 0 and metrics["accuracy"] <= 1

        results["compute_metrics"] = True
        print_status("compute_metrics()", True)
    except Exception as e:
        results["compute_metrics"] = False
        print_status("compute_metrics()", False, str(e))

    # Test MetricTracker
    try:
        tracker = MetricTracker(num_classes=9)

        for _ in range(3):
            preds = torch.randint(0, 9, (10,))
            targets = torch.randint(0, 9, (10,))
            tracker.update(preds, targets, loss=0.5)

        metrics = tracker.compute()
        assert "loss" in metrics
        assert "accuracy" in metrics

        results["metric_tracker"] = True
        print_status("MetricTracker", True)
    except Exception as e:
        results["metric_tracker"] = False
        print_status("MetricTracker", False, str(e))

    return results


def check_transforms() -> dict[str, bool]:
    """Check data transforms."""
    print_header("Checking Transforms")
    results = {}

    from beauty_scorer.data.transforms import get_train_transforms, get_val_transforms

    # Create dummy image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    # Test train transforms
    try:
        train_transform = get_train_transforms(image_size=(224, 224))
        result = train_transform(image=image)["image"]

        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

        results["train_transforms"] = True
        print_status("Train transforms", True)
    except Exception as e:
        results["train_transforms"] = False
        print_status("Train transforms", False, str(e))

    # Test val transforms
    try:
        val_transform = get_val_transforms(image_size=(224, 224))
        result = val_transform(image=image)["image"]

        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

        results["val_transforms"] = True
        print_status("Val transforms", True)
    except Exception as e:
        results["val_transforms"] = False
        print_status("Val transforms", False, str(e))

    return results


def check_callbacks() -> dict[str, bool]:
    """Check training callbacks."""
    print_header("Checking Callbacks")
    results = {}

    import torch.nn as nn

    from beauty_scorer.training.callbacks import CallbackList, EarlyStopping

    # Test EarlyStopping
    try:
        es = EarlyStopping(monitor="val_loss", patience=2, mode="min")

        class MockTrainer:
            stop_training = False
            model = nn.Linear(10, 10)

        trainer = MockTrainer()
        es.on_train_begin(trainer)

        es.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        assert not trainer.stop_training

        es.on_epoch_end(trainer, 1, {"val_loss": 1.1})
        es.on_epoch_end(trainer, 2, {"val_loss": 1.2})

        assert trainer.stop_training

        results["early_stopping"] = True
        print_status("EarlyStopping callback", True)
    except Exception as e:
        results["early_stopping"] = False
        print_status("EarlyStopping callback", False, str(e))

    # Test CallbackList
    try:
        callbacks = CallbackList(
            [
                EarlyStopping(monitor="val_loss", patience=5),
            ]
        )

        class MockTrainer:
            stop_training = False
            model = nn.Linear(10, 10)

        trainer = MockTrainer()
        callbacks.on_train_begin(trainer)
        callbacks.on_epoch_begin(trainer, 0)
        callbacks.on_epoch_end(trainer, 0, {"val_loss": 1.0})

        results["callback_list"] = True
        print_status("CallbackList", True)
    except Exception as e:
        results["callback_list"] = False
        print_status("CallbackList", False, str(e))

    return results


def check_device() -> dict[str, bool]:
    """Check device utilities."""
    print_header("Checking Device Utilities")
    results = {}

    from beauty_scorer.utils.device import get_device, set_seed

    # Test get_device
    try:
        device = get_device()
        assert isinstance(device, torch.device)
        results["get_device"] = True
        print_status(f"get_device() -> {device}", True)
    except Exception as e:
        results["get_device"] = False
        print_status("get_device()", False, str(e))

    # Test set_seed
    try:
        set_seed(42)
        results["set_seed"] = True
        print_status("set_seed()", True)
    except Exception as e:
        results["set_seed"] = False
        print_status("set_seed()", False, str(e))

    return results


def main() -> int:
    """Run all checks and return exit code."""
    print("\n" + "=" * 60)
    print(" BeautyScorer Quick Sanity Check")
    print("=" * 60)

    all_results = {}

    # Run all checks
    all_results.update(check_imports())
    all_results.update(check_config())
    all_results.update(check_models())
    all_results.update(check_losses())
    all_results.update(check_metrics())
    all_results.update(check_transforms())
    all_results.update(check_callbacks())
    all_results.update(check_device())

    # Print summary
    print_header("Summary")

    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)

    print(f"\n  Passed: {passed}/{total}")

    if passed == total:
        print("\n  \033[92mAll checks passed!\033[0m\n")
        return 0
    else:
        failed = [k for k, v in all_results.items() if not v]
        print(f"\n  \033[91mFailed checks: {', '.join(failed)}\033[0m\n")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n\033[91mFatal error: {e}\033[0m")
        traceback.print_exc()
        sys.exit(1)
