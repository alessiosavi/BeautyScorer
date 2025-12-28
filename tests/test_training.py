"""
Tests for training module.

Tests:
- Loss functions
- Optimizers and schedulers
- Metrics computation
- Callbacks
- Trainer functionality
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from beauty_scorer.config import ModelConfig, TrainingConfig
from beauty_scorer.models.factory import create_model
from beauty_scorer.training.callbacks import CallbackList, EarlyStopping, ModelCheckpoint
from beauty_scorer.training.losses import (
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingCrossEntropy,
    OrdinalRegressionLoss,
    get_loss_function,
)
from beauty_scorer.training.metrics import MAE, Accuracy, MetricTracker, compute_metrics
from beauty_scorer.training.optimizers import create_optimizer, create_scheduler


# Test fixtures
@pytest.fixture
def num_classes():
    return 9


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def sample_logits(batch_size, num_classes):
    """Create sample logits for testing."""
    return torch.randn(batch_size, num_classes)


@pytest.fixture
def sample_targets(batch_size, num_classes):
    """Create sample targets for testing."""
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def small_model():
    """Create small model for testing."""
    config = ModelConfig(
        architecture="lightweight_cpu",
        backbone="mobilenet_v2",
        embed_dim=64,
        num_encoder_layers=1,
        num_classes=9,
    )
    return create_model(config=config)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_cross_entropy_loss(self, sample_logits, sample_targets):
        """Test standard cross entropy loss."""
        loss_fn = CrossEntropyLoss()
        loss = loss_fn(sample_logits, sample_targets)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_cross_entropy_with_weights(self, sample_logits, sample_targets, num_classes):
        """Test cross entropy with class weights."""
        weights = torch.ones(num_classes)
        weights[0] = 2.0  # Higher weight for first class

        loss_fn = CrossEntropyLoss(weight=weights)
        loss = loss_fn(sample_logits, sample_targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_label_smoothing_loss(self, sample_logits, sample_targets):
        """Test label smoothing cross entropy."""
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        loss = loss_fn(sample_logits, sample_targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_label_smoothing_vs_standard(self, sample_logits, sample_targets):
        """Test that label smoothing gives different loss."""
        loss_standard = CrossEntropyLoss()(sample_logits, sample_targets)
        loss_smoothed = LabelSmoothingCrossEntropy(smoothing=0.1)(sample_logits, sample_targets)

        # With smoothing, loss should be different
        assert not torch.isclose(loss_standard, loss_smoothed)

    def test_focal_loss(self, sample_logits, sample_targets):
        """Test focal loss."""
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_loss_gamma(self, sample_logits, sample_targets):
        """Test focal loss with different gamma values."""
        loss_gamma0 = FocalLoss(gamma=0.0)(sample_logits, sample_targets)
        loss_gamma2 = FocalLoss(gamma=2.0)(sample_logits, sample_targets)

        # Higher gamma should focus more on hard examples
        # Both should be valid losses
        assert loss_gamma0.item() > 0
        assert loss_gamma2.item() > 0

    def test_ordinal_regression_loss(self, batch_size, num_classes):
        """Test ordinal regression loss."""
        # Ordinal loss expects num_classes - 1 outputs
        logits = torch.randn(batch_size, num_classes - 1)
        targets = torch.randint(0, num_classes, (batch_size,))

        loss_fn = OrdinalRegressionLoss(num_classes=num_classes)
        loss = loss_fn(logits, targets)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_get_loss_function(self, num_classes):
        """Test loss function factory."""
        ce_loss = get_loss_function("cross_entropy", num_classes=num_classes)
        assert isinstance(ce_loss, (CrossEntropyLoss, LabelSmoothingCrossEntropy))

        focal_loss = get_loss_function("focal", num_classes=num_classes)
        assert isinstance(focal_loss, FocalLoss)

        ordinal_loss = get_loss_function("ordinal", num_classes=num_classes)
        assert isinstance(ordinal_loss, OrdinalRegressionLoss)


class TestMetrics:
    """Tests for metric computation."""

    def test_accuracy_metric(self):
        """Test accuracy metric."""
        acc = Accuracy()

        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 3, 4, 4])  # 4/5 correct

        acc.update(predictions, targets)
        result = acc.compute()

        assert result == 0.8

    def test_accuracy_reset(self):
        """Test accuracy reset."""
        acc = Accuracy()

        acc.update(np.array([1, 1]), np.array([1, 1]))
        acc.reset()
        acc.update(np.array([1, 0]), np.array([0, 0]))

        assert acc.compute() == 0.5  # Only second batch counts

    def test_mae_metric(self):
        """Test MAE metric."""
        mae = MAE()

        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = np.array([1.0, 2.0, 3.0, 4.0, 6.0])  # Last one off by 1

        mae.update(predictions, targets)
        result = mae.compute()

        assert result == 0.2  # Average error = 1/5

    def test_metric_tracker(self, num_classes):
        """Test metric tracker."""
        tracker = MetricTracker(num_classes=num_classes)

        # Add some predictions
        for _ in range(3):
            preds = torch.randint(0, num_classes, (10,))
            targets = torch.randint(0, num_classes, (10,))
            tracker.update(preds, targets, loss=0.5)

        metrics = tracker.compute()

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "mae" in metrics
        assert "within_1" in metrics

    def test_metric_tracker_confusion_matrix(self, num_classes):
        """Test confusion matrix from tracker."""
        tracker = MetricTracker(num_classes=num_classes)

        preds = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 2])
        tracker.update(preds, targets)

        matrix = tracker.get_confusion_matrix()

        assert matrix.shape == (num_classes, num_classes)
        assert matrix.sum() == 5

    def test_compute_metrics(self, num_classes):
        """Test compute_metrics function."""
        predictions = np.array([1, 2, 3, 4, 5])
        targets = np.array([1, 2, 2, 4, 4])

        metrics = compute_metrics(predictions, targets, num_classes=num_classes)

        assert "accuracy" in metrics
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "within_1" in metrics
        assert "within_2" in metrics
        assert "confusion_matrix" in metrics


class TestOptimizers:
    """Tests for optimizer and scheduler creation."""

    def test_create_adamw_optimizer(self, small_model):
        """Test creating AdamW optimizer."""
        config = TrainingConfig(learning_rate=1e-4)
        optimizer = create_optimizer(small_model, config, optimizer_type="adamw")

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) > 0

    def test_create_sgd_optimizer(self, small_model):
        """Test creating SGD optimizer."""
        config = TrainingConfig(learning_rate=1e-3)
        optimizer = create_optimizer(small_model, config, optimizer_type="sgd")

        assert isinstance(optimizer, torch.optim.SGD)

    def test_optimizer_param_groups(self, small_model):
        """Test that optimizer has correct parameter groups."""
        config = TrainingConfig(learning_rate=1e-4, backbone_lr_multiplier=0.1)
        optimizer = create_optimizer(small_model, config)

        # Should have multiple param groups
        assert len(optimizer.param_groups) >= 1

    def test_create_cosine_scheduler(self, small_model):
        """Test creating cosine scheduler."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        config = TrainingConfig(scheduler="cosine", epochs=10)

        scheduler = create_scheduler(optimizer, config, num_epochs=10)

        assert scheduler is not None

    def test_create_step_scheduler(self, small_model):
        """Test creating step scheduler."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        config = TrainingConfig(scheduler="step", epochs=10)

        scheduler = create_scheduler(optimizer, config, num_epochs=10)

        assert scheduler is not None

    def test_no_scheduler(self, small_model):
        """Test no scheduler option."""
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        config = TrainingConfig(scheduler="none")

        scheduler = create_scheduler(optimizer, config)

        assert scheduler is None


class TestCallbacks:
    """Tests for training callbacks."""

    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        es = EarlyStopping(monitor="val_loss", patience=3, mode="min")

        # Simulate improving loss
        class MockTrainer:
            stop_training = False
            model = nn.Linear(10, 10)

        trainer = MockTrainer()
        es.on_train_begin(trainer)

        es.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        assert not trainer.stop_training

        es.on_epoch_end(trainer, 1, {"val_loss": 0.9})
        assert not trainer.stop_training

        es.on_epoch_end(trainer, 2, {"val_loss": 0.8})
        assert not trainer.stop_training

    def test_early_stopping_no_improvement(self):
        """Test early stopping triggers after patience."""
        es = EarlyStopping(monitor="val_loss", patience=2, mode="min")

        class MockTrainer:
            stop_training = False
            model = nn.Linear(10, 10)

        trainer = MockTrainer()
        es.on_train_begin(trainer)

        es.on_epoch_end(trainer, 0, {"val_loss": 1.0})
        es.on_epoch_end(trainer, 1, {"val_loss": 1.1})  # Worse
        es.on_epoch_end(trainer, 2, {"val_loss": 1.2})  # Worse

        assert trainer.stop_training  # Should stop after patience=2

    def test_model_checkpoint(self, small_model):
        """Test model checkpointing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = ModelCheckpoint(
                save_dir=tmpdir,
                monitor="val_loss",
                save_best_only=True,
            )

            class MockTrainer:
                model = small_model
                optimizer = torch.optim.Adam(small_model.parameters())
                scheduler = None
                current_epoch = 0

            trainer = MockTrainer()
            checkpoint.on_train_begin(trainer)

            # Save on improvement
            checkpoint.on_epoch_end(trainer, 0, {"val_loss": 1.0})
            checkpoint.on_epoch_end(trainer, 1, {"val_loss": 0.9})

            assert (Path(tmpdir) / "best.pt").exists()

    def test_callback_list(self):
        """Test callback list aggregation."""
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
        callbacks.on_train_end(trainer)


class TestTrainingIntegration:
    """Integration tests for training pipeline."""

    @pytest.mark.slow
    def test_single_training_step(self, small_model):
        """Test a single training step."""
        batch_size = 2
        max_photos = 4

        # Create dummy batch
        photos = torch.randn(batch_size, max_photos, 3, 224, 224)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, 224, 224)
        faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        targets = torch.randint(0, 9, (batch_size,))

        # Setup
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Forward pass
        small_model.train()
        logits = small_model(photos, photos_mask, faces, faces_mask)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    @pytest.mark.slow
    def test_gradient_accumulation(self, small_model):
        """Test gradient accumulation."""
        batch_size = 2
        max_photos = 4
        accumulation_steps = 2

        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        optimizer.zero_grad()

        for _ in range(accumulation_steps):
            photos = torch.randn(batch_size, max_photos, 3, 224, 224)
            photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            faces = torch.randn(batch_size, max_photos, 3, 224, 224)
            faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            targets = torch.randint(0, 9, (batch_size,))

            logits = small_model(photos, photos_mask, faces, faces_mask)
            loss = criterion(logits, targets) / accumulation_steps
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Should complete without error
        assert True
