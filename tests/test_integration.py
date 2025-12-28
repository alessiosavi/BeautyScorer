"""
Integration tests for BeautyScorer.

Tests end-to-end functionality across multiple components.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from beauty_scorer.config import BeautyConfig, ModelConfig
from beauty_scorer.data.dataset import BeautyDataset, compute_class_weights
from beauty_scorer.data.transforms import get_train_transforms, get_val_transforms
from beauty_scorer.inference.export import get_model_size
from beauty_scorer.models.factory import create_model, load_model
from beauty_scorer.training.losses import get_loss_function
from beauty_scorer.training.metrics import compute_metrics
from beauty_scorer.training.optimizers import create_optimizer


class TestModelPipeline:
    """Test complete model creation and inference pipeline."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=64,
            num_encoder_layers=1,
            num_classes=9,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch."""
        batch_size = 2
        max_photos = 4
        return {
            "photos": torch.randn(batch_size, max_photos, 3, 224, 224),
            "photos_mask": torch.ones(batch_size, max_photos, dtype=torch.bool),
            "faces": torch.randn(batch_size, max_photos, 3, 224, 224),
            "faces_mask": torch.ones(batch_size, max_photos, dtype=torch.bool),
            "targets": torch.randint(0, 9, (batch_size,)),
        }

    def test_model_creation_to_inference(self, config, sample_batch):
        """Test model from creation to inference."""
        # Create model
        model = create_model(config=config)

        # Inference
        model.eval()
        with torch.no_grad():
            logits = model(
                sample_batch["photos"],
                sample_batch["photos_mask"],
                sample_batch["faces"],
                sample_batch["faces_mask"],
            )

        assert logits.shape == (2, 9)
        assert not torch.isnan(logits).any()

    def test_model_save_and_load(self, config, sample_batch):
        """Test model save and load cycle."""
        model = create_model(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            save_path = Path(tmpdir) / "model.pt"
            model.save(str(save_path))

            # Load
            loaded_model = load_model(str(save_path))

            # Compare outputs
            model.eval()
            loaded_model.eval()

            with torch.no_grad():
                output1 = model(
                    sample_batch["photos"],
                    sample_batch["photos_mask"],
                    sample_batch["faces"],
                    sample_batch["faces_mask"],
                )
                output2 = loaded_model(
                    sample_batch["photos"],
                    sample_batch["photos_mask"],
                    sample_batch["faces"],
                    sample_batch["faces_mask"],
                )

            assert torch.allclose(output1, output2)

    # def test_model_export_and_inference(self, config, sample_batch):
    #     """Test model export and inference with exported model."""
    #     model = create_model(config=config)

    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         # Export to TorchScript
    #         export_path = Path(tmpdir) / "model.pt"
    #         export_to_torchscript(
    #             model,
    #             export_path,
    #             max_photos=4,
    #         )

    #         # Load and run
    #         loaded = torch.jit.load(str(export_path))
    #         loaded.eval()

    #         with torch.no_grad():
    #             output = loaded(
    #                 sample_batch["photos"],
    #                 sample_batch["photos_mask"],
    #                 sample_batch["faces"],
    #                 sample_batch["faces_mask"],
    #             )

    #         assert output.shape == (2, 9)


class TestTrainingPipeline:
    """Test training pipeline components together."""

    @pytest.fixture
    def model(self):
        """Create model for training tests."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=64,
            num_encoder_layers=1,
        )
        return create_model(config=config)

    def test_training_step(self, model):
        """Test complete training step."""
        batch_size = 2
        max_photos = 4

        # Create batch
        photos = torch.randn(batch_size, max_photos, 3, 224, 224)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, 224, 224)
        faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        targets = torch.randint(0, 9, (batch_size,))

        # Setup training
        optimizer = create_optimizer(model)
        criterion = get_loss_function("cross_entropy")

        # Training step
        model.train()
        optimizer.zero_grad()

        logits = model(photos, photos_mask, faces, faces_mask)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_validation_step(self, model):
        """Test validation step."""
        batch_size = 2
        max_photos = 4

        photos = torch.randn(batch_size, max_photos, 3, 224, 224)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, 224, 224)
        faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        targets = torch.randint(0, 9, (batch_size,))

        model.eval()
        with torch.no_grad():
            logits = model(photos, photos_mask, faces, faces_mask)
            predictions = logits.argmax(dim=-1)

        # Compute metrics
        metrics = compute_metrics(
            predictions.numpy(),
            targets.numpy(),
            num_classes=9,
        )

        assert "accuracy" in metrics
        assert "mae" in metrics


class TestDataPipeline:
    """Test data loading and processing pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data entries."""
        return [{"id": f"person_{i}", "photos": [], "score": (i % 9) + 1} for i in range(10)]

    def test_class_weights_computation(self, sample_data):
        """Test class weight computation."""
        weights = compute_class_weights(sample_data, num_classes=9)

        assert weights.shape == (9,)
        assert (weights > 0).all()

    def test_transforms_pipeline(self):
        """Test transforms create valid tensors."""
        train_transform = get_train_transforms(image_size=(224, 224))
        val_transform = get_val_transforms(image_size=(224, 224))

        # Create dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Apply transforms
        train_result = train_transform(image=image)["image"]
        val_result = val_transform(image=image)["image"]

        assert train_result.shape == (3, 224, 224)
        assert val_result.shape == (3, 224, 224)
        assert train_result.dtype == torch.float32

    def test_dataset_collate(self):
        """Test dataset collate function."""
        batch = [
            {
                "id": "p1",
                "photos_tensor": torch.randn(4, 3, 224, 224),
                "photos_mask": torch.ones(4, dtype=torch.bool),
                "faces_tensor": torch.randn(4, 3, 224, 224),
                "faces_mask": torch.ones(4, dtype=torch.bool),
                "score": 5,
            },
            {
                "id": "p2",
                "photos_tensor": torch.randn(4, 3, 224, 224),
                "photos_mask": torch.ones(4, dtype=torch.bool),
                "faces_tensor": torch.randn(4, 3, 224, 224),
                "faces_mask": torch.ones(4, dtype=torch.bool),
                "score": 7,
            },
        ]

        collated = BeautyDataset.collate_fn(batch)

        assert collated["photos_tensor"].shape == (2, 4, 3, 224, 224)
        assert collated["scores"].shape == (2,)
        assert len(collated["ids"]) == 2


class TestEndToEnd:
    """End-to-end tests for full workflow."""

    @pytest.mark.slow
    def test_full_training_loop(self):
        """Test a minimal training loop."""
        # Create config
        config = BeautyConfig(
            model=ModelConfig(
                architecture="lightweight_cpu",
                backbone="mobilenet_v2",
                embed_dim=64,
                num_encoder_layers=1,
            ),
        )

        # Create model
        model = create_model(config=config.model)

        # Create dummy data
        batch_size = 2
        max_photos = 4
        num_batches = 3

        # Setup training
        optimizer = create_optimizer(model, config.training)
        criterion = get_loss_function(
            config.training.loss_function,
            num_classes=config.model.num_classes,
        )

        # Training loop
        model.train()
        losses = []

        for _ in range(num_batches):
            photos = torch.randn(batch_size, max_photos, 3, 224, 224)
            photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            faces = torch.randn(batch_size, max_photos, 3, 224, 224)
            faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            targets = torch.randint(0, 9, (batch_size,))

            optimizer.zero_grad()
            logits = model(photos, photos_mask, faces, faces_mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Should complete without error
        assert len(losses) == num_batches
        assert all(loss > 0 for loss in losses)

    @pytest.mark.slow
    def test_inference_pipeline(self):
        """Test inference pipeline end-to-end."""
        # Create and save model
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=64,
            num_encoder_layers=1,
        )
        model = create_model(config=config)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            model.save(str(model_path))

            # Load model
            loaded = load_model(str(model_path))

            # Run inference
            batch_size = 1
            max_photos = 4

            photos = torch.randn(batch_size, max_photos, 3, 224, 224)
            photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
            faces = torch.randn(batch_size, max_photos, 3, 224, 224)
            faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

            scores, probs = loaded.predict(photos, photos_mask, faces, faces_mask)

            assert scores.shape == (1,)
            assert probs.shape == (1, 9)
            assert scores[0].item() >= 1
            assert scores[0].item() <= 9
            assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-5)


class TestModelSize:
    """Tests for model size and efficiency."""

    def test_lightweight_model_size(self):
        """Test lightweight model is actually smaller."""
        light_config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=128,
            num_encoder_layers=1,
        )
        light_model = create_model(config=light_config)

        full_config = ModelConfig(
            architecture="mobilenet_transformer",
            backbone="mobilenet_v3_large",
            embed_dim=512,
            num_encoder_layers=4,
        )
        full_model = create_model(config=full_config)

        light_params = light_model.count_parameters(trainable_only=False)
        full_params = full_model.count_parameters(trainable_only=False)

        # Lightweight should have fewer parameters
        assert light_params < full_params

    def test_model_size_info(self):
        """Test model size reporting."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            embed_dim=64,
            num_encoder_layers=1,
        )
        model = create_model(config=config)

        size_info = get_model_size(model)

        assert "total_params" in size_info
        assert "saved_size_mb" in size_info
        assert size_info["total_params"] > 0
        assert size_info["saved_size_mb"] > 0
