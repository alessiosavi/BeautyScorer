"""
Tests for inference module.

Tests:
- Predictor functionality
- Model export
"""

import pytest
import torch

from beauty_scorer.config import DataConfig, ModelConfig
from beauty_scorer.inference.export import get_model_size
from beauty_scorer.models.factory import create_model


# Test fixtures
@pytest.fixture
def model():
    """Create a small model for testing."""
    config = ModelConfig(
        architecture="lightweight_cpu",
        backbone="mobilenet_v2",
        embed_dim=128,
        num_encoder_layers=1,
        num_classes=9,
    )
    return create_model(config=config)


@pytest.fixture
def sample_inputs():
    """Create sample inputs for inference."""
    batch_size = 1
    max_photos = 4
    h, w = 224, 224

    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    photos_mask[:, -1] = False
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces_mask[:, -2:] = False

    return photos, photos_mask, faces, faces_mask


class TestModelExport:
    """Tests for model export functionality."""

    # def test_export_torchscript(self, model, sample_inputs):
    #     """Test exporting to TorchScript."""
    #     with tempfile.TemporaryDirectory() as tmpdir:
    #         output_path = Path(tmpdir) / "model.pt"
    #         export_to_torchscript(
    #             model,
    #             output_path,
    #             max_photos=4,
    #             image_size=(224, 224),
    #             face_size=(224, 224),
    #         )

    #         assert output_path.exists()

    #         # Load and test
    #         loaded = torch.jit.load(str(output_path))
    #         photos, photos_mask, faces, faces_mask = sample_inputs

    #         with torch.no_grad():
    #             output = loaded(photos, photos_mask, faces, faces_mask)

    #         assert output.shape == (1, 9)

    def test_get_model_size(self, model):
        """Test getting model size information."""
        size_info = get_model_size(model)

        assert "total_params" in size_info
        assert "trainable_params" in size_info
        assert "saved_size_mb" in size_info
        assert size_info["total_params"] > 0
        assert size_info["saved_size_mb"] > 0


class TestModelInference:
    """Tests for model inference."""

    def test_inference_output_shape(self, model, sample_inputs):
        """Test inference output shape."""
        photos, photos_mask, faces, faces_mask = sample_inputs

        model.eval()
        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (1, 9)

    def test_inference_probabilities(self, model, sample_inputs):
        """Test that outputs can be converted to valid probabilities."""
        photos, photos_mask, faces, faces_mask = sample_inputs

        model.eval()
        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)
            probs = torch.softmax(output, dim=-1)

        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1), atol=1e-5)
        # All probabilities should be non-negative
        assert (probs >= 0).all()

    def test_predict_method(self, model, sample_inputs):
        """Test the predict method."""
        photos, photos_mask, faces, faces_mask = sample_inputs

        scores, probs = model.predict(photos, photos_mask, faces, faces_mask)

        assert scores.shape == (1,)
        assert probs.shape == (1, 9)
        assert scores[0].item() >= 1
        assert scores[0].item() <= 9

    def test_batch_inference(self, model):
        """Test inference with larger batch."""
        batch_size = 4
        max_photos = 4
        h, w = 224, 224

        photos = torch.randn(batch_size, max_photos, 3, h, w)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, h, w)
        faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, 9)


class TestDataConfig:
    """Tests for data configuration in inference."""

    def test_default_data_config(self):
        """Test default data config values."""
        config = DataConfig()
        assert config.image_size == (224, 224)
        assert config.face_size == (224, 224)
        assert config.max_photos == 9

    def test_custom_data_config(self):
        """Test custom data config."""
        config = DataConfig(
            image_size=(384, 384),
            face_size=(112, 112),
            max_photos=6,
        )
        assert config.image_size == (384, 384)
        assert config.face_size == (112, 112)
        assert config.max_photos == 6
