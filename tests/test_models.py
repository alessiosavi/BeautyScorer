"""
Tests for model architectures.

Tests:
- Model forward pass shapes
- Permutation invariance
- Parameter counting
- Model factory
"""

import pytest
import torch

from beauty_scorer.config import ModelConfig
from beauty_scorer.models.factory import ModelFactory
from beauty_scorer.models.lightweight_cnn import LightweightCPUModel
from beauty_scorer.models.mobilenet_transformer import MobileNetTransformerModel


# Test fixtures
@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def max_photos():
    return 4


@pytest.fixture
def image_size():
    return (224, 224)


@pytest.fixture
def num_classes():
    return 9


@pytest.fixture
def sample_batch(batch_size, max_photos, image_size):
    """Create a sample batch for testing."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    photos_mask[:, -1] = False  # Mask out last photo
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces_mask[:, -2:] = False  # Mask out last two faces
    return photos, photos_mask, faces, faces_mask


class TestModelFactory:
    """Tests for model factory."""

    def test_list_available(self):
        """Test listing available models."""
        available = ModelFactory.list_available()
        assert "mobilenet_transformer" in available
        assert "vit_arcface" in available
        assert "lightweight_cpu" in available

    def test_create_mobilenet_transformer(self):
        """Test creating MobileNet+Transformer model."""
        config = ModelConfig(architecture="mobilenet_transformer")
        model = ModelFactory.create(config)
        assert isinstance(model, MobileNetTransformerModel)

    def test_create_lightweight_cpu(self):
        """Test creating lightweight CPU model."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
        )
        model = ModelFactory.create(config)
        assert isinstance(model, LightweightCPUModel)

    def test_create_with_override(self):
        """Test creating model with architecture override."""
        config = ModelConfig(architecture="mobilenet_transformer")
        model = ModelFactory.create(config, architecture="lightweight_cpu")
        assert isinstance(model, LightweightCPUModel)


class TestMobileNetTransformer:
    """Tests for MobileNet+Transformer model."""

    @pytest.fixture
    def model(self, num_classes):
        """Create model for testing."""
        config = ModelConfig(
            architecture="mobilenet_transformer",
            embed_dim=256,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
            unfreeze_backbone_layers=0,
            use_cross_attention=True,
            use_se_attention=True,
        )
        return MobileNetTransformerModel(config)

    def test_forward_shape(self, model, sample_batch, batch_size, num_classes):
        """Test forward pass output shape."""
        photos, photos_mask, faces, faces_mask = sample_batch
        output = model(photos, photos_mask, faces, faces_mask)
        assert output.shape == (batch_size, num_classes)

    def test_forward_with_all_masked(self, model, batch_size, max_photos, image_size, num_classes):
        """Test forward pass with all photos masked."""
        h, w = image_size
        photos = torch.randn(batch_size, max_photos, 3, h, w)
        photos_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        photos_mask[:, 0] = True  # At least one valid
        faces = torch.randn(batch_size, max_photos, 3, h, w)
        faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        faces_mask[:, 0] = True

        output = model(photos, photos_mask, faces, faces_mask)
        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any()

    def test_permutation_invariance(self, model, batch_size, image_size, num_classes):
        """Test that shuffling photo order doesn't change output."""
        h, w = image_size
        max_photos = 4

        # Create batch with valid photos
        photos = torch.randn(batch_size, max_photos, 3, h, w)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, h, w)
        faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            output1 = model(photos, photos_mask, faces, faces_mask)

            # Shuffle photos and faces together
            perm = torch.randperm(max_photos)
            photos_shuffled = photos[:, perm]
            photos_mask_shuffled = photos_mask[:, perm]
            faces_shuffled = faces[:, perm]
            faces_mask_shuffled = faces_mask[:, perm]

            output2 = model(
                photos_shuffled,
                photos_mask_shuffled,
                faces_shuffled,
                faces_mask_shuffled,
            )

        # Outputs should be very close (small numerical differences allowed)
        assert torch.allclose(output1, output2, atol=1e-5)

    def test_trainable_params(self, model):
        """Test getting trainable parameters."""
        param_groups = model.get_trainable_params()
        assert len(param_groups) > 0
        for group in param_groups:
            assert "params" in group
            assert "lr_scale" in group

    def test_parameter_count(self, model):
        """Test parameter counting."""
        total = model.count_parameters(trainable_only=False)
        trainable = model.count_parameters(trainable_only=True)
        assert total > 0
        assert trainable <= total


class TestLightweightCPU:
    """Tests for lightweight CPU model."""

    @pytest.fixture
    def model(self, num_classes):
        """Create model for testing."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="mobilenet_v2",
            embed_dim=128,
            num_encoder_layers=1,  # No attention for speed
            num_classes=num_classes,
        )
        return LightweightCPUModel(config)

    def test_forward_shape(self, model, sample_batch, batch_size, num_classes):
        """Test forward pass output shape."""
        photos, photos_mask, faces, faces_mask = sample_batch
        output = model(photos, photos_mask, faces, faces_mask)
        assert output.shape == (batch_size, num_classes)

    def test_cpu_inference(self, model, sample_batch, batch_size, num_classes):
        """Test inference on CPU."""
        model.cpu()
        photos, photos_mask, faces, faces_mask = sample_batch

        # Ensure everything is on CPU
        photos = photos.cpu()
        photos_mask = photos_mask.cpu()
        faces = faces.cpu()
        faces_mask = faces_mask.cpu()

        model.eval()
        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert output.device.type == "cpu"


class TestModelConfig:
    """Tests for model configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ModelConfig()
        assert config.architecture == "mobilenet_transformer"
        assert config.num_classes == 9
        assert config.embed_dim == 512

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ModelConfig(embed_dim=256, num_heads=8)
        assert config.embed_dim == 256

        # Invalid should raise error
        with pytest.raises(ValueError):
            ModelConfig(embed_dim=32)  # Too small

    def test_config_copy(self):
        """Test configuration copying with updates."""
        config1 = ModelConfig(embed_dim=512)
        config2 = config1.model_copy(update={"embed_dim": 256})
        assert config1.embed_dim == 512
        assert config2.embed_dim == 256
