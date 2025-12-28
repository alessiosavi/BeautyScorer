"""
Tests for configuration system.

Tests:
- Config loading and validation
- YAML serialization
- Config merging
"""

import tempfile

import pytest

from beauty_scorer.config import (
    BeautyConfig,
    DataConfig,
    LoggingConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    load_config,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ModelConfig()

        assert config.architecture == "mobilenet_transformer"
        assert config.backbone == "mobilenet_v3_large"
        assert config.embed_dim == 512
        assert config.num_encoder_layers == 4
        assert config.num_heads == 8
        assert config.num_classes == 9

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            embed_dim=256,
            num_encoder_layers=2,
        )

        assert config.architecture == "lightweight_cpu"
        assert config.embed_dim == 256
        assert config.num_encoder_layers == 2

    def test_validation_embed_dim(self):
        """Test embed_dim validation."""
        with pytest.raises(ValueError):
            ModelConfig(embed_dim=32)  # Too small (< 64)

        with pytest.raises(ValueError):
            ModelConfig(embed_dim=3000)  # Too large (> 2048)

    def test_validation_dropout(self):
        """Test dropout validation."""
        with pytest.raises(ValueError):
            ModelConfig(dropout=-0.1)  # Negative

        with pytest.raises(ValueError):
            ModelConfig(dropout=0.9)  # Too high (> 0.8)

    def test_architecture_choices(self):
        """Test architecture must be valid choice."""
        # Valid choices
        ModelConfig(architecture="mobilenet_transformer")
        ModelConfig(architecture="vit_arcface")
        ModelConfig(architecture="lightweight_cpu")

        # Invalid choice
        with pytest.raises(ValueError):
            ModelConfig(architecture="invalid_arch")


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.epochs == 50
        assert config.patience == 5
        assert config.use_amp is True

    def test_validation_batch_size(self):
        """Test batch_size validation."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

        with pytest.raises(ValueError):
            TrainingConfig(batch_size=300)  # > 256

    def test_validation_val_split(self):
        """Test val_split validation."""
        TrainingConfig(val_split=0.0)  # Valid
        TrainingConfig(val_split=0.5)  # Valid

        with pytest.raises(ValueError):
            TrainingConfig(val_split=-0.1)

        with pytest.raises(ValueError):
            TrainingConfig(val_split=0.6)  # > 0.5

    def test_loss_function_choices(self):
        """Test loss function choices."""
        TrainingConfig(loss_function="cross_entropy")
        TrainingConfig(loss_function="focal")
        TrainingConfig(loss_function="ordinal")

        with pytest.raises(ValueError):
            TrainingConfig(loss_function="invalid_loss")

    def test_scheduler_choices(self):
        """Test scheduler choices."""
        TrainingConfig(scheduler="cosine")
        TrainingConfig(scheduler="step")
        TrainingConfig(scheduler="plateau")
        TrainingConfig(scheduler="none")

        with pytest.raises(ValueError):
            TrainingConfig(scheduler="invalid_scheduler")


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()

        assert config.image_size == (224, 224)
        assert config.face_size == (224, 224)
        assert config.max_photos == 9
        assert config.augmentation is True

    def test_tuple_conversion(self):
        """Test that lists are converted to tuples."""
        config = DataConfig(image_size=[384, 384])

        assert config.image_size == (384, 384)
        assert isinstance(config.image_size, tuple)

    def test_face_detector_choices(self):
        """Test face detector choices."""
        DataConfig(face_detector="yolov8")
        DataConfig(face_detector="retinaface")
        DataConfig(face_detector="mtcnn")
        DataConfig(face_detector="opencv")

        with pytest.raises(ValueError):
            DataConfig(face_detector="invalid_detector")

    def test_augmentation_strength_choices(self):
        """Test augmentation strength choices."""
        DataConfig(augmentation_strength="light")
        DataConfig(augmentation_strength="medium")
        DataConfig(augmentation_strength="heavy")

        with pytest.raises(ValueError):
            DataConfig(augmentation_strength="invalid")


class TestBeautyConfig:
    """Tests for main BeautyConfig."""

    def test_default_construction(self):
        """Test default configuration."""
        config = BeautyConfig()

        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_nested_config(self):
        """Test nested configuration."""
        config = BeautyConfig(
            model=ModelConfig(embed_dim=256),
            training=TrainingConfig(epochs=100),
        )

        assert config.model.embed_dim == 256
        assert config.training.epochs == 100

    # def test_yaml_round_trip(self):
    #     """Test YAML serialization and deserialization."""
    #     original = BeautyConfig(
    #         model=ModelConfig(embed_dim=256),
    #         training=TrainingConfig(epochs=100),
    #     )

    #     with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
    #         original.to_yaml(f.name)

    #         loaded = BeautyConfig.from_yaml(f.name)

    #         assert loaded.model.embed_dim == 256
    #         assert loaded.training.epochs == 100

    def test_merge_with(self):
        """Test config merging."""
        base = BeautyConfig()
        merged = base.merge_with(
            {
                "model": {"embed_dim": 256},
                "training": {"epochs": 100},
            }
        )

        # Original unchanged
        assert base.model.embed_dim == 512
        assert base.training.epochs == 50

        # Merged has new values
        assert merged.model.embed_dim == 256
        assert merged.training.epochs == 100

    def test_deep_merge(self):
        """Test deep merging of nested configs."""
        base = BeautyConfig()
        merged = base.merge_with(
            {
                "model": {
                    "embed_dim": 256,
                    # Other model values should be preserved
                }
            }
        )

        # Changed value
        assert merged.model.embed_dim == 256

        # Preserved values
        assert merged.model.num_heads == 8
        assert merged.model.backbone == "mobilenet_v3_large"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = """
model:
  architecture: vit_arcface
  embed_dim: 768

training:
  batch_size: 16
  epochs: 100
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(f.name)

            assert config.model.architecture == "vit_arcface"
            assert config.model.embed_dim == 768
            assert config.training.batch_size == 16
            assert config.training.epochs == 100

    def test_load_with_overrides(self):
        """Test loading with overrides."""
        yaml_content = """
model:
  embed_dim: 512

training:
  epochs: 50
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            config = load_config(
                f.name,
                model={"embed_dim": 256},
                training={"epochs": 100},
            )

            assert config.model.embed_dim == 256
            assert config.training.epochs == 100

    def test_load_none_returns_default(self):
        """Test that None path returns default config."""
        config = load_config(None)

        assert isinstance(config, BeautyConfig)
        assert config.model.embed_dim == 512

    def test_load_missing_file(self):
        """Test loading missing file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")


class TestConfigIntegration:
    """Integration tests for configuration."""

    def test_config_for_training(self):
        """Test config can be used for training setup."""
        config = BeautyConfig(
            model=ModelConfig(
                architecture="lightweight_cpu",
                embed_dim=128,
            ),
            training=TrainingConfig(
                batch_size=8,
                learning_rate=1e-4,
            ),
        )

        # Should be able to access all needed values
        assert config.model.architecture == "lightweight_cpu"
        assert config.training.batch_size == 8
        assert config.data.max_photos == 9
        assert config.paths.output_dir == "outputs"

    def test_config_dump(self):
        """Test config can be dumped to dict."""
        config = BeautyConfig()
        dumped = config.model_dump()

        assert isinstance(dumped, dict)
        assert "model" in dumped
        assert "training" in dumped
        assert "data" in dumped
