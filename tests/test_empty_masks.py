"""
Tests for handling empty face masks (no faces detected in images).

These tests verify that all models handle edge cases gracefully:
- No faces detected in some images
- No faces detected in all images in a batch
- Mixed scenarios with varying face detection rates
- Full pipeline from dataset to training step

This is critical for production use where face detection may fail.
"""

import pytest
import torch
import torch.nn as nn

from beauty_scorer.config import DataConfig, ModelConfig, TrainingConfig
from beauty_scorer.models.components.encoders import (
    CrossAttentionFusion,
    LightweightAttention,
    TransformerEncoder,
)
from beauty_scorer.models.factory import ModelFactory, create_model
from beauty_scorer.training.losses import get_loss_function

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def max_photos():
    return 6


@pytest.fixture
def image_size():
    return (224, 224)


@pytest.fixture
def face_size():
    return (112, 112)


@pytest.fixture
def num_classes():
    return 9


@pytest.fixture
def embed_dim():
    return 256


@pytest.fixture
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Test Data Generators
# =============================================================================


def create_batch_all_faces_valid(batch_size, max_photos, image_size):
    """Create batch where all samples have all valid faces."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    return photos, photos_mask, faces, faces_mask


def create_batch_no_faces(batch_size, max_photos, image_size):
    """Create batch where NO samples have valid faces (worst case)."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)  # ALL FALSE
    return photos, photos_mask, faces, faces_mask


def create_batch_some_faces_missing(batch_size, max_photos, image_size):
    """Create batch where SOME samples have no valid faces."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    # Make samples 1 and 3 have no valid faces
    faces_mask[1] = False
    faces_mask[3] = False
    return photos, photos_mask, faces, faces_mask


def create_batch_partial_faces(batch_size, max_photos, image_size):
    """Create batch where samples have varying numbers of valid faces."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
    # Sample 0: 3 faces, Sample 1: 0 faces, Sample 2: 1 face, Sample 3: all faces
    faces_mask[0, :3] = True
    faces_mask[1] = False
    faces_mask[2, 0] = True
    faces_mask[3] = True
    return photos, photos_mask, faces, faces_mask


def create_batch_no_photos_no_faces(batch_size, max_photos, image_size):
    """Create batch where some samples have neither photos nor faces (edge case)."""
    h, w = image_size
    photos = torch.randn(batch_size, max_photos, 3, h, w)
    photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
    photos_mask[1] = False  # Sample 1 has no valid photos
    faces = torch.randn(batch_size, max_photos, 3, h, w)
    faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
    faces_mask[0, :2] = True  # Sample 0 has 2 valid faces
    # Sample 1: no photos, no faces
    faces_mask[2, 0] = True  # Sample 2 has 1 valid face
    faces_mask[3] = True  # Sample 3 has all valid faces
    return photos, photos_mask, faces, faces_mask


# =============================================================================
# Component Tests
# =============================================================================


class TestLightweightAttentionEmptyMasks:
    """Test LightweightAttention with empty masks."""

    @pytest.fixture
    def attention(self, embed_dim):
        return LightweightAttention(embed_dim=embed_dim, dropout=0.0)

    def test_all_valid_masks(self, attention, batch_size, max_photos, embed_dim):
        """Test with all valid masks - baseline."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        output = attention(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    def test_all_invalid_masks(self, attention, batch_size, max_photos, embed_dim):
        """Test with ALL masks invalid - worst case."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)

        output = attention(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN with all-invalid masks"
        assert not torch.isinf(output).any(), "Output contains Inf with all-invalid masks"

    def test_mixed_valid_invalid_masks(self, attention, batch_size, max_photos, embed_dim):
        """Test with some samples having invalid masks."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        mask[1] = False  # Sample 1 all invalid
        mask[3] = False  # Sample 3 all invalid

        output = attention(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN with mixed masks"
        assert not torch.isinf(output).any(), "Output contains Inf with mixed masks"

    def test_single_valid_per_sample(self, attention, batch_size, max_photos, embed_dim):
        """Test with only one valid token per sample."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        mask[:, 0] = True  # Only first token valid

        output = attention(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN with single valid"

    def test_no_mask(self, attention, batch_size, max_photos, embed_dim):
        """Test with no mask provided."""
        x = torch.randn(batch_size, max_photos, embed_dim)

        output = attention(x, mask=None)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any(), "Output contains NaN with no mask"


class TestCrossAttentionFusionEmptyMasks:
    """Test CrossAttentionFusion with empty masks."""

    @pytest.fixture
    def cross_attention(self, embed_dim):
        return CrossAttentionFusion(embed_dim=embed_dim, num_heads=4, dropout=0.0)

    def test_all_valid(self, cross_attention, batch_size, max_photos, embed_dim):
        """Test with all valid masks."""
        photo = torch.randn(batch_size, max_photos, embed_dim)
        face = torch.randn(batch_size, max_photos, embed_dim)
        photo_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        face_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        photo_out, face_out = cross_attention(photo, face, photo_mask, face_mask)

        assert photo_out.shape == (batch_size, max_photos, embed_dim)
        assert face_out.shape == (batch_size, max_photos, embed_dim)
        assert not torch.isnan(photo_out).any(), "Photo output contains NaN"
        assert not torch.isnan(face_out).any(), "Face output contains NaN"

    def test_no_valid_faces(self, cross_attention, batch_size, max_photos, embed_dim):
        """Test with NO valid faces in any sample."""
        photo = torch.randn(batch_size, max_photos, embed_dim)
        face = torch.randn(batch_size, max_photos, embed_dim)
        photo_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        face_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)  # All invalid

        photo_out, face_out = cross_attention(photo, face, photo_mask, face_mask)

        assert photo_out.shape == (batch_size, max_photos, embed_dim)
        assert face_out.shape == (batch_size, max_photos, embed_dim)
        assert not torch.isnan(photo_out).any(), "Photo output contains NaN with no faces"
        assert not torch.isnan(face_out).any(), "Face output contains NaN with no faces"

    def test_some_samples_no_faces(self, cross_attention, batch_size, max_photos, embed_dim):
        """Test with some samples having no valid faces."""
        photo = torch.randn(batch_size, max_photos, embed_dim)
        face = torch.randn(batch_size, max_photos, embed_dim)
        photo_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        face_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        face_mask[1] = False  # Sample 1 has no faces
        face_mask[2] = False  # Sample 2 has no faces

        photo_out, face_out = cross_attention(photo, face, photo_mask, face_mask)

        assert not torch.isnan(photo_out).any(), "Photo output contains NaN"
        assert not torch.isnan(face_out).any(), "Face output contains NaN"

    def test_no_photos_no_faces(self, cross_attention, batch_size, max_photos, embed_dim):
        """Test edge case with neither photos nor faces valid."""
        photo = torch.randn(batch_size, max_photos, embed_dim)
        face = torch.randn(batch_size, max_photos, embed_dim)
        photo_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        face_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)

        photo_out, face_out = cross_attention(photo, face, photo_mask, face_mask)

        assert not torch.isnan(photo_out).any(), "Photo output contains NaN"
        assert not torch.isnan(face_out).any(), "Face output contains NaN"


class TestTransformerEncoderEmptyMasks:
    """Test TransformerEncoder with empty masks."""

    @pytest.fixture
    def encoder(self, embed_dim):
        return TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=2,
            num_heads=4,
            ff_dim=embed_dim * 4,
            dropout=0.0,
        )

    def test_all_valid(self, encoder, batch_size, max_photos, embed_dim):
        """Test with all valid masks."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        output = encoder(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any()

    def test_all_invalid(self, encoder, batch_size, max_photos, embed_dim):
        """Test with all invalid masks (CLS token should still work)."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)

        output = encoder(x, mask)

        assert output.shape == (batch_size, embed_dim)
        # CLS token can attend to itself, so should not be NaN
        assert not torch.isnan(output).any(), "TransformerEncoder produces NaN with all-invalid"

    def test_mixed_masks(self, encoder, batch_size, max_photos, embed_dim):
        """Test with mixed validity."""
        x = torch.randn(batch_size, max_photos, embed_dim)
        mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        mask[0, :3] = True
        mask[2, 0] = True

        output = encoder(x, mask)

        assert output.shape == (batch_size, embed_dim)
        assert not torch.isnan(output).any()


# =============================================================================
# Full Model Tests - All Architectures
# =============================================================================


class TestMobileNetTransformerEmptyFaces:
    """Test MobileNetTransformer with empty face masks."""

    @pytest.fixture
    def model(self, num_classes):
        config = ModelConfig(
            architecture="mobilenet_transformer",
            backbone="mobilenet_v3_small",  # Smaller for faster tests
            embed_dim=256,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
            unfreeze_backbone_layers=0,
            use_cross_attention=True,
            use_se_attention=True,
        )
        model = ModelFactory.create(config)
        model.eval()
        return model

    def test_all_faces_valid(self, model, batch_size, max_photos, image_size, num_classes):
        """Baseline test with all faces valid."""
        photos, photos_mask, faces, faces_mask = create_batch_all_faces_valid(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN in output with all valid faces"

    def test_no_faces_detected(self, model, batch_size, max_photos, image_size, num_classes):
        """Test with NO faces detected in any image."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN in output with no faces"
        assert not torch.isinf(output).any(), "Inf in output with no faces"

    def test_some_samples_no_faces(self, model, batch_size, max_photos, image_size, num_classes):
        """Test with some samples having no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_some_faces_missing(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN with some samples missing faces"

    def test_partial_faces(self, model, batch_size, max_photos, image_size, num_classes):
        """Test with varying face counts per sample."""
        photos, photos_mask, faces, faces_mask = create_batch_partial_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN with partial faces"


class TestViTArcFaceEmptyFaces:
    """Test ViT+ArcFace with empty face masks."""

    @pytest.fixture
    def model(self, num_classes):
        config = ModelConfig(
            architecture="vit_arcface",
            backbone="vit_b_16",  # Use registered backbone
            embed_dim=256,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
            use_cross_attention=False,
        )
        model = ModelFactory.create(config)
        model.eval()
        return model

    def test_no_faces_detected(self, model, batch_size, max_photos, image_size, num_classes):
        """Test with NO faces detected."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN in ViT output with no faces"

    def test_some_samples_no_faces(self, model, batch_size, max_photos, image_size, num_classes):
        """Test with some samples having no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_some_faces_missing(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN with some samples missing faces"


class TestLightweightCPUEmptyFaces:
    """Test LightweightCPU model with empty face masks."""

    @pytest.fixture
    def model_with_attention(self, num_classes):
        """Model with attention (num_encoder_layers > 0)."""
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="efficientnet_lite0",
            embed_dim=256,
            num_encoder_layers=2,  # Uses LightweightAttention
            num_classes=num_classes,
        )
        model = ModelFactory.create(config)
        model.eval()
        return model

    @pytest.fixture
    def model_without_attention(self, num_classes):
        """Model with minimal attention (num_encoder_layers = 1)."""
        # Note: num_encoder_layers must be >= 1 per config validation
        # Testing pooling path requires model modification, so we test minimal attention
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="efficientnet_lite0",
            embed_dim=256,
            num_encoder_layers=1,  # Minimal attention
            num_classes=num_classes,
        )
        model = ModelFactory.create(config)
        model.eval()
        return model

    def test_with_attention_no_faces(
        self, model_with_attention, batch_size, max_photos, image_size, num_classes
    ):
        """Test attention model with no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model_with_attention(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN in lightweight+attention with no faces"

    def test_without_attention_no_faces(
        self, model_without_attention, batch_size, max_photos, image_size, num_classes
    ):
        """Test pooling model with no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model_without_attention(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN in lightweight+pooling with no faces"

    def test_mixed_scenarios(
        self, model_with_attention, batch_size, max_photos, image_size, num_classes
    ):
        """Test various mixed scenarios."""
        scenarios = [
            ("all_valid", create_batch_all_faces_valid),
            ("no_faces", create_batch_no_faces),
            ("some_missing", create_batch_some_faces_missing),
            ("partial", create_batch_partial_faces),
            ("no_photos_no_faces", create_batch_no_photos_no_faces),
        ]

        for name, create_fn in scenarios:
            photos, photos_mask, faces, faces_mask = create_fn(batch_size, max_photos, image_size)

            with torch.no_grad():
                output = model_with_attention(photos, photos_mask, faces, faces_mask)

            assert output.shape == (batch_size, num_classes), f"Wrong shape in {name}"
            assert not torch.isnan(output).any(), f"NaN in {name} scenario"
            assert not torch.isinf(output).any(), f"Inf in {name} scenario"


# =============================================================================
# Training Pipeline Tests
# =============================================================================


class TestTrainingWithEmptyFaces:
    """Test training loop with empty face masks."""

    @pytest.fixture
    def model(self, num_classes):
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="efficientnet_lite0",
            embed_dim=128,
            num_encoder_layers=1,
            num_classes=num_classes,
        )
        return ModelFactory.create(config)

    @pytest.fixture
    def criterion(self, num_classes):
        return get_loss_function("cross_entropy", num_classes=num_classes)

    def test_forward_backward_no_faces(
        self, model, criterion, batch_size, max_photos, image_size, num_classes
    ):
        """Test forward and backward pass with no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )
        targets = torch.randint(0, num_classes, (batch_size,))

        model.train()

        # Forward pass
        output = model(photos, photos_mask, faces, faces_mask)
        assert not torch.isnan(output).any(), "NaN in forward pass"

        # Compute loss
        loss = criterion(output, targets)
        assert not torch.isnan(loss), "NaN loss with no faces"
        assert not torch.isinf(loss), "Inf loss with no faces"

        # Backward pass
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_forward_backward_mixed_faces(
        self, model, criterion, batch_size, max_photos, image_size, num_classes
    ):
        """Test forward and backward pass with mixed face validity."""
        photos, photos_mask, faces, faces_mask = create_batch_partial_faces(
            batch_size, max_photos, image_size
        )
        targets = torch.randint(0, num_classes, (batch_size,))

        model.train()
        model.zero_grad()

        # Forward
        output = model(photos, photos_mask, faces, faces_mask)
        loss = criterion(output, targets)

        assert not torch.isnan(loss), "NaN loss with mixed faces"

        # Backward
        loss.backward()

        # Verify gradients exist and are valid
        grad_count = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), "NaN in gradients"

        assert grad_count > 0, "No gradients computed"

    def test_multiple_training_steps(
        self, model, criterion, batch_size, max_photos, image_size, num_classes
    ):
        """Test multiple training steps with varying face availability."""
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()

        batch_creators = [
            create_batch_all_faces_valid,
            create_batch_no_faces,
            create_batch_some_faces_missing,
            create_batch_partial_faces,
        ]

        losses = []
        for i, create_fn in enumerate(batch_creators):
            photos, photos_mask, faces, faces_mask = create_fn(batch_size, max_photos, image_size)
            targets = torch.randint(0, num_classes, (batch_size,))

            optimizer.zero_grad()
            output = model(photos, photos_mask, faces, faces_mask)
            loss = criterion(output, targets)

            assert not torch.isnan(loss), f"NaN loss at step {i}"
            assert not torch.isinf(loss), f"Inf loss at step {i}"

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Verify all losses are finite
        assert all(not (l != l) for l in losses), "NaN in loss history"  # NaN != NaN


# =============================================================================
# AMP (Mixed Precision) Tests
# =============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP tests")
class TestAMPWithEmptyFaces:
    """Test AMP training with empty face masks."""

    @pytest.fixture
    def model(self, num_classes, device):
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="efficientnet_lite0",
            embed_dim=128,
            num_encoder_layers=1,
            num_classes=num_classes,
        )
        model = ModelFactory.create(config)
        return model.to(device)

    @pytest.fixture
    def criterion(self, num_classes):
        return get_loss_function("cross_entropy", num_classes=num_classes)

    def test_amp_forward_no_faces(
        self, model, criterion, device, batch_size, max_photos, image_size, num_classes
    ):
        """Test AMP forward pass with no faces."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )
        photos = photos.to(device)
        photos_mask = photos_mask.to(device)
        faces = faces.to(device)
        faces_mask = faces_mask.to(device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)

        model.train()

        with torch.amp.autocast(device_type="cuda"):
            output = model(photos, photos_mask, faces, faces_mask)
            loss = criterion(output, targets)

        assert not torch.isnan(output).any(), "NaN in AMP forward"
        assert not torch.isnan(loss), "NaN loss in AMP"

    def test_amp_training_step(
        self, model, criterion, device, batch_size, max_photos, image_size, num_classes
    ):
        """Test full AMP training step with gradient scaling."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )
        photos = photos.to(device)
        photos_mask = photos_mask.to(device)
        faces = faces.to(device)
        faces_mask = faces_mask.to(device)
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()

        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            output = model(photos, photos_mask, faces, faces_mask)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        assert not torch.isnan(loss), "NaN loss in AMP training step"


# =============================================================================
# All Models Parametrized Test
# =============================================================================


@pytest.mark.parametrize(
    "architecture,backbone,embed_dim",
    [
        ("mobilenet_transformer", "mobilenet_v3_small", 256),
        ("vit_arcface", "vit_b_16", 256),
        ("lightweight_cpu", "efficientnet_lite0", 256),
    ],
)
class TestAllModelsEmptyFaces:
    """Parametrized tests for all model architectures."""

    def test_no_faces_no_nan(
        self, architecture, backbone, embed_dim, batch_size, max_photos, image_size, num_classes
    ):
        """Test that no NaN is produced with empty face masks."""
        config = ModelConfig(
            architecture=architecture,
            backbone=backbone,
            embed_dim=embed_dim,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
            use_cross_attention=(architecture == "mobilenet_transformer"),
        )
        model = ModelFactory.create(config)
        model.eval()

        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )

        with torch.no_grad():
            output = model(photos, photos_mask, faces, faces_mask)

        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), f"NaN in {architecture} with no faces"
        assert not torch.isinf(output).any(), f"Inf in {architecture} with no faces"

    def test_training_step_no_faces(
        self, architecture, backbone, embed_dim, batch_size, max_photos, image_size, num_classes
    ):
        """Test training step doesn't produce NaN."""
        config = ModelConfig(
            architecture=architecture,
            backbone=backbone,
            embed_dim=embed_dim,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
            use_cross_attention=(architecture == "mobilenet_transformer"),
        )
        model = ModelFactory.create(config)
        criterion = get_loss_function("cross_entropy", num_classes=num_classes)

        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )
        targets = torch.randint(0, num_classes, (batch_size,))

        model.train()
        output = model(photos, photos_mask, faces, faces_mask)
        loss = criterion(output, targets)

        assert not torch.isnan(loss), f"NaN loss in {architecture}"

        loss.backward()

        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                nan_grads.append(name)

        assert len(nan_grads) == 0, f"NaN gradients in {architecture}: {nan_grads}"


# =============================================================================
# Learnable No-Face Embedding Tests
# =============================================================================


class TestLearnableNoFaceEmbedding:
    """Test that learnable no-face embeddings are used correctly."""

    @pytest.fixture
    def mobilenet_model(self, num_classes):
        config = ModelConfig(
            architecture="mobilenet_transformer",
            backbone="mobilenet_v3_small",
            embed_dim=256,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
        )
        return ModelFactory.create(config)

    @pytest.fixture
    def lightweight_model(self, num_classes):
        config = ModelConfig(
            architecture="lightweight_cpu",
            backbone="efficientnet_lite0",
            embed_dim=256,
            num_encoder_layers=2,
            num_classes=num_classes,
        )
        return ModelFactory.create(config)

    def test_no_face_embedding_exists(self, mobilenet_model, lightweight_model):
        """Test that no_face_embedding parameter exists in all models."""
        assert hasattr(
            mobilenet_model, "no_face_embedding"
        ), "Missing no_face_embedding in MobileNet model"
        assert hasattr(
            mobilenet_model, "no_photo_embedding"
        ), "Missing no_photo_embedding in MobileNet model"
        assert hasattr(
            lightweight_model, "no_face_embedding"
        ), "Missing no_face_embedding in Lightweight model"
        assert hasattr(
            lightweight_model, "no_photo_embedding"
        ), "Missing no_photo_embedding in Lightweight model"

        # Verify they are nn.Parameter
        assert isinstance(mobilenet_model.no_face_embedding, nn.Parameter)
        assert isinstance(lightweight_model.no_face_embedding, nn.Parameter)

    def test_no_face_embedding_is_trainable(self, mobilenet_model, lightweight_model):
        """Test that no_face_embedding is trainable."""
        assert mobilenet_model.no_face_embedding.requires_grad
        assert mobilenet_model.no_photo_embedding.requires_grad
        assert lightweight_model.no_face_embedding.requires_grad
        assert lightweight_model.no_photo_embedding.requires_grad

    def test_no_face_embedding_receives_gradients(
        self, mobilenet_model, batch_size, max_photos, image_size, num_classes
    ):
        """Test that no_face_embedding receives gradients when faces are missing."""
        photos, photos_mask, faces, faces_mask = create_batch_no_faces(
            batch_size, max_photos, image_size
        )
        targets = torch.randint(0, num_classes, (batch_size,))
        criterion = get_loss_function("cross_entropy", num_classes=num_classes)

        mobilenet_model.train()
        mobilenet_model.zero_grad()

        output = mobilenet_model(photos, photos_mask, faces, faces_mask)
        loss = criterion(output, targets)
        loss.backward()

        # no_face_embedding should receive gradients when faces are missing
        assert (
            mobilenet_model.no_face_embedding.grad is not None
        ), "no_face_embedding should receive gradients"
        assert not torch.isnan(
            mobilenet_model.no_face_embedding.grad
        ).any(), "NaN in no_face_embedding gradient"
        assert (
            mobilenet_model.no_face_embedding.grad.abs().sum() > 0
        ), "no_face_embedding gradient should be non-zero"

    def test_output_differs_with_and_without_faces(
        self, mobilenet_model, batch_size, max_photos, image_size, num_classes
    ):
        """Test that model output differs meaningfully with and without faces."""
        mobilenet_model.eval()

        # Same photos, but different face availability
        torch.manual_seed(42)
        photos = torch.randn(batch_size, max_photos, 3, *image_size)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, *image_size)

        # With faces
        faces_mask_valid = torch.ones(batch_size, max_photos, dtype=torch.bool)
        with torch.no_grad():
            output_with_faces = mobilenet_model(photos, photos_mask, faces, faces_mask_valid)

        # Without faces
        faces_mask_invalid = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        with torch.no_grad():
            output_without_faces = mobilenet_model(photos, photos_mask, faces, faces_mask_invalid)

        # Outputs should differ (model uses different face representation)
        assert not torch.allclose(
            output_with_faces, output_without_faces, atol=1e-4
        ), "Output should differ with and without faces"

    def test_consistent_output_for_no_faces(
        self, mobilenet_model, batch_size, max_photos, image_size, num_classes
    ):
        """Test that output is consistent when no faces are present (uses learned embedding)."""
        mobilenet_model.eval()

        # Same photos, both with no faces
        torch.manual_seed(42)
        photos = torch.randn(batch_size, max_photos, 3, *image_size)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)

        # Different face tensors but both with no valid faces
        faces1 = torch.randn(batch_size, max_photos, 3, *image_size)
        faces2 = torch.randn(batch_size, max_photos, 3, *image_size) * 2  # Different values
        faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)

        with torch.no_grad():
            output1 = mobilenet_model(photos, photos_mask, faces1, faces_mask)
            output2 = mobilenet_model(photos, photos_mask, faces2, faces_mask)

        # Outputs should be the same (face tensor values don't matter when mask is all False)
        assert torch.allclose(
            output1, output2, atol=1e-5
        ), "Output should be identical when no faces (learned embedding is used)"

    def test_per_sample_no_face_handling(
        self, mobilenet_model, batch_size, max_photos, image_size, num_classes
    ):
        """Test that per-sample no-face handling works correctly."""
        mobilenet_model.eval()

        torch.manual_seed(42)
        photos = torch.randn(batch_size, max_photos, 3, *image_size)
        photos_mask = torch.ones(batch_size, max_photos, dtype=torch.bool)
        faces = torch.randn(batch_size, max_photos, 3, *image_size)

        # Only sample 0 has valid faces
        faces_mask = torch.zeros(batch_size, max_photos, dtype=torch.bool)
        faces_mask[0] = True

        with torch.no_grad():
            output = mobilenet_model(photos, photos_mask, faces, faces_mask)

        # Should produce valid output without NaN
        assert output.shape == (batch_size, num_classes)
        assert not torch.isnan(output).any(), "NaN when only some samples have faces"
        assert not torch.isinf(output).any(), "Inf when only some samples have faces"


@pytest.mark.parametrize(
    "architecture,backbone",
    [
        ("mobilenet_transformer", "mobilenet_v3_small"),
        ("vit_arcface", "vit_b_16"),
        ("lightweight_cpu", "efficientnet_lite0"),
    ],
)
class TestAllModelsHaveLearnableEmbeddings:
    """Test that all model architectures have learnable no-face embeddings."""

    def test_embedding_exists_and_trainable(self, architecture, backbone, num_classes):
        """Test embeddings exist and are trainable in all models."""
        config = ModelConfig(
            architecture=architecture,
            backbone=backbone,
            embed_dim=256,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
        )
        model = ModelFactory.create(config)

        assert hasattr(model, "no_face_embedding"), f"Missing no_face_embedding in {architecture}"
        assert hasattr(model, "no_photo_embedding"), f"Missing no_photo_embedding in {architecture}"
        assert isinstance(model.no_face_embedding, nn.Parameter)
        assert isinstance(model.no_photo_embedding, nn.Parameter)
        assert model.no_face_embedding.requires_grad
        assert model.no_photo_embedding.requires_grad

    def test_embedding_shape_matches_embed_dim(
        self, architecture, backbone, num_classes, embed_dim
    ):
        """Test embedding shape matches model embed_dim."""
        config = ModelConfig(
            architecture=architecture,
            backbone=backbone,
            embed_dim=embed_dim,
            num_encoder_layers=2,
            num_heads=4,
            num_classes=num_classes,
        )
        model = ModelFactory.create(config)

        assert model.no_face_embedding.shape == (
            embed_dim,
        ), f"no_face_embedding shape mismatch in {architecture}"
        assert model.no_photo_embedding.shape == (
            embed_dim,
        ), f"no_photo_embedding shape mismatch in {architecture}"
