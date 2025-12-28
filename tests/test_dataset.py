"""
Tests for dataset and data loading.

Tests:
- Dataset creation
- Collate function
- Data transformations
- Train/val splitting
"""

import numpy as np
import pytest
import torch

from beauty_scorer.config import DataConfig
from beauty_scorer.data.dataset import BeautyDataset, compute_class_weights, train_val_split
from beauty_scorer.data.transforms import get_train_transforms, get_val_transforms


# Test fixtures
@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create fake image data (in-memory)
    return [
        {"id": "person1", "photos": ["fake_path1.jpg", "fake_path2.jpg"], "score": 5},
        {"id": "person2", "photos": ["fake_path3.jpg"], "score": 7},
        {
            "id": "person3",
            "photos": ["fake_path4.jpg", "fake_path5.jpg", "fake_path6.jpg"],
            "score": 3,
        },
    ]


@pytest.fixture
def data_config():
    """Create data configuration."""
    return DataConfig(
        image_size=(224, 224),
        face_size=(224, 224),
        max_photos=4,
        augmentation=False,
    )


class TestDataTransforms:
    """Tests for data transformations."""

    def test_train_transforms(self):
        """Test training transforms creation."""
        transforms = get_train_transforms(image_size=(224, 224))
        assert transforms is not None

        # Apply to dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transforms(image=image)
        assert "image" in result
        assert result["image"].shape == (3, 224, 224)

    def test_val_transforms(self):
        """Test validation transforms creation."""
        transforms = get_val_transforms(image_size=(224, 224))
        assert transforms is not None

        # Apply to dummy image
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = transforms(image=image)
        assert result["image"].shape == (3, 224, 224)

    def test_transform_normalization(self):
        """Test that transforms normalize correctly."""
        transforms = get_val_transforms(image_size=(224, 224))

        # Create white image
        image = np.ones((224, 224, 3), dtype=np.uint8) * 255
        result = transforms(image=image)

        # Should be normalized (not in 0-255 range)
        assert result["image"].max() < 10
        assert result["image"].min() > -10

    def test_augmentation_strength_levels(self):
        """Test different augmentation strength levels."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        for strength in ["light", "medium", "heavy"]:
            transforms = get_train_transforms(
                image_size=(224, 224),
                augmentation_strength=strength,
            )
            result = transforms(image=image)
            assert result["image"].shape == (3, 224, 224)


class TestTrainValSplit:
    """Tests for train/validation splitting."""

    def test_basic_split(self, sample_data):
        """Test basic train/val split."""
        train_data, val_data = train_val_split(
            sample_data,
            val_ratio=0.34,  # With 3 samples, int(3*0.34)=1 ensures at least 1 val sample
            stratify=False,
            seed=42,
        )

        assert len(train_data) + len(val_data) == len(sample_data)
        assert len(val_data) >= 1

    def test_stratified_split(self):
        """Test stratified splitting."""
        # Create data with multiple samples per class
        data = []
        for score in range(1, 10):
            for i in range(10):
                data.append(
                    {
                        "id": f"person_{score}_{i}",
                        "photos": [f"photo_{score}_{i}.jpg"],
                        "score": score,
                    }
                )

        train_data, val_data = train_val_split(
            data,
            val_ratio=0.2,
            stratify=True,
            seed=42,
        )

        # Check that each class is represented in validation
        val_scores = set(item["score"] for item in val_data)
        assert len(val_scores) >= 5  # Most classes should be present

    def test_reproducibility(self, sample_data):
        """Test that splitting is reproducible with same seed."""
        train1, val1 = train_val_split(sample_data, val_ratio=0.33, seed=42)
        train2, val2 = train_val_split(sample_data, val_ratio=0.33, seed=42)

        assert [d["id"] for d in train1] == [d["id"] for d in train2]
        assert [d["id"] for d in val1] == [d["id"] for d in val2]


class TestClassWeights:
    """Tests for class weight computation."""

    def test_compute_weights(self):
        """Test class weight computation."""
        data = [
            {"id": "p1", "photos": [], "score": 1},
            {"id": "p2", "photos": [], "score": 1},
            {"id": "p3", "photos": [], "score": 5},
            {"id": "p4", "photos": [], "score": 9},
        ]

        weights = compute_class_weights(data, num_classes=9, min_score=1)
        assert weights.shape == (9,)
        assert weights[0] < weights[4]  # Score 1 (index 0) has more samples, lower weight

    def test_weights_normalization(self):
        """Test that weights are properly bounded."""
        data = [
            {"id": "p1", "photos": [], "score": 1},
        ] * 100 + [
            {"id": "p2", "photos": [], "score": 9},
        ]

        weights = compute_class_weights(data, num_classes=9, max_weight=5.0)
        assert weights.max() <= 5.0


class TestCollateFunction:
    """Tests for dataset collate function."""

    def test_collate_basic(self):
        """Test basic collation."""
        # Create mock dataset items
        batch = [
            {
                "id": "p1",
                "photos_tensor": torch.randn(4, 3, 224, 224),
                "photos_mask": torch.tensor([True, True, False, False]),
                "faces_tensor": torch.randn(4, 3, 224, 224),
                "faces_mask": torch.tensor([True, False, False, False]),
                "score": 5,
            },
            {
                "id": "p2",
                "photos_tensor": torch.randn(4, 3, 224, 224),
                "photos_mask": torch.tensor([True, True, True, False]),
                "faces_tensor": torch.randn(4, 3, 224, 224),
                "faces_mask": torch.tensor([True, True, False, False]),
                "score": 7,
            },
        ]

        collated = BeautyDataset.collate_fn(batch)

        assert collated["photos_tensor"].shape == (2, 4, 3, 224, 224)
        assert collated["photos_mask"].shape == (2, 4)
        assert collated["faces_tensor"].shape == (2, 4, 3, 224, 224)
        assert collated["faces_mask"].shape == (2, 4)
        assert collated["scores"].shape == (2,)
        assert len(collated["ids"]) == 2

    def test_collate_preserves_values(self):
        """Test that collation preserves tensor values."""
        tensor1 = torch.randn(4, 3, 224, 224)
        tensor2 = torch.randn(4, 3, 224, 224)

        batch = [
            {
                "id": "p1",
                "photos_tensor": tensor1,
                "photos_mask": torch.ones(4, dtype=torch.bool),
                "faces_tensor": tensor1,
                "faces_mask": torch.ones(4, dtype=torch.bool),
                "score": 1,
            },
            {
                "id": "p2",
                "photos_tensor": tensor2,
                "photos_mask": torch.ones(4, dtype=torch.bool),
                "faces_tensor": tensor2,
                "faces_mask": torch.ones(4, dtype=torch.bool),
                "score": 2,
            },
        ]

        collated = BeautyDataset.collate_fn(batch)

        assert torch.allclose(collated["photos_tensor"][0], tensor1)
        assert torch.allclose(collated["photos_tensor"][1], tensor2)
