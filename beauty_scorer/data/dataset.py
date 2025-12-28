"""
Dataset classes for BeautyScorer.

Provides PyTorch Dataset implementations for loading and processing
beauty scoring data with variable-length photo sequences.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from beauty_scorer.config import DataConfig
from beauty_scorer.data.preprocessing import FaceExtractor, load_image
from beauty_scorer.data.transforms import (
    get_face_transforms,
    get_train_transforms,
    get_val_transforms,
)
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class BeautyDataset(Dataset):
    """
    Dataset for beauty score prediction.

    Each sample contains:
    - Variable number of photos of a person
    - Extracted faces from each photo
    - Beauty score label (1-9)

    The dataset handles:
    - Loading and caching images
    - Face extraction
    - Data augmentation
    - Padding to fixed sequence length
    """

    def __init__(
        self,
        data: list[dict],
        config: DataConfig | None = None,
        transform=None,
        face_transform=None,
        is_training: bool = True,
        cache_faces: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            data: List of dicts with keys: 'id', 'photos' (list of paths), 'score'.
            config: Data configuration.
            transform: Transform for photos. If None, uses default based on is_training.
            face_transform: Transform for faces. If None, uses default.
            is_training: Whether this is training data (enables augmentation).
            cache_faces: Whether to cache extracted faces (uses more memory).
        """
        super().__init__()
        self.data = data
        self.config = config or DataConfig()
        self.is_training = is_training
        self.cache_faces = cache_faces

        # Set up transforms
        if transform is None:
            if is_training and self.config.augmentation:
                self.transform = get_train_transforms(self.config)
            else:
                self.transform = get_val_transforms(self.config)
        else:
            self.transform = transform

        if face_transform is None:
            self.face_transform = get_face_transforms(
                self.config,
                augmentation=is_training and self.config.augmentation,
            )
        else:
            self.face_transform = face_transform

        # Face extractor
        self.face_extractor = FaceExtractor(
            backend=self.config.face_detector,
            expand_percentage=self.config.face_expand_percentage,
            confidence_threshold=self.config.face_confidence_threshold,
        )

        # Face cache
        self._face_cache: dict[str, np.ndarray | None] = {}

        logger.info(
            f"Created BeautyDataset with {len(data)} samples, "
            f"is_training={is_training}, cache_faces={cache_faces}"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns:
            Dictionary with keys:
            - id: Person ID
            - photos_tensor: (max_photos, C, H, W) tensor
            - photos_mask: (max_photos,) boolean mask
            - faces_tensor: (max_photos, C, H, W) tensor
            - faces_mask: (max_photos,) boolean mask
            - score: Integer score (1-9)
        """
        entry = self.data[idx]
        photo_paths = entry["photos"]
        score = entry["score"]
        person_id = entry["id"]

        # Limit and optionally shuffle photos
        if self.is_training:
            photo_paths = random.sample(photo_paths, min(len(photo_paths), self.config.max_photos))
        else:
            photo_paths = photo_paths[: self.config.max_photos]

        # Load images
        raw_photos = []
        for path in photo_paths:
            try:
                img = load_image(path)
                raw_photos.append((path, img))
            except Exception as e:
                logger.debug(f"Failed to load image {path}: {e}")

        # Get dimensions from config
        h, w = self.config.image_size
        fh, fw = self.config.face_size
        max_photos = self.config.max_photos

        # Initialize tensors
        photos_tensor = torch.zeros(max_photos, 3, h, w, dtype=torch.float32)
        photos_mask = torch.zeros(max_photos, dtype=torch.bool)
        faces_tensor = torch.zeros(max_photos, 3, fh, fw, dtype=torch.float32)
        faces_mask = torch.zeros(max_photos, dtype=torch.bool)

        # Process each photo
        for i, (path, raw_photo) in enumerate(raw_photos):
            # Apply transform to photo
            transformed = self.transform(image=raw_photo)["image"]
            photos_tensor[i] = transformed
            photos_mask[i] = True

            # Extract and process face
            face = self._get_face(path, raw_photo)
            if face is not None:
                face_transformed = self.face_transform(image=face)["image"]
                faces_tensor[i] = face_transformed
                faces_mask[i] = True

        return {
            "id": person_id,
            "photos_tensor": photos_tensor,
            "photos_mask": photos_mask,
            "faces_tensor": faces_tensor,
            "faces_mask": faces_mask,
            "score": score,
        }

    def _get_face(self, path: str, image: np.ndarray) -> np.ndarray | None:
        """Get face from image, using cache if available."""
        if self.cache_faces and path in self._face_cache:
            return self._face_cache[path]

        face = self.face_extractor.extract(image)

        if self.cache_faces:
            self._face_cache[path] = face

        return face

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Custom collate function for DataLoader.

        Stacks individual samples into batched tensors.

        Args:
            batch: List of sample dictionaries.

        Returns:
            Batched dictionary.
        """
        return {
            "ids": [item["id"] for item in batch],
            "photos_tensor": torch.stack([item["photos_tensor"] for item in batch]),
            "photos_mask": torch.stack([item["photos_mask"] for item in batch]),
            "faces_tensor": torch.stack([item["faces_tensor"] for item in batch]),
            "faces_mask": torch.stack([item["faces_mask"] for item in batch]),
            "scores": torch.tensor([item["score"] for item in batch], dtype=torch.long),
        }


def load_dataset_from_csv(
    csv_path: str | Path,
    base_path: str | Path,
    folder_column: str = "folder_name",
    score_column: str = "score",
    min_score: int | None = None,
    max_score: int | None = None,
) -> list[dict]:
    """
    Load dataset from CSV file.

    Args:
        csv_path: Path to CSV file with folder names and scores.
        base_path: Base path where image folders are located.
        folder_column: Column name for folder names.
        score_column: Column name for scores.
        min_score: Minimum score to include.
        max_score: Maximum score to include.

    Returns:
        List of data dictionaries.
    """
    csv_path = Path(csv_path)
    base_path = Path(base_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=[folder_column, score_column])
    df = df.dropna().drop_duplicates().reset_index(drop=True)

    # Filter by score range
    if min_score is not None:
        df = df[df[score_column] >= min_score]
    if max_score is not None:
        df = df[df[score_column] <= max_score]

    dataset = []
    for _, row in df.iterrows():
        folder = row[folder_column]
        score = int(row[score_column])

        # Find images in folder
        folder_path = base_path / str(folder)
        images = (
            list(folder_path.glob("*.jpg"))
            + list(folder_path.glob("*.jpeg"))
            + list(folder_path.glob("*.png"))
        )

        if images:
            dataset.append(
                {
                    "id": folder,
                    "score": score,
                    "photos": [str(img) for img in images],
                }
            )

    logger.info(f"Loaded {len(dataset)} samples from {csv_path}")
    return dataset


def create_data_loaders(
    train_data: list[dict],
    val_data: list[dict] | None = None,
    config: DataConfig | None = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create training and validation data loaders.

    Args:
        train_data: Training data list.
        val_data: Validation data list (optional).
        config: Data configuration.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    config = config or DataConfig()

    train_dataset = BeautyDataset(
        train_data,
        config=config,
        is_training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=BeautyDataset.collate_fn,
        drop_last=True,
    )

    val_loader = None
    if val_data:
        val_dataset = BeautyDataset(
            val_data,
            config=config,
            is_training=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=BeautyDataset.collate_fn,
        )

    return train_loader, val_loader


def train_val_split(
    data: list[dict],
    val_ratio: float = 0.1,
    stratify: bool = True,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split data into training and validation sets.

    Args:
        data: Full dataset.
        val_ratio: Ratio of data to use for validation.
        stratify: Whether to stratify by score.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, val_data).
    """
    random.seed(seed)

    if stratify:
        # Group by score
        score_groups: dict[int, list[dict]] = {}
        for item in data:
            score = item["score"]
            if score not in score_groups:
                score_groups[score] = []
            score_groups[score].append(item)

        train_data = []
        val_data = []

        for items in score_groups.values():
            random.shuffle(items)
            n_val = max(1, int(len(items) * val_ratio))
            val_data.extend(items[:n_val])
            train_data.extend(items[n_val:])
    else:
        shuffled = data.copy()
        random.shuffle(shuffled)
        n_val = int(len(shuffled) * val_ratio)
        val_data = shuffled[:n_val]
        train_data = shuffled[n_val:]

    logger.info(f"Split data: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def compute_class_weights(
    data: list[dict],
    num_classes: int = 9,
    min_score: int = 1,
    max_weight: float = 5.0,
) -> torch.Tensor:
    """
    Compute class weights for imbalanced data.

    Args:
        data: Dataset list.
        num_classes: Number of classes.
        min_score: Minimum score value.
        max_weight: Maximum weight to prevent instability.

    Returns:
        Tensor of class weights.
    """
    counts = torch.zeros(num_classes)

    for item in data:
        score = item["score"]
        idx = score - min_score
        if 0 <= idx < num_classes:
            counts[idx] += 1

    # Avoid division by zero
    counts = counts + 1e-6

    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (num_classes * counts)

    # Clamp weights
    weights = torch.clamp(weights, max=max_weight)

    return weights
