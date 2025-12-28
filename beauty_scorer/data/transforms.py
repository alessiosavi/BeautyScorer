"""
Data augmentation pipelines using Albumentations.

Provides training, validation, and inference transforms with
configurable augmentation strength levels.
"""

from typing import Literal

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

from beauty_scorer.config import DataConfig


def get_train_transforms(
    config: DataConfig | None = None,
    image_size: tuple[int, int] = (224, 224),
    augmentation_strength: Literal["light", "medium", "heavy"] = "medium",
) -> A.Compose:
    """
    Get training transforms with augmentation.

    Args:
        config: Data configuration. If provided, uses config values.
        image_size: Target image size (height, width).
        augmentation_strength: Level of augmentation.

    Returns:
        Albumentations Compose object.
    """
    if config is not None:
        image_size = config.image_size
        augmentation_strength = config.augmentation_strength
        mean = config.normalize_mean
        std = config.normalize_std
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    # Base transforms
    transforms = [
        A.Resize(image_size[0], image_size[1]),
    ]

    # Augmentation based on strength
    if augmentation_strength == "light":
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            ]
        )
    elif augmentation_strength == "medium":
        transforms.extend(
            [
                A.RandomResizedCrop(
                    size=(image_size[0], image_size[1]),
                    scale=(0.85, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.3,
                ),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.1),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.1),
            ]
        )
    elif augmentation_strength == "heavy":
        transforms.extend(
            [
                A.RandomResizedCrop(
                    size=(image_size[0], image_size[1]),
                    scale=(0.7, 1.0),
                    ratio=(0.8, 1.2),
                    p=0.5,
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.15,
                    rotate_limit=15,
                    border_mode=0,
                    p=0.5,
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                        A.MotionBlur(blur_limit=(3, 7), p=1.0),
                    ],
                    p=0.2,
                ),
                A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
                A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(10, 30),
                    hole_width_range=(10, 30),
                    fill="random",
                    p=0.2,
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            ]
        )

    # Final normalization and conversion
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


def get_val_transforms(
    config: DataConfig | None = None,
    image_size: tuple[int, int] = (224, 224),
) -> A.Compose:
    """
    Get validation transforms (no augmentation).

    Args:
        config: Data configuration. If provided, uses config values.
        image_size: Target image size (height, width).

    Returns:
        Albumentations Compose object.
    """
    if config is not None:
        image_size = config.image_size
        mean = config.normalize_mean
        std = config.normalize_std
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    return A.Compose(
        [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_inference_transforms(
    config: DataConfig | None = None,
    image_size: tuple[int, int] = (224, 224),
) -> A.Compose:
    """
    Get inference transforms (same as validation).

    Args:
        config: Data configuration. If provided, uses config values.
        image_size: Target image size (height, width).

    Returns:
        Albumentations Compose object.
    """
    return get_val_transforms(config, image_size)


def get_face_transforms(
    config: DataConfig | None = None,
    face_size: tuple[int, int] = (224, 224),
    augmentation: bool = False,
) -> A.Compose:
    """
    Get transforms specifically for face crops.

    Args:
        config: Data configuration.
        face_size: Target face size (height, width).
        augmentation: Whether to apply augmentation.

    Returns:
        Albumentations Compose object.
    """
    if config is not None:
        face_size = config.face_size
        mean = config.normalize_mean
        std = config.normalize_std
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    transforms = [
        A.Resize(face_size[0], face_size[1]),
    ]

    if augmentation:
        transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
            ]
        )

    transforms.extend(
        [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    return A.Compose(transforms)


class TransformWrapper:
    """
    Wrapper for albumentations transforms to work with PyTorch-style __call__.

    This allows using albumentations transforms like torchvision transforms.
    """

    def __init__(self, transform: A.Compose):
        """
        Initialize wrapper.

        Args:
            transform: Albumentations transform.
        """
        self.transform = transform

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply transform to image.

        Args:
            image: Input image (H, W, C) or (H, W).

        Returns:
            Transformed image as tensor.
        """
        if isinstance(image, np.ndarray):
            return self.transform(image=image)["image"]
        return self.transform(image=np.array(image))["image"]
