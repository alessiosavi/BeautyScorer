"""
Image preprocessing utilities including face extraction.

Provides functions for loading images, extracting faces using various
detection backends, and preparing data for the model.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


def load_image(
    path: str | Path,
    to_rgb: bool = True,
    as_numpy: bool = True,
) -> np.ndarray | Image.Image:
    """
    Load an image from file.

    Args:
        path: Path to image file.
        to_rgb: Convert to RGB if needed.
        as_numpy: Return as numpy array instead of PIL Image.

    Returns:
        Image as numpy array (H, W, C) or PIL Image.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)

    if to_rgb and img.mode != "RGB":
        img = img.convert("RGB")

    if as_numpy:
        return np.array(img)

    return img


def load_image_tensor(
    path: str | Path,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Load image as a tensor.

    Args:
        path: Path to image file.
        device: Device to place tensor on.

    Returns:
        Image tensor (C, H, W) with values in [0, 255].
    """
    from torchvision.io import read_image

    tensor = read_image(str(path))

    if device is not None:
        tensor = tensor.to(device)

    return tensor


class FaceExtractor:
    """
    Face extraction using various detection backends.

    Supports YOLOv8, RetinaFace, MTCNN, and OpenCV backends
    through DeepFace or direct implementations.
    """

    def __init__(
        self,
        backend: Literal["yolov8", "retinaface", "mtcnn", "opencv"] = "yolov8",
        expand_percentage: int = 50,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize face extractor.

        Args:
            backend: Face detection backend.
            expand_percentage: Percentage to expand face crop.
            confidence_threshold: Minimum detection confidence.
        """
        self.backend = backend
        self.expand_percentage = expand_percentage
        self.confidence_threshold = confidence_threshold
        self._deepface_available = None

    @property
    def deepface_available(self) -> bool:
        """Check if DeepFace is available."""
        if self._deepface_available is None:
            try:
                import deepface  # noqa: F401

                self._deepface_available = True
            except ImportError:
                self._deepface_available = False
        return self._deepface_available

    def extract(
        self,
        image: np.ndarray | torch.Tensor | str | Path,
    ) -> np.ndarray | None:
        """
        Extract face from image.

        Args:
            image: Input image (numpy array, tensor, or path).

        Returns:
            Face crop as numpy array (H, W, C), or None if no face found.
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = load_image(image)

        # Convert tensor to numpy
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Use DeepFace if available, with OpenCV fallback
        if self.deepface_available:
            result = self._extract_deepface(image)
            if result is not None:
                return result
            # Fall through to OpenCV if DeepFace failed

        # Fallback to OpenCV
        return self._extract_opencv(image)

    def _extract_deepface(self, image: np.ndarray) -> np.ndarray | None:
        """Extract face using DeepFace."""
        try:
            from deepface import DeepFace

            results = DeepFace.extract_faces(
                image,
                detector_backend=self.backend,
                enforce_detection=False,
                expand_percentage=self.expand_percentage,
                color_face="rgb",  # Return RGB format
            )

            if not results:
                return None

            # Get best detection
            best_result = max(results, key=lambda x: x.get("confidence", 0))

            if best_result.get("confidence", 0) < self.confidence_threshold:
                return None

            face = best_result["face"]

            # Ensure proper format
            if face.dtype != np.uint8:
                if face.max() <= 1.0:
                    face = (face * 255).astype(np.uint8)
                else:
                    face = face.astype(np.uint8)

            return face

        except Exception as e:
            logger.debug(f"Face extraction failed: {e}")
            return None

    def _extract_opencv(self, image: np.ndarray) -> np.ndarray | None:
        """Extract face using OpenCV Haar cascades."""
        try:
            import cv2

            # Load cascade classifier
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            if len(faces) == 0:
                return None

            # Get largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

            # Expand region
            expand = self.expand_percentage / 100
            dx = int(w * expand / 2)
            dy = int(h * expand / 2)

            x1 = max(0, x - dx)
            y1 = max(0, y - dy)
            x2 = min(image.shape[1], x + w + dx)
            y2 = min(image.shape[0], y + h + dy)

            return image[y1:y2, x1:x2].copy()

        except Exception as e:
            logger.debug(f"OpenCV face extraction failed: {e}")
            return None

    def extract_batch(
        self,
        images: list[np.ndarray | torch.Tensor | str | Path],
    ) -> list[np.ndarray | None]:
        """
        Extract faces from multiple images.

        Args:
            images: List of input images.

        Returns:
            List of face crops (or None for images where no face was found).
        """
        return [self.extract(img) for img in images]


def extract_face(
    image: np.ndarray | torch.Tensor | str | Path,
    backend: Literal["yolov8", "retinaface", "mtcnn", "opencv"] = "yolov8",
    expand_percentage: int = 50,
    confidence_threshold: float = 0.3,
) -> np.ndarray | None:
    """
    Convenience function to extract face from image.

    Args:
        image: Input image.
        backend: Face detection backend.
        expand_percentage: Percentage to expand face crop.
        confidence_threshold: Minimum detection confidence.

    Returns:
        Face crop as numpy array, or None if no face found.
    """
    extractor = FaceExtractor(
        backend=backend,
        expand_percentage=expand_percentage,
        confidence_threshold=confidence_threshold,
    )
    return extractor.extract(image)


def prepare_image_batch(
    images: list[np.ndarray],
    max_images: int,
    transform,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of images with padding.

    Args:
        images: List of images as numpy arrays.
        max_images: Maximum number of images (for padding).
        transform: Transform to apply to each image.

    Returns:
        Tuple of (images_tensor, mask_tensor).
        - images_tensor: (max_images, C, H, W)
        - mask_tensor: (max_images,) boolean mask where True = valid image
    """
    n_images = min(len(images), max_images)

    # Apply transforms to get tensor dimensions
    sample = transform(image=images[0])["image"]
    c, h, w = sample.shape

    # Initialize tensors
    images_tensor = torch.zeros(max_images, c, h, w, dtype=torch.float32)
    mask_tensor = torch.zeros(max_images, dtype=torch.bool)

    for i in range(n_images):
        transformed = transform(image=images[i])["image"]
        images_tensor[i] = transformed
        mask_tensor[i] = True

    return images_tensor, mask_tensor
