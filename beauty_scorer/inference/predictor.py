"""
High-level inference API for beauty scoring.

Provides a simple interface for predicting beauty scores from images.
"""

import numpy as np
import torch

from beauty_scorer.config import DataConfig, ModelConfig
from beauty_scorer.data.preprocessing import FaceExtractor, load_image
from beauty_scorer.data.transforms import get_inference_transforms
from beauty_scorer.models.factory import load_model
from beauty_scorer.utils.device import get_device
from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


class BeautyPredictor:
    """
    High-level API for beauty score prediction.

    Handles image loading, face extraction, preprocessing,
    and model inference in a simple interface.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        config: DataConfig | None = None,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint.
            device: Device to run on ('auto', 'cuda', 'cpu', 'mps').
            config: Data configuration for preprocessing.
        """
        self.device = get_device(device)
        self.config = config or DataConfig()

        # Load model
        self.model = load_model(model_path, device=self.device)
        self.model.eval()

        # Setup transforms
        self.photo_transform = get_inference_transforms(self.config)
        self.face_transform = get_inference_transforms(
            config=DataConfig(image_size=self.config.face_size),
        )

        # Face extractor
        self.face_extractor = FaceExtractor(
            backend=self.config.face_detector,
            expand_percentage=self.config.face_expand_percentage,
            confidence_threshold=self.config.face_confidence_threshold,
        )

        # Model info
        model_config = self.model.config if hasattr(self.model, "config") else ModelConfig()
        self.num_classes = model_config.num_classes
        self.max_photos = self.config.max_photos

        logger.info(f"BeautyPredictor initialized: model={model_path}, device={self.device}")

    def predict_person(
        self,
        photo_paths: list[str],
        return_features: bool = False,
    ) -> dict:
        """
        Predict beauty score for a person from photo paths.

        Args:
            photo_paths: List of paths to photos of the person.
            return_features: Whether to return intermediate features.

        Returns:
            Dictionary with:
            - score: Predicted score (1-9)
            - probabilities: Dictionary mapping scores to probabilities
            - confidence: Confidence of prediction
            - features: (optional) Intermediate features
        """
        # Load and preprocess images
        photos, faces, photos_mask, faces_mask = self._preprocess_photos(photo_paths)

        # Run inference
        with torch.no_grad():
            logits = self.model(photos, photos_mask, faces, faces_mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        # Get prediction
        predicted_class = probs.argmax().item()
        predicted_score = predicted_class + 1  # 1-indexed
        confidence = probs[predicted_class].item()

        # Format probabilities
        probabilities = {i + 1: probs[i].item() for i in range(self.num_classes)}

        result = {
            "score": predicted_score,
            "probabilities": probabilities,
            "confidence": confidence,
        }

        # Get features if requested
        if return_features and hasattr(self.model, "get_intermediate_features"):
            with torch.no_grad():
                features = self.model.get_intermediate_features(
                    photos, photos_mask, faces, faces_mask
                )
                result["features"] = {k: v.cpu().numpy() for k, v in features.items()}

        return result

    def predict_from_images(
        self,
        images: list[np.ndarray],
    ) -> dict:
        """
        Predict from pre-loaded images.

        Args:
            images: List of images as numpy arrays (H, W, C).

        Returns:
            Prediction result dictionary.
        """
        # Preprocess
        photos, faces, photos_mask, faces_mask = self._preprocess_images(images)

        # Run inference
        with torch.no_grad():
            logits = self.model(photos, photos_mask, faces, faces_mask)
            probs = torch.softmax(logits, dim=-1).squeeze(0)

        predicted_class = probs.argmax().item()
        predicted_score = predicted_class + 1

        return {
            "score": predicted_score,
            "probabilities": {i + 1: probs[i].item() for i in range(self.num_classes)},
            "confidence": probs[predicted_class].item(),
        }

    def predict_batch(
        self,
        persons: list[list[str]],
    ) -> list[dict]:
        """
        Batch prediction for multiple persons.

        Args:
            persons: List of photo path lists, one per person.

        Returns:
            List of prediction result dictionaries.
        """
        results = []
        for photo_paths in persons:
            result = self.predict_person(photo_paths)
            results.append(result)
        return results

    def _preprocess_photos(
        self,
        photo_paths: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess photos from paths."""
        images = []
        for path in photo_paths[: self.max_photos]:
            try:
                img = load_image(path)
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        return self._preprocess_images(images)

    def _preprocess_images(
        self,
        images: list[np.ndarray],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess loaded images."""
        h, w = self.config.image_size
        fh, fw = self.config.face_size

        # Initialize tensors
        photos_tensor = torch.zeros(1, self.max_photos, 3, h, w, dtype=torch.float32)
        photos_mask = torch.zeros(1, self.max_photos, dtype=torch.bool)
        faces_tensor = torch.zeros(1, self.max_photos, 3, fh, fw, dtype=torch.float32)
        faces_mask = torch.zeros(1, self.max_photos, dtype=torch.bool)

        for i, img in enumerate(images[: self.max_photos]):
            # Process photo
            photo_processed = self.photo_transform(image=img)["image"]
            photos_tensor[0, i] = photo_processed
            photos_mask[0, i] = True

            # Extract and process face
            face = self.face_extractor.extract(img)
            if face is not None:
                face_processed = self.face_transform(image=face)["image"]
                faces_tensor[0, i] = face_processed
                faces_mask[0, i] = True

        # Move to device
        photos_tensor = photos_tensor.to(self.device)
        photos_mask = photos_mask.to(self.device)
        faces_tensor = faces_tensor.to(self.device)
        faces_mask = faces_mask.to(self.device)

        return photos_tensor, faces_tensor, photos_mask, faces_mask

    def get_top_predictions(
        self,
        photo_paths: list[str],
        top_k: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Get top-k predictions with probabilities.

        Args:
            photo_paths: List of photo paths.
            top_k: Number of top predictions to return.

        Returns:
            List of (score, probability) tuples.
        """
        result = self.predict_person(photo_paths)
        probs = result["probabilities"]

        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_k]

    def explain_prediction(
        self,
        photo_paths: list[str],
    ) -> dict:
        """
        Get detailed explanation of prediction.

        Args:
            photo_paths: List of photo paths.

        Returns:
            Detailed prediction explanation.
        """
        result = self.predict_person(photo_paths, return_features=True)

        # Get top predictions
        probs = result["probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        explanation = {
            "predicted_score": result["score"],
            "confidence": result["confidence"],
            "top_3_predictions": sorted_probs[:3],
            "score_range_confidence": sum(
                p for s, p in probs.items() if abs(s - result["score"]) <= 1
            ),
            "num_photos_used": len(photo_paths),
        }

        return explanation


def predict_single(
    model_path: str,
    photo_paths: list[str],
    device: str = "auto",
) -> dict:
    """
    Convenience function for single prediction.

    Args:
        model_path: Path to model checkpoint.
        photo_paths: List of photo paths.
        device: Device to run on.

    Returns:
        Prediction result.
    """
    predictor = BeautyPredictor(model_path, device=device)
    return predictor.predict_person(photo_paths)
