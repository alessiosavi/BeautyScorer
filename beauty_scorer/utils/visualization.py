"""
Visualization utilities for BeautyScorer.

Provides functions for plotting training history, confusion matrices,
and visualizing model predictions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from beauty_scorer.utils.logging import get_logger

logger = get_logger(__name__)


def plot_training_history(
    history: dict,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot training history curves.

    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path: Path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    ax1 = axes[0]
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss", color="blue")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2 = axes[1]
    if "train_acc" in history:
        ax2.plot(history["train_acc"], label="Train Acc", color="blue")
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="Val Acc", color="orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training history plot saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    normalize: bool = False,
    cmap: str = "Blues",
) -> plt.Figure:
    """
    Plot confusion matrix.

    Args:
        confusion_matrix: NxN confusion matrix array.
        class_names: Names for each class.
        save_path: Path to save the figure.
        show: Whether to display the figure.
        normalize: Whether to normalize the matrix.
        cmap: Colormap name.

    Returns:
        Matplotlib figure object.
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype("float") / confusion_matrix.sum(
            axis=1, keepdims=True
        )
        confusion_matrix = np.nan_to_num(confusion_matrix)

    num_classes = confusion_matrix.shape[0]
    if class_names is None:
        class_names = [str(i + 1) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_matrix, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = ".2f" if normalize else "d"
    thresh = confusion_matrix.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                format(confusion_matrix[i, j], fmt),
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > thresh else "black",
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    if show:
        plt.show()

    return fig


def visualize_predictions(
    images: list[np.ndarray | torch.Tensor | Image.Image],
    predictions: list[int],
    probabilities: list[np.ndarray] | None = None,
    true_labels: list[int] | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
    max_images: int = 9,
) -> plt.Figure:
    """
    Visualize model predictions on images.

    Args:
        images: List of images (numpy arrays, tensors, or PIL Images).
        predictions: List of predicted class indices.
        probabilities: List of probability distributions for each prediction.
        true_labels: List of true labels (if available).
        save_path: Path to save the figure.
        show: Whether to display the figure.
        max_images: Maximum number of images to display.

    Returns:
        Matplotlib figure object.
    """
    n_images = min(len(images), max_images)
    n_cols = min(3, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= n_images:
            ax.axis("off")
            continue

        img = images[idx]
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
        elif isinstance(img, Image.Image):
            img = np.array(img)

        ax.imshow(img)
        ax.axis("off")

        # Build title
        pred = predictions[idx]
        title = f"Pred: {pred}"

        if true_labels is not None:
            true = true_labels[idx]
            title += f" | True: {true}"
            color = "green" if pred == true else "red"
        else:
            color = "black"

        if probabilities is not None:
            prob = probabilities[idx]
            if isinstance(prob, torch.Tensor):
                prob = prob.cpu().numpy()
            confidence = prob.max() * 100
            title += f"\nConf: {confidence:.1f}%"

        ax.set_title(title, color=color, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Prediction visualization saved to {save_path}")

    if show:
        plt.show()

    return fig


def show_image(
    image: np.ndarray | torch.Tensor | Image.Image | str,
    title: str = "",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Display a single image.

    Args:
        image: Image to display (array, tensor, PIL Image, or path).
        title: Title for the image.
        save_path: Path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        if image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.cpu().numpy()
        # Denormalize if needed
        if image.max() <= 1.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image, 0, 1)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Image saved to {save_path}")

    if show:
        plt.show()

    return fig


def plot_score_distribution(
    predictions: list[int] | np.ndarray,
    true_labels: list[int] | np.ndarray | None = None,
    num_classes: int = 9,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot distribution of predicted vs true scores.

    Args:
        predictions: List of predicted scores.
        true_labels: List of true scores (if available).
        num_classes: Number of score classes.
        save_path: Path to save the figure.
        show: Whether to display the figure.

    Returns:
        Matplotlib figure object.
    """
    predictions = np.array(predictions)
    bins = np.arange(1, num_classes + 2) - 0.5

    fig, ax = plt.subplots(figsize=(10, 6))

    if true_labels is not None:
        true_labels = np.array(true_labels)
        ax.hist(
            [predictions, true_labels],
            bins=bins,
            label=["Predicted", "True"],
            alpha=0.7,
            edgecolor="black",
        )
    else:
        ax.hist(predictions, bins=bins, label="Predicted", alpha=0.7, edgecolor="black")

    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.set_xticks(range(1, num_classes + 1))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Score distribution plot saved to {save_path}")

    if show:
        plt.show()

    return fig
