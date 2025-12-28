#!/usr/bin/env python
"""
Evaluation script for BeautyScorer models.

Usage:
    python scripts/evaluate.py --model outputs/checkpoints/best.pt --data datasets/test.csv
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from beauty_scorer.config import DataConfig, load_config
from beauty_scorer.data.dataset import BeautyDataset, load_dataset_from_csv
from beauty_scorer.models.factory import load_model
from beauty_scorer.training.metrics import MetricTracker
from beauty_scorer.utils.device import get_device
from beauty_scorer.utils.logging import get_logger, setup_logging
from beauty_scorer.utils.visualization import plot_confusion_matrix, plot_score_distribution


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate BeautyScorer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to evaluation dataset CSV",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Base path to images",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for data settings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to JSON",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup
    setup_logging()
    logger = get_logger(__name__)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    if args.config:
        config = load_config(args.config)
        data_config = config.data
    else:
        data_config = DataConfig()

    # Device
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model, device=device)
    model.eval()

    model_config = model.config if hasattr(model, "config") else None
    num_classes = model_config.num_classes if model_config else 9

    # Load dataset
    logger.info(f"Loading dataset from {args.data}")
    data = load_dataset_from_csv(
        csv_path=args.data,
        base_path=args.data_path,
    )
    logger.info(f"Loaded {len(data)} samples")

    # Create data loader
    dataset = BeautyDataset(data, config=data_config, is_training=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=BeautyDataset.collate_fn,
    )

    # Evaluate
    logger.info("Running evaluation...")
    tracker = MetricTracker(num_classes=num_classes)
    predictions_list = []

    with torch.no_grad():
        for batch in loader:
            photos = batch["photos_tensor"].to(device)
            photos_mask = batch["photos_mask"].to(device)
            faces = batch["faces_tensor"].to(device)
            faces_mask = batch["faces_mask"].to(device)
            targets = batch["scores"].to(device) - 1  # 0-indexed

            logits = model(photos, photos_mask, faces, faces_mask)
            preds = logits.argmax(dim=-1)

            tracker.update(preds, targets)

            if args.save_predictions:
                probs = torch.softmax(logits, dim=-1)
                for _, (pred, target, prob, person_id) in enumerate(
                    zip(preds, targets, probs, batch["ids"], strict=False)
                ):
                    predictions_list.append(
                        {
                            "id": person_id,
                            "predicted": int(pred.item() + 1),
                            "actual": int(target.item() + 1),
                            "probabilities": {
                                j + 1: round(prob[j].item(), 4) for j in range(num_classes)
                            },
                        }
                    )

    # Compute metrics
    metrics = tracker.compute()
    confusion = tracker.get_confusion_matrix()
    per_class_acc = tracker.get_per_class_accuracy()

    # Log results
    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"Within 1: {metrics['within_1']:.4f}")
    logger.info(f"Within 2: {metrics['within_2']:.4f}")
    logger.info("=" * 50)
    logger.info("Per-class accuracy:")
    for cls, acc in per_class_acc.items():
        logger.info(f"  Score {cls + 1}: {acc:.4f}")

    # Save results
    results = {
        "metrics": {k: float(v) for k, v in metrics.items() if k != "confusion_matrix"},
        "per_class_accuracy": {str(k + 1): float(v) for k, v in per_class_acc.items()},
        "num_samples": len(data),
        "model_path": args.model,
        "data_path": args.data,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to {output_dir / 'metrics.json'}")

    # Save predictions
    if args.save_predictions:
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions_list, f, indent=2)
        logger.info(f"Predictions saved to {output_dir / 'predictions.json'}")

    # Plot confusion matrix
    class_names = [str(i + 1) for i in range(num_classes)]
    plot_confusion_matrix(
        confusion,
        class_names=class_names,
        save_path=output_dir / "confusion_matrix.png",
        show=False,
        normalize=True,
    )
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    # Plot score distribution
    all_preds = [p["predicted"] for p in predictions_list] if predictions_list else []
    all_targets = [p["actual"] for p in predictions_list] if predictions_list else []
    if all_preds:
        plot_score_distribution(
            all_preds,
            true_labels=all_targets,
            num_classes=num_classes,
            save_path=output_dir / "score_distribution.png",
            show=False,
        )
        logger.info(f"Score distribution saved to {output_dir / 'score_distribution.png'}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
