#!/usr/bin/env python
"""
Training script for BeautyScorer models.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/cpu_training.yaml --model lightweight_cpu
    python scripts/train.py --config configs/advanced_model.yaml --model vit_arcface
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from beauty_scorer.config import BeautyConfig, load_config
from beauty_scorer.data.dataset import (
    compute_class_weights,
    create_data_loaders,
    get_class_distribution,
    load_dataset_from_csv,
    sample_balanced_dataset,
    train_val_split,
)
from beauty_scorer.models.factory import create_model
from beauty_scorer.training.callbacks import EarlyStopping, LRMonitor, ModelCheckpoint, ProgressBar
from beauty_scorer.training.trainer import Trainer
from beauty_scorer.utils.device import set_seed
from beauty_scorer.utils.logging import get_logger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train BeautyScorer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        choices=["mobilenet_transformer", "vit_arcface", "lightweight_cpu"],
        default=None,
        help="Model architecture (overrides config)",
    )

    # Data
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset CSV file (overrides config)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to image directory (overrides config)",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on (auto, cuda, cpu, mps)",
    )

    # Sampling
    parser.add_argument(
        "--sample-size",
        type=str,
        default=None,
        help=(
            "Sample size for dataset. Use float (0-1) for percentage (e.g., '0.1' = 10%%), "
            "or int for absolute count (e.g., '1000'). Overrides config."
        ),
    )
    parser.add_argument(
        "--balance-classes",
        action="store_true",
        help="Balance class distribution when sampling",
    )
    parser.add_argument(
        "--balance-strategy",
        type=str,
        choices=["undersample", "sqrt", "proportional"],
        default=None,
        help="Strategy for class balancing (overrides config)",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = BeautyConfig()

    # Apply CLI overrides
    if args.model:
        config.model.architecture = args.model
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.seed:
        config.training.seed = args.seed
    if args.dataset:
        config.paths.dataset_file = args.dataset
    if args.data_path:
        config.paths.dataset_base_path = args.data_path
    if args.output_dir:
        config.paths.output_dir = args.output_dir

    # Apply sampling overrides
    if args.sample_size is not None:
        # Parse sample_size: float if contains '.', else int
        if "." in args.sample_size:
            config.data.sample_size = float(args.sample_size)
        else:
            config.data.sample_size = int(args.sample_size)
    if args.balance_classes:
        config.data.balance_classes = True
    if args.balance_strategy:
        config.data.balance_strategy = args.balance_strategy

    # Setup logging
    output_dir = Path(config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_level=config.logging.log_level,
        log_dir=config.logging.log_dir,
    )
    logger = get_logger(__name__)

    # Set seed
    set_seed(config.training.seed)

    logger.info(f"Configuration: {config.model.architecture}")
    logger.info(f"Output directory: {output_dir}")

    # Load dataset
    logger.info(f"Loading dataset from {config.paths.dataset_file}")
    data = load_dataset_from_csv(
        csv_path=config.paths.dataset_file,
        base_path=config.paths.dataset_base_path,
    )
    logger.info(f"Loaded {len(data)} samples")

    # Apply dataset sampling if configured
    if config.data.sample_size is not None or config.data.balance_classes:
        logger.info(
            f"Sampling dataset: size={config.data.sample_size}, "
            f"balance={config.data.balance_classes}, "
            f"strategy={config.data.balance_strategy}"
        )
        original_dist = get_class_distribution(data)
        logger.info(f"Original class distribution: {original_dist}")

        data = sample_balanced_dataset(
            data,
            sample_size=config.data.sample_size,
            balance_classes=config.data.balance_classes,
            balance_strategy=config.data.balance_strategy,
            seed=config.training.seed,
        )

        sampled_dist = get_class_distribution(data)
        logger.info(f"Sampled class distribution: {sampled_dist}")

    # Split data
    train_data, val_data = train_val_split(
        data,
        val_ratio=config.training.val_split,
        stratify=True,
        seed=config.training.seed,
    )

    # Compute class weights
    class_weights = compute_class_weights(
        train_data,
        num_classes=config.model.num_classes,
    )
    logger.info(f"Class weights: {class_weights.tolist()}")

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        config=config.data,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    # Create model
    logger.info(f"Creating model: {config.model.architecture}")
    model = create_model(
        architecture=config.model.architecture,
        config=config.model,
        compile_model=args.compile,
    )
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Setup callbacks
    callbacks = [
        ProgressBar(show_batch_progress=True),
        EarlyStopping(
            monitor="val_loss",
            patience=config.training.patience,
            restore_best=True,
        ),
        ModelCheckpoint(
            save_dir=output_dir / "checkpoints",
            monitor="val_loss",
            save_best_only=True,
            save_last=True,
        ),
        LRMonitor(log_every_n_steps=100),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        callbacks=callbacks,
        class_weights=class_weights,
    )

    # Train
    logger.info("Starting training...")
    results = trainer.fit(resume_from=args.resume)

    # Log final results
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {results.get('best_val_loss', 'N/A')}")

    # Save final config
    config.to_yaml(output_dir / "config.yaml")
    logger.info(f"Config saved to {output_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
