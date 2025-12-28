#!/usr/bin/env python
"""
Model export script for BeautyScorer.

Exports models to ONNX, TorchScript, or quantized formats.

Usage:
    python scripts/export_model.py --model outputs/best.pt --format onnx
    python scripts/export_model.py --model outputs/best.pt --format torchscript
    python scripts/export_model.py --model outputs/best.pt --format quantized
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beauty_scorer.inference.export import (
    export_to_onnx,
    export_to_quantized,
    export_to_torchscript,
    get_model_size,
)
from beauty_scorer.models.factory import load_model
from beauty_scorer.utils.logging import get_logger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Export BeautyScorer models for deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "torchscript", "quantized", "all"],
        default="onnx",
        help="Export format",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Output directory",
    )
    parser.add_argument(
        "--max-photos",
        type=int,
        default=9,
        help="Maximum number of photos",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Image size (height width)",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Face size (height width)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version",
    )

    return parser.parse_args()


def main():
    """Main export function."""
    args = parse_args()

    setup_logging()
    logger = get_logger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model)

    # Get model size info
    size_info = get_model_size(model)
    logger.info(f"Model size: {size_info['saved_size_mb']:.2f} MB")
    logger.info(f"Parameters: {size_info['total_params']:,}")

    image_size = tuple(args.image_size)
    face_size = tuple(args.face_size)

    # Export based on format
    formats_to_export = []
    if args.format == "all":
        formats_to_export = ["onnx", "torchscript", "quantized"]
    else:
        formats_to_export = [args.format]

    for fmt in formats_to_export:
        logger.info(f"Exporting to {fmt}...")

        if fmt == "onnx":
            output_path = output_dir / "model.onnx"
            export_to_onnx(
                model,
                output_path,
                max_photos=args.max_photos,
                image_size=image_size,
                face_size=face_size,
                opset_version=args.opset,
            )
            logger.info(f"ONNX model exported to {output_path}")

        elif fmt == "torchscript":
            output_path = output_dir / "model.pt"
            export_to_torchscript(
                model,
                output_path,
                max_photos=args.max_photos,
                image_size=image_size,
                face_size=face_size,
            )
            logger.info(f"TorchScript model exported to {output_path}")

        elif fmt == "quantized":
            output_path = output_dir / "model_quantized.pt"
            export_to_quantized(
                model,
                output_path,
            )
            logger.info(f"Quantized model exported to {output_path}")

    logger.info("Export complete!")

    # Print summary
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    print(f"Source model: {args.model}")
    print(f"Output directory: {output_dir}")
    print(f"Formats exported: {', '.join(formats_to_export)}")
    print(f"Model size: {size_info['saved_size_mb']:.2f} MB")
    print("Input shapes:")
    print(f"  - Photos: (batch, {args.max_photos}, 3, {image_size[0]}, {image_size[1]})")
    print(f"  - Faces: (batch, {args.max_photos}, 3, {face_size[0]}, {face_size[1]})")
    print("=" * 50)


if __name__ == "__main__":
    main()
