#!/usr/bin/env python
"""
Inference script for BeautyScorer models.

Usage:
    python scripts/infer.py --model outputs/best.pt --photos photo1.jpg photo2.jpg
    python scripts/infer.py --model outputs/best.pt --folder /path/to/person/photos
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from beauty_scorer.inference.predictor import BeautyPredictor
from beauty_scorer.utils.logging import get_logger, setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with BeautyScorer models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--photos",
        type=str,
        nargs="+",
        help="List of photo paths",
    )
    input_group.add_argument(
        "--folder",
        type=str,
        help="Folder containing photos",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, cpu, mps)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output",
    )

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    setup_logging()
    logger = get_logger(__name__)

    # Get photo paths
    if args.photos:
        photo_paths = args.photos
    else:
        folder = Path(args.folder)
        photo_paths = (
            list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        )
        photo_paths = [str(p) for p in photo_paths]

    if not photo_paths:
        logger.error("No photos found!")
        sys.exit(1)

    logger.info(f"Found {len(photo_paths)} photos")

    # Create predictor
    logger.info(f"Loading model from {args.model}")
    predictor = BeautyPredictor(args.model, device=args.device)

    # Run inference
    logger.info("Running inference...")
    result = predictor.predict_person(photo_paths)

    # Display results
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    print(f"Predicted Score: {result['score']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print()

    # Top predictions
    print(f"Top {args.top_k} predictions:")
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for _, (score, prob) in enumerate(sorted_probs[: args.top_k]):
        bar = "#" * int(prob * 30)
        print(f"  Score {score}: {prob * 100:5.1f}% {bar}")

    if args.verbose:
        print()
        print("All probabilities:")
        for score in range(1, len(result["probabilities"]) + 1):
            prob = result["probabilities"][score]
            bar = "#" * int(prob * 30)
            marker = " <--" if score == result["score"] else ""
            print(f"  Score {score}: {prob * 100:5.1f}% {bar}{marker}")

        # Score range confidence
        score = result["score"]
        range_conf = sum(
            result["probabilities"].get(s, 0) for s in range(max(1, score - 1), min(10, score + 2))
        )
        print()
        print(
            f"Confidence in score {max(1, score - 1)}-{min(9, score + 1)}: {range_conf * 100:.1f}%"
        )

    print("=" * 50)

    # Save results
    if args.output:
        output = {
            "photos": photo_paths,
            "score": result["score"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"],
            "model": args.model,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
