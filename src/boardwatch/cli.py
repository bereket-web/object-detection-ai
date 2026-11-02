from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from PIL import Image

from boardwatch.constants import CHECKPOINT_PATH, DEFAULT_SCORE_THRESHOLD
from boardwatch.evaluate import evaluate_checkpoint, select_score_threshold
from boardwatch.inference import BoardDetector
from boardwatch.prepare import prepare_dataset
from boardwatch.visualization import draw_detections


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="boardwatch", description="Local object detection for tic-tac-toe boards"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download, validate, and split the pinned CC BY dataset")
    prepare.add_argument("--force-download", action="store_true")
    prepare.add_argument("--seed", type=int, default=2026)

    train = subparsers.add_parser("train", help="Fine-tune the CPU MobileNetV3 Faster R-CNN detector")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=0.003)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--freeze-backbone-epochs", type=int, default=1)

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate the saved model on validation or held-out test data"
    )
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    evaluate.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)

    threshold = subparsers.add_parser(
        "select-threshold", help="Choose a confidence threshold using validation data only"
    )
    threshold.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)

    predict = subparsers.add_parser("predict", help="Run a single image and save the bounding-box overlay")
    predict.add_argument("image", type=Path)
    predict.add_argument("--output", type=Path, default=Path("prediction.jpg"))
    predict.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    predict.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)

    serve = subparsers.add_parser("serve", help="Start the local browser demo and JSON API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_dataset(force_download=args.force_download, seed=args.seed), indent=2))
    elif args.command == "train":
        from boardwatch.train import train_detector

        train_detector(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            freeze_backbone_epochs=args.freeze_backbone_epochs,
        )
    elif args.command == "evaluate":
        evaluate_checkpoint(
            split=args.split, score_threshold=args.score_threshold, checkpoint=args.checkpoint
        )
    elif args.command == "select-threshold":
        select_score_threshold(checkpoint=args.checkpoint)
    elif args.command == "predict":
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        detector = BoardDetector(args.checkpoint)
        detections = detector.predict(image, score_threshold=args.score_threshold)
        draw_detections(image, detections).save(args.output)
        print(
            json.dumps(
                {"image": str(args.image), "output": str(args.output), "detections": detections}, indent=2
            )
        )
    elif args.command == "serve":
        uvicorn.run("boardwatch.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
