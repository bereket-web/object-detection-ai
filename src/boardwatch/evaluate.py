from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from boardwatch.constants import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DEFAULT_SCORE_THRESHOLD,
    PROCESSED_ROOT,
    REPORT_ROOT,
)
from boardwatch.dataset import BoardDataset
from boardwatch.inference import BoardDetector
from boardwatch.metrics import evaluate_detector
from boardwatch.visualization import draw_detections

THRESHOLDS_TO_TEST = (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.90)


def _best_threshold(sweep: list[dict[str, Any]]) -> dict[str, Any]:
    return max(sweep, key=lambda row: (row["micro_f1"], row["micro_precision"], row["threshold"]))


def evaluate_checkpoint(
    split: str = "test",
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    checkpoint: str | Path = CHECKPOINT_PATH,
    output_dir: str | Path = REPORT_ROOT,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError(
            "Evaluation split must be `validation` or `test`; training split is never scored here."
        )
    dataset = BoardDataset(PROCESSED_ROOT / split, augment=False)
    detector = BoardDetector(checkpoint)
    metrics = evaluate_detector(detector.model, dataset, score_threshold=score_threshold)
    metrics["split"] = split
    metrics["checkpoint_metadata"] = detector.metadata
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"evaluation_{split}.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _plot_metrics(metrics, out_dir / f"evaluation_{split}.png")
    _save_overlay(detector, dataset, out_dir / f"example_{split}_overlay.jpg", score_threshold)
    print(json.dumps(metrics, indent=2))
    return metrics


def select_score_threshold(
    checkpoint: str | Path = CHECKPOINT_PATH,
    output_path: str | Path = REPORT_ROOT / "threshold_selection.json",
    thresholds: tuple[float, ...] = THRESHOLDS_TO_TEST,
) -> dict[str, Any]:
    """Choose an operating score threshold by micro-F1 on validation only."""
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be a non-empty sequence of values in [0, 1]")
    dataset = BoardDataset(PROCESSED_ROOT / "validation", augment=False)
    detector = BoardDetector(checkpoint)
    sweep = []
    for threshold in thresholds:
        metrics = evaluate_detector(detector.model, dataset, score_threshold=threshold)
        sweep.append(
            {
                "threshold": threshold,
                "micro_precision": metrics["micro_precision"],
                "micro_recall": metrics["micro_recall"],
                "micro_f1": metrics["micro_f1"],
                "tp": metrics["true_positives"],
                "fp": metrics["false_positives"],
                "fn": metrics["false_negatives"],
            }
        )
    selected = _best_threshold(sweep)
    result = {
        "selection_split": "validation",
        "selection_objective": "maximum micro-F1 at IoU >= 0.50; ties prefer precision, then higher threshold",
        "thresholds_tested": list(thresholds),
        "selected_threshold": selected["threshold"],
        "selection_metrics": {
            "micro_precision": selected["micro_precision"],
            "micro_recall": selected["micro_recall"],
            "micro_f1": selected["micro_f1"],
            "true_positives": selected["tp"],
            "false_positives": selected["fp"],
            "false_negatives": selected["fn"],
        },
        "sweep": sweep,
        "validation_is_capture_group_independent_from_train": False,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _plot_metrics(metrics: dict[str, Any], destination: Path) -> None:
    names = list(CLASS_NAMES)
    ap = [metrics["classes"][name]["ap50"] for name in names]
    recall = [metrics["classes"][name]["recall"] for name in names]
    precision = [metrics["classes"][name]["precision"] for name in names]
    colors = ["#18a999", "#2463eb", "#e34b4b"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    axes[0].bar(names, ap, color=colors)
    axes[0].set_title("Per-class AP@0.50 IoU")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Average precision")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(names, precision, color=colors, alpha=0.9, label="Precision")
    axes[1].bar(names, recall, color=colors, alpha=0.38, label="Recall")
    axes[1].set_title(f"At score ≥ {metrics['score_threshold']:.2f}")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].legend(frameon=False)
    figure.suptitle(
        f"Held-out {metrics['split']} set · mAP@0.50 = {metrics['map50']:.3f} · "
        f"micro-F1 = {metrics['micro_f1']:.3f}",
        fontsize=12,
    )
    figure.savefig(destination, dpi=170, facecolor="white")
    plt.close(figure)


def _save_overlay(
    detector: BoardDetector, dataset: BoardDataset, destination: Path, threshold: float
) -> None:
    image_path = dataset.images[0]
    with Image.open(image_path) as image:
        source = image.convert("RGB")
    detections = detector.predict(source, score_threshold=threshold)
    draw_detections(source, detections).save(destination, quality=92)
