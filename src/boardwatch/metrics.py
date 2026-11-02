from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from boardwatch.constants import CLASS_NAMES


def box_iou(box_a: list[float] | torch.Tensor, box_b: list[float] | torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def interpolated_ap(recall: list[float], precision: list[float]) -> float:
    """101-point interpolated AP, with recall already ordered by descending score."""
    if not recall:
        return 0.0
    values: list[float] = []
    for threshold_index in range(101):
        threshold = threshold_index / 100.0
        values.append(max((p for r, p in zip(recall, precision, strict=True) if r >= threshold), default=0.0))
    return sum(values) / 101.0


def _match_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
) -> tuple[list[bool], dict[int, set[int]]]:
    matched: dict[int, set[int]] = defaultdict(set)
    true_positives: list[bool] = []
    for prediction in predictions:
        candidates = ground_truth.get(prediction["image_id"], [])
        best_index = -1
        best_iou = 0.0
        for index, target in enumerate(candidates):
            if index in matched[prediction["image_id"]] or target["label"] != prediction["label"]:
                continue
            overlap = box_iou(prediction["box"], target["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_index = index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        true_positives.append(is_match)
        if is_match:
            matched[prediction["image_id"]].add(best_index)
    return true_positives, matched


def evaluate_detector(
    model: torch.nn.Module,
    dataset,
    device: str | torch.device = "cpu",
    score_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate a small detector on a dataset; AP uses scores >= .001 before operating-point metrics."""
    model.eval()
    device = torch.device(device)
    ground_truth: dict[int, list[dict[str, Any]]] = {}
    predictions_by_class: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, len(CLASS_NAMES) + 1)}

    with torch.inference_mode():
        for image_id, (image, target) in enumerate(dataset):
            output = model([image.to(device)])[0]
            boxes = target["boxes"].tolist()
            labels = target["labels"].tolist()
            ground_truth[image_id] = [
                {"box": box, "label": int(label)} for box, label in zip(boxes, labels, strict=True)
            ]
            for box, label, score in zip(
                output["boxes"].cpu().tolist(),
                output["labels"].cpu().tolist(),
                output["scores"].cpu().tolist(),
                strict=True,
            ):
                if int(label) in predictions_by_class and float(score) >= 0.001:
                    predictions_by_class[int(label)].append(
                        {"image_id": image_id, "box": box, "label": int(label), "score": float(score)}
                    )

    class_results: dict[str, dict[str, float | int]] = {}
    supported_aps: list[float] = []
    total_tp = total_fp = total_fn = total_gt = 0
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        class_gt = {
            image_id: [row for row in rows if row["label"] == class_id]
            for image_id, rows in ground_truth.items()
        }
        positives = sum(len(rows) for rows in class_gt.values())
        candidates = sorted(predictions_by_class[class_id], key=lambda row: row["score"], reverse=True)
        true_positive_flags, _ = _match_predictions(candidates, class_gt, iou_threshold)
        running_tp = 0
        running_fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for is_tp in true_positive_flags:
            running_tp += int(is_tp)
            running_fp += int(not is_tp)
            recalls.append(running_tp / positives if positives else 0.0)
            precisions.append(running_tp / (running_tp + running_fp) if running_tp + running_fp else 0.0)
        ap = interpolated_ap(recalls, precisions) if positives else 0.0

        operating_predictions = [row for row in candidates if row["score"] >= score_threshold]
        operating_flags, operating_matched = _match_predictions(
            operating_predictions, class_gt, iou_threshold
        )
        tp = sum(operating_flags)
        fp = len(operating_flags) - tp
        fn = positives - sum(len(indices) for indices in operating_matched.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / positives if positives else 0.0
        class_results[class_name] = {
            "ground_truth": positives,
            "predictions_at_threshold": len(operating_predictions),
            "ap50": ap,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }
        if positives:
            supported_aps.append(ap)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += positives

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    return {
        "images": len(dataset),
        "ground_truth_objects": total_gt,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "map50": sum(supported_aps) / len(supported_aps) if supported_aps else 0.0,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "classes": class_results,
    }
