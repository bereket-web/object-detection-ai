from __future__ import annotations

import csv
import json
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from boardwatch.constants import CHECKPOINT_PATH, PROCESSED_ROOT, PROJECT_ROOT, REPORT_ROOT
from boardwatch.dataset import BoardDataset, detection_collate
from boardwatch.metrics import evaluate_detector
from boardwatch.model import build_detector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_payload(
    model: torch.nn.Module, epoch: int, metrics: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return {
        "state_dict": state,
        "epoch": epoch,
        "validation_map50": metrics["map50"],
        "classes": ["empty_cell", "white_circle_cell", "black_cross_cell"],
        "architecture": "torchvision Faster R-CNN MobileNetV3-Large-320-FPN",
        "config": config,
    }


def train_detector(
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 0.003,
    seed: int = 2026,
    freeze_backbone_epochs: int = 1,
    max_train_images: int | None = None,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    train_set = BoardDataset(PROCESSED_ROOT / "train", augment=True)
    validation_set = BoardDataset(PROCESSED_ROOT / "validation", augment=False)
    if max_train_images is not None:
        if max_train_images < 1:
            raise ValueError("max_train_images must be positive")
        indices = list(range(min(max_train_images, len(train_set))))
        from torch.utils.data import Subset

        train_data = Subset(train_set, indices)
    else:
        train_data = train_set

    seed_everything(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=detection_collate,
        generator=loader_generator,
    )

    device = torch.device("cpu")
    model = build_detector(pretrained=True).to(device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(2, epochs // 2), gamma=0.2)
    config = {
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "freeze_backbone_epochs": freeze_backbone_epochs,
        "train_images": len(train_data),
        "validation_images": len(validation_set),
        "device": str(device),
        "pretrained_weights": "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
    }
    history: list[dict[str, Any]] = []
    best_map = -1.0
    best_epoch = 0
    started = time.perf_counter()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch_index in range(epochs):
        epoch_number = epoch_index + 1
        if epoch_index == freeze_backbone_epochs:
            for parameter in model.backbone.parameters():
                parameter.requires_grad = True
        model.train()
        loss_sum = 0.0
        batches = 0
        for images, targets in train_loader:
            moved_images = [image.to(device) for image in images]
            moved_targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)
            loss_parts = model(moved_images, moved_targets)
            loss = sum(loss_parts.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss in epoch {epoch_number}: {loss.item()}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_sum += float(loss.detach())
            batches += 1
        scheduler.step()
        validation_metrics = evaluate_detector(model, validation_set, device=device)
        row = {
            "epoch": epoch_number,
            "train_loss": loss_sum / max(1, batches),
            "validation_map50": validation_metrics["map50"],
            "validation_micro_f1": validation_metrics["micro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        history.append(row)
        print(
            f"epoch {epoch_number:02d}/{epochs}: loss={row['train_loss']:.4f} "
            f"val_mAP50={row['validation_map50']:.4f} val_F1={row['validation_micro_f1']:.4f}",
            flush=True,
        )
        if validation_metrics["map50"] > best_map:
            best_map = validation_metrics["map50"]
            best_epoch = epoch_number
            torch.save(_checkpoint_payload(model, epoch_number, validation_metrics, config), CHECKPOINT_PATH)

    history_path = REPORT_ROOT / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    result = {
        "checkpoint": CHECKPOINT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "best_epoch": best_epoch,
        "best_validation_map50": best_map,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "training_config": config,
        "history": history,
    }
    (REPORT_ROOT / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"best checkpoint saved: {CHECKPOINT_PATH} (epoch {best_epoch}, val mAP@0.50={best_map:.4f})",
        flush=True,
    )
    return result
