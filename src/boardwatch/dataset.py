from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


def parse_yolo_label(
    text: str,
    image_width: int,
    image_height: int,
    label_offset: int = 1,
) -> list[tuple[int, float, float, float, float]]:
    """Parse normalized `class cx cy width height` rows into pixel xyxy boxes."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    rows: list[tuple[int, float, float, float, float]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Line {line_number}: expected 5 YOLO fields, got {len(fields)}")
        try:
            class_id = int(fields[0])
            cx, cy, box_width, box_height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid numeric YOLO value") from exc
        if class_id not in (0, 1, 2):
            raise ValueError(f"Line {line_number}: class id must be 0, 1, or 2")
        if not all(0.0 <= value <= 1.0 for value in (cx, cy, box_width, box_height)):
            raise ValueError(f"Line {line_number}: normalized coordinates must be in [0, 1]")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"Line {line_number}: box width and height must be positive")
        x1 = (cx - box_width / 2.0) * image_width
        y1 = (cy - box_height / 2.0) * image_height
        x2 = (cx + box_width / 2.0) * image_width
        y2 = (cy + box_height / 2.0) * image_height
        if x1 < -1e-3 or y1 < -1e-3 or x2 > image_width + 1e-3 or y2 > image_height + 1e-3:
            raise ValueError(f"Line {line_number}: box extends outside the image")
        rows.append(
            (
                class_id + label_offset,
                max(0.0, x1),
                max(0.0, y1),
                min(float(image_width), x2),
                min(float(image_height), y2),
            )
        )
    return rows


class BoardDataset(Dataset[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    """Local image/YOLO dataset, returning images and detector targets."""

    def __init__(self, split_dir: str | Path, augment: bool = False) -> None:
        self.split_dir = Path(split_dir)
        self.image_dir = self.split_dir / "images"
        self.label_dir = self.split_dir / "labels"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Expected images/ and labels/ under {self.split_dir}")
        self.images = sorted(
            path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.images:
            raise ValueError(f"No images found under {self.image_dir}")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.images[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing annotation for {image_path.name}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        records = parse_yolo_label(label_path.read_text(encoding="utf-8"), width, height)
        boxes = torch.tensor([row[1:] for row in records], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([row[0] for row in records], dtype=torch.int64)
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)

        if self.augment:
            if random.random() < 0.5:
                tensor = tensor.flip(-1)
                old_x1 = boxes[:, 0].clone()
                old_x2 = boxes[:, 2].clone()
                boxes[:, 0] = width - old_x2
                boxes[:, 2] = width - old_x1
            if random.random() < 0.5:
                tensor = tensor.flip(-2)
                old_y1 = boxes[:, 1].clone()
                old_y2 = boxes[:, 3].clone()
                boxes[:, 1] = height - old_y2
                boxes[:, 3] = height - old_y1
            brightness = random.uniform(0.82, 1.18)
            contrast = random.uniform(0.82, 1.18)
            channel_mean = tensor.mean(dim=(-2, -1), keepdim=True)
            tensor = ((tensor - channel_mean) * contrast + channel_mean) * brightness
            tensor = tensor.clamp_(0.0, 1.0)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return tensor, target


def detection_collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return tuple(zip(*batch, strict=False))
