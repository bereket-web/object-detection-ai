from pathlib import Path

import torch
from PIL import Image

from boardwatch.dataset import BoardDataset, parse_yolo_label


def test_parse_yolo_box_converts_normalized_xywh_to_pixels():
    result = parse_yolo_label("2 0.5 0.5 0.25 0.50", image_width=200, image_height=100)
    assert result == [(3, 75.0, 25.0, 125.0, 75.0)]


def test_parse_yolo_label_rejects_invalid_class_and_coordinates():
    for bad in ("9 0.5 0.5 0.2 0.2", "1 1.2 0.5 0.2 0.2", "1 0.5 0.5 -0.2 0.2", "1 0.5 0.5 0.2"):
        try:
            parse_yolo_label(bad, image_width=100, image_height=80)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid label to be rejected: {bad}")


def test_dataset_reads_image_and_returns_detector_target(tmp_path: Path):
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    Image.new("RGB", (80, 60), color=(100, 120, 140)).save(split / "images" / "sample.jpg")
    (split / "labels" / "sample.txt").write_text("0 0.5 0.5 0.4 0.4\n1 0.25 0.25 0.2 0.2\n", encoding="utf-8")

    dataset = BoardDataset(split, augment=False)
    image, target = dataset[0]
    assert image.shape == (3, 60, 80)
    assert image.dtype == torch.float32
    assert target["boxes"].shape == (2, 4)
    assert target["labels"].tolist() == [1, 2]
    assert target["boxes"].min() >= 0
    assert target["boxes"][:, [0, 2]].max() <= 80
    assert target["boxes"][:, [1, 3]].max() <= 60


def test_augmented_dataset_preserves_nine_in_bounds_boxes(tmp_path: Path):
    split = tmp_path / "train"
    (split / "images").mkdir(parents=True)
    (split / "labels").mkdir()
    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(split / "images" / "board.jpg")
    rows = [
        f"{index % 3} {0.2 + 0.2 * (index % 3)} {0.2 + 0.2 * (index // 3)} 0.12 0.12" for index in range(9)
    ]
    (split / "labels" / "board.txt").write_text("\n".join(rows), encoding="utf-8")

    dataset = BoardDataset(split, augment=True)
    for seed in range(8):
        torch.manual_seed(seed)
        image, target = dataset[0]
        assert image.shape == (3, 100, 100)
        assert target["boxes"].shape == (9, 4)
        assert torch.all(target["boxes"][:, 0] < target["boxes"][:, 2])
        assert torch.all(target["boxes"][:, 1] < target["boxes"][:, 3])
        assert target["boxes"].min() >= 0
        assert target["boxes"][:, [0, 2]].max() <= 100
        assert target["boxes"][:, [1, 3]].max() <= 100
