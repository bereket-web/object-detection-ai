from __future__ import annotations

import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, UnidentifiedImageError

from boardwatch.constants import (
    CLASS_NAMES,
    DATASET_IMAGE_COUNT,
    DATASET_REPO,
    DATASET_REVISION,
    PROCESSED_ROOT,
    RAW_ROOT,
    REPORT_ROOT,
)
from boardwatch.dataset import parse_yolo_label

BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}"
SPLIT_SEED = 2026
PHASH_DISTANCE_THRESHOLD = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()
    content = response.content
    if not content:
        raise ValueError(f"Empty response for {url}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(content)
    temporary.replace(destination)


def _download_pair(index: int, force: bool) -> tuple[str, Path, Path]:
    stem = f"top-camera{index:02d}"
    image_path = RAW_ROOT / "images" / f"{stem}.jpg"
    label_path = RAW_ROOT / "labels" / f"{stem}.txt"
    if force or not image_path.exists():
        _download(f"{BASE_URL}/real/images/{image_path.name}?download=true", image_path)
    if force or not label_path.exists():
        _download(f"{BASE_URL}/real/labels/{label_path.name}?download=true", label_path)
    return stem, image_path, label_path


def _perceptual_hash(image_path: Path) -> int:
    """Return a 63-bit low-frequency DCT hash for near-duplicate screening."""
    size = 32
    coordinates = np.arange(size)
    frequencies = np.arange(size)[:, None]
    basis = np.cos(np.pi * (2 * coordinates + 1) * frequencies / (2 * size)) * np.sqrt(2 / size)
    basis[0, :] = np.sqrt(1 / size)
    with Image.open(image_path) as image:
        pixels = np.asarray(
            image.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float64
        )
    coefficients = (basis @ pixels @ basis.T)[:8, :8].ravel()[1:]
    median = float(np.median(coefficients))
    value = 0
    for bit in coefficients > median:
        value = (value << 1) | int(bit)
    return value


def _group_similar_hashes(
    hashes: dict[str, int], max_distance: int = PHASH_DISTANCE_THRESHOLD
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build deterministic connected components from perceptually similar images."""
    names = sorted(hashes)
    parent = {name: name for name in names}

    def find(name: str) -> str:
        if parent[name] != name:
            parent[name] = find(parent[name])
        return parent[name]

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            first, second = sorted((root_left, root_right))
            parent[second] = first

    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if (hashes[left] ^ hashes[right]).bit_count() <= max_distance:
                union(left, right)

    members_by_root: dict[str, list[str]] = {}
    for name in names:
        members_by_root.setdefault(find(name), []).append(name)
    groups: dict[str, list[str]] = {}
    group_by_name: dict[str, str] = {}
    for index, members in enumerate(sorted(members_by_root.values(), key=lambda group: group[0]), start=1):
        group_id = f"visual-{index:02d}"
        groups[group_id] = members
        group_by_name.update({name: group_id for name in members})
    return group_by_name, groups


def _assign_grouped_splits(
    hashes: dict[str, int], seed: int = SPLIT_SEED
) -> tuple[dict[str, str], dict[str, str], dict[str, list[str]], str]:
    """Hold out one complete multi-image similarity group, then seed-split the remainder."""
    group_by_name, groups = _group_similar_hashes(hashes)
    candidates = [group_id for group_id, members in groups.items() if len(members) > 1]
    if not candidates:
        raise ValueError("No multi-image visual group is available for a group-held-out test set")
    test_group = min(candidates, key=lambda group_id: (len(groups[group_id]), group_id))
    test_names = set(groups[test_group])
    train_validation = sorted(set(hashes) - test_names)
    validation_count = max(1, round(len(hashes) * 0.15))
    if len(train_validation) <= validation_count:
        raise ValueError("Not enough non-test images for non-empty training and validation splits")

    random.Random(seed).shuffle(train_validation)
    validation_names = set(train_validation[:validation_count])
    assignments = {
        name: "test" if name in test_names else "validation" if name in validation_names else "train"
        for name in sorted(hashes)
    }
    return assignments, group_by_name, groups, test_group


def validate_pair(
    image_path: Path, label_path: Path
) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            width, height = image.size
            if image.format not in {"JPEG", "PNG"}:
                raise ValueError(f"Unsupported image format {image.format}: {image_path}")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid image file: {image_path}") from exc
    records = parse_yolo_label(label_path.read_text(encoding="utf-8"), width, height, label_offset=0)
    if len(records) != 9:
        raise ValueError(f"Expected 9 cell boxes in {label_path.name}, found {len(records)}")
    return width, height, records


def prepare_dataset(force_download: bool = False, seed: int = SPLIT_SEED) -> dict[str, Any]:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[str, Path, Path]] = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs = [
            pool.submit(_download_pair, index, force_download) for index in range(1, DATASET_IMAGE_COUNT + 1)
        ]
        for job in as_completed(jobs):
            pairs.append(job.result())
    pairs.sort(key=lambda item: item[0])

    sample_records: list[dict[str, Any]] = []
    for stem, image_path, label_path in pairs:
        width, height, boxes = validate_pair(image_path, label_path)
        sample_records.append(
            {
                "stem": stem,
                "width": width,
                "height": height,
                "objects": boxes,
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "perceptual_hash": _perceptual_hash(image_path),
            }
        )

    perceptual_hashes = {record["stem"]: record["perceptual_hash"] for record in sample_records}
    assignments, group_by_stem, visual_groups, test_group = _assign_grouped_splits(
        perceptual_hashes, seed=seed
    )

    # Rebuild prepared folders so a changed seed cannot leave stale split members behind.
    for split in ("train", "validation", "test"):
        split_dir = PROCESSED_ROOT / split
        if split_dir.exists():
            for child in split_dir.iterdir():
                if child.is_dir():
                    for path in child.iterdir():
                        path.unlink()
                else:
                    child.unlink()
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    class_counts = {split: [0] * len(CLASS_NAMES) for split in ("train", "validation", "test")}
    split_counts = {split: 0 for split in class_counts}
    with (REPORT_ROOT / "split_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image",
                "split",
                "visual_group",
                "width",
                "height",
                "image_sha256",
                "label_sha256",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for record in sample_records:
            split = assignments[record["stem"]]
            split_counts[split] += 1
            for class_id, *_ in record["objects"]:
                class_counts[split][class_id] += 1
            stem = record["stem"]
            (PROCESSED_ROOT / split / "images" / f"{stem}.jpg").write_bytes(
                (RAW_ROOT / "images" / f"{stem}.jpg").read_bytes()
            )
            (PROCESSED_ROOT / split / "labels" / f"{stem}.txt").write_bytes(
                (RAW_ROOT / "labels" / f"{stem}.txt").read_bytes()
            )
            writer.writerow(
                {
                    "image": f"{stem}.jpg",
                    "split": split,
                    "visual_group": group_by_stem[stem],
                    "width": record["width"],
                    "height": record["height"],
                    "image_sha256": record["image_sha256"],
                    "label_sha256": record["label_sha256"],
                }
            )

    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "license_declared_by_source": "CC BY 4.0",
        "split_seed": seed,
        "split_strategy": (
            "63-bit low-frequency DCT perceptual-hash connected components (Hamming distance <= 4); "
            "smallest multi-image component held out as test; remaining images shuffled into train/validation"
        ),
        "perceptual_hash_distance_threshold": PHASH_DISTANCE_THRESHOLD,
        "visual_group_count": len(visual_groups),
        "visual_group_image_counts": {group_id: len(members) for group_id, members in visual_groups.items()},
        "held_out_test_visual_group": test_group,
        "validation_is_capture_group_independent_from_train": False,
        "images": len(sample_records),
        "objects": sum(sum(class_counts[split]) for split in class_counts),
        "classes": list(CLASS_NAMES),
        "split_image_counts": split_counts,
        "split_object_counts": {
            split: {name: class_counts[split][i] for i, name in enumerate(CLASS_NAMES)}
            for split in class_counts
        },
        "records": [
            {
                "image": f"{record['stem']}.jpg",
                "split": assignments[record["stem"]],
                "visual_group": group_by_stem[record["stem"]],
                "width": record["width"],
                "height": record["height"],
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
            }
            for record in sample_records
        ],
    }
    (PROCESSED_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in manifest.items() if key != "records"}
    (REPORT_ROOT / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    test_to_other_pairs_within_threshold = sum(
        1
        for left_index, left in enumerate(sorted(perceptual_hashes))
        for right in sorted(perceptual_hashes)[left_index + 1 :]
        if (assignments[left] == "test") != (assignments[right] == "test")
        and (perceptual_hashes[left] ^ perceptual_hashes[right]).bit_count() <= PHASH_DISTANCE_THRESHOLD
    )
    if test_to_other_pairs_within_threshold:
        raise RuntimeError("The group-held-out test split contains a near-duplicate cross-split pair")
    similarity_audit = {
        "hash": "grayscale 32x32 DCT, low-frequency 8x8 coefficients excluding DC (63 bits)",
        "hamming_distance_threshold": PHASH_DISTANCE_THRESHOLD,
        "transitive_connected_components": {
            group_id: {
                "image_count": len(members),
                "split_counts": {
                    split: sum(assignments[name] == split for name in members)
                    for split in ("train", "validation", "test")
                },
            }
            for group_id, members in visual_groups.items()
        },
        "held_out_test_visual_group": test_group,
        "test_to_train_or_validation_pairs_within_threshold": test_to_other_pairs_within_threshold,
        "validation_shares_a_visual_group_with_training": any(
            group_by_stem[name] == group_by_stem[other]
            for name, split in assignments.items()
            if split == "validation"
            for other, other_split in assignments.items()
            if other_split == "train"
        ),
    }
    (REPORT_ROOT / "similarity_audit.json").write_text(
        json.dumps(similarity_audit, indent=2) + "\n", encoding="utf-8"
    )
    return summary
