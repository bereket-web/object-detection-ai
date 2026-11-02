from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from boardwatch.constants import CHECKPOINT_PATH, CLASS_NAMES, DEFAULT_SCORE_THRESHOLD
from boardwatch.model import load_checkpoint


class BoardDetector:
    def __init__(self, checkpoint: str | Path = CHECKPOINT_PATH, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model, self.metadata = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    @torch.inference_mode()
    def predict(
        self, image: Image.Image, score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> list[dict[str, Any]]:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        image = ImageOps.exif_transpose(image).convert("RGB")
        tensor = pil_to_tensor(image).to(dtype=torch.float32).div_(255.0).to(self.device)
        output = self.model([tensor])[0]
        detections: list[dict[str, Any]] = []
        for box, label, score in zip(
            output["boxes"].cpu().tolist(),
            output["labels"].cpu().tolist(),
            output["scores"].cpu().tolist(),
            strict=True,
        ):
            score = float(score)
            label = int(label)
            if score < score_threshold or not 1 <= label <= len(CLASS_NAMES):
                continue
            detections.append(
                {
                    "label": CLASS_NAMES[label - 1],
                    "score": score,
                    "box": [float(value) for value in box],
                }
            )
        return detections
