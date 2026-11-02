from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from boardwatch.constants import CHECKPOINT_PATH, NUM_CLASSES


def build_detector(pretrained: bool = True) -> torch.nn.Module:
    """Build the lightweight detector, optionally initialized with TorchVision COCO_V1 weights."""
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights,
        weights_backbone=None,
        box_detections_per_img=30,
        rpn_pre_nms_top_n_train=300,
        rpn_post_nms_top_n_train=100,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def load_checkpoint(
    path: str | Path = CHECKPOINT_PATH, device: str | torch.device = "cpu"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. Run `boardwatch prepare`, then `boardwatch train`."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = build_detector(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    model.eval()
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    return model, metadata
