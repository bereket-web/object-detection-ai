# Data and model provenance

## Training and evaluation images

This project uses the real-image **Tic-tac-toe Board Dataset** published by `zoncoen` on Hugging Face: <https://huggingface.co/datasets/zoncoen/tic-tac-toe-board-dataset>. The source dataset card declares the 59 photos and YOLO annotations as **Creative Commons Attribution 4.0 International (CC BY 4.0)**. This project records the exact source revision `8d3782b5a9992abca0e5611e9473eac7a1f98822`.

Attribution for local use: `zoncoen, Tic-tac-toe Board Dataset, CC BY 4.0, Hugging Face dataset revision 8d3782b5a9992abca0e5611e9473eac7a1f98822`.

The preparation script downloads files anonymously from the pinned revision, validates every image/label pair, records SHA-256 hashes, and forms transitive perceptual-similarity groups with a 63-bit grayscale DCT hash and Hamming threshold 4. The smallest multi-image group (20 photos) is held out for test; the remaining 39 photos are shuffled with seed 2026 into train and validation. This is a visual-similarity safeguard, not verified capture-session metadata: the source provides no session IDs, and validation shares the larger visual group with training.

Downloaded photographs and all image-derived splits stay in ignored local directories. The public repository does not contain source photos, prediction overlays, or other pixel-derived examples. It contains code, attribution/provenance, image hashes and split assignments, aggregate metrics, and non-photographic diagrams/charts. Do not redistribute or republish source or derived images without satisfying the source license and carrying the required attribution.

The dataset card links upstream project code at <https://github.com/guren-kaina/AMD_Robotics_Hackathon_2025_ProjectTemplate/tree/main/mission2/code/tic_tac_toe_overlay>. This project uses images and annotation files from the pinned dataset revision, not that repository's source code.

## Pretrained model

Training starts from TorchVision's `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1` checkpoint, then replaces the classification head and fine-tunes on the project training split. See the [official TorchVision model documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html). The source checkpoint is downloaded by TorchVision on first training run. This repository does not redistribute the upstream checkpoint or the fine-tuned weights; their distribution terms are separate from this repository's MIT source-code license and the dataset's CC BY 4.0 declaration. Review the upstream terms before any weight redistribution.

## References

- [Dataset card](https://huggingface.co/datasets/zoncoen/tic-tac-toe-board-dataset)
- [Pinned dataset revision](https://huggingface.co/datasets/zoncoen/tic-tac-toe-board-dataset/tree/8d3782b5a9992abca0e5611e9473eac7a1f98822)
- [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/)
- [TorchVision Faster R-CNN MobileNetV3 320 FPN documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)
