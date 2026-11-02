# BoardWatch detector — model card

## Model

A fine-tuned TorchVision Faster R-CNN detector with a MobileNetV3-Large 320 FPN backbone. Its classification head predicts background plus three board-cell states: `empty_cell`, `white_circle_cell`, and `black_cross_cell`. It detects each cell box and assigns its occupancy label.

## Intended use

Local educational demonstration and experimentation on top-down images of 3×3 tic-tac-toe boards with visual styles similar to the training data. CPU inference is available through the CLI, FastAPI JSON endpoint, browser image upload, and optional low-rate webcam polling. This is not a general-purpose object detector and is not validated for consequential use.

## Training data and method

The model was fine-tuned on 30 images from the 59-image CC BY 4.0 dataset published by `zoncoen`, pinned at `8d3782b5a9992abca0e5611e9473eac7a1f98822`. Preparation groups near-similar images with a 63-bit grayscale DCT perceptual hash (Hamming distance ≤4). One entire 20-image group is held out for test; the other 39 images are split into 30 training and 9 validation images. Validation shares the 37-image visual group with training and is used for checkpoint and threshold selection. The source supplies no capture-session metadata, so the pHash groups are only a proxy for session independence. Full attribution and release boundaries are in `../DATA_SOURCES.md`.

TorchVision `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1` initializes the detector. The classification/regression head is replaced for the three project labels. Training uses SGD with momentum and weight decay, one frozen-backbone warm-up epoch, then an unfrozen backbone; seed 2026; batch size 2; learning rate 0.003; and train-only flips plus brightness/contrast jitter. Ten CPU epochs completed in 101.7 seconds. Epoch 10 was selected using validation mAP@0.50 = 0.9504. Full configuration and per-epoch history are in `../reports/training_summary.json` and `../reports/training_history.csv`.

## Evaluation results

The confidence threshold 0.55 was selected from a fixed sweep on validation only by maximum micro-F1 at IoU ≥0.50 (validation precision 0.846, recall 0.951, F1 0.895). On the group-held-out test set of 20 images and 180 cell boxes, mAP@0.50 = **0.7986**; micro-precision = **0.7024**, recall = **0.8000**, and F1 = **0.7481** (144 TP, 61 FP, 36 FN). Per-class AP@0.50 is 0.8227 for empty cells, 0.8671 for circles, and 0.7061 for crosses.

AP uses 101-point interpolation at IoU 0.50 and detections with score ≥0.001; this is not COCO mAP@[.50:.95]. A similarity audit found no test-to-training/validation pair within the configured pHash threshold, but true session independence is not known. The corrected score is lower than the former random-split estimate; see the [evaluation protocol](../docs/evaluation.md).

Warm CPU latency on one 640×480 test image was 77.3 ms median (67.8–83.9 ms) after two warm-ups and eight timed runs in the current environment. This is a spot measurement, not a guaranteed device benchmark.

## Limitations

The dataset is small, specialized, and top-down; only two multi-image visual groups were found, and there is no capture-session metadata. Validation is not capture-group-independent from training, while the test estimate comes from a single group. Angled views, varied boards, other token styles, and unrelated objects are out of scope. Cross-class AP is lower than the other two classes. Confidence scores are not calibrated probabilities. No demographic, safety, fairness, or external-domain evaluation is claimed. The optional browser-camera mode is rate-limited and was not tested against physical camera hardware.

## Distribution

Fine-tuned weights and upstream COCO_V1 weights are not distributed. Source photos, derived splits, and prediction overlays are not committed. TorchVision's checkpoint and the CC BY 4.0 source data have terms separate from this repository's MIT code license; review upstream terms before any weight or image redistribution. No Hugging Face publication is configured and no project-scoped Hugging Face token was used.
