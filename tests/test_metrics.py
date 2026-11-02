import torch

from boardwatch.metrics import box_iou, evaluate_detector, interpolated_ap


def test_box_iou_known_overlap():
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == 1 / 3
    assert box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0


def test_interpolated_ap_perfect_and_no_predictions():
    assert interpolated_ap([1.0], [1.0]) == 1.0
    assert interpolated_ap([], []) == 0.0


class PerfectDetector:
    def eval(self):
        return self

    def __call__(self, images):
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "labels": torch.tensor([1, 2]),
                "scores": torch.tensor([0.99, 0.95]),
            }
            for _ in images
        ]


def test_evaluation_uses_class_aware_exact_box_matching():
    dataset = [
        (
            torch.zeros((3, 32, 32)),
            {
                "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
                "labels": torch.tensor([1, 2]),
            },
        )
    ]
    metrics = evaluate_detector(PerfectDetector(), dataset, score_threshold=0.25)
    assert metrics["map50"] == 1.0
    assert metrics["micro_precision"] == 1.0
    assert metrics["micro_recall"] == 1.0
    assert metrics["ground_truth_objects"] == 2
    assert metrics["classes"]["black_cross_cell"]["ground_truth"] == 0
