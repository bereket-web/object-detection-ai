from boardwatch.evaluate import _best_threshold


def test_best_threshold_maximizes_micro_f1_then_precision_then_threshold():
    sweep = [
        {"threshold": 0.25, "micro_f1": 0.8, "micro_precision": 0.6},
        {"threshold": 0.50, "micro_f1": 0.8, "micro_precision": 0.75},
        {"threshold": 0.75, "micro_f1": 0.8, "micro_precision": 0.75},
        {"threshold": 0.90, "micro_f1": 0.79, "micro_precision": 0.99},
    ]

    assert _best_threshold(sweep)["threshold"] == 0.75
