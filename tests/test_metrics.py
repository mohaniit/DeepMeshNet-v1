import numpy as np
import pytest

from src.learning.metrics import (
    ClassificationMetrics,
    accuracy,
    compute_classification_metrics,
    confusion_matrix_as_list,
    macro_f1,
    weighted_f1,
)


def test_classification_metrics_to_dict():
    metrics = ClassificationMetrics(
        accuracy=1.0,
        macro_f1=1.0,
        weighted_f1=1.0,
        confusion_matrix=[[2, 0], [0, 2]],
        report={"accuracy": 1.0},
    )

    assert metrics.to_dict() == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
        "confusion_matrix": [[2, 0], [0, 2]],
        "report": {"accuracy": 1.0},
    }


def test_compute_classification_metrics_perfect_prediction():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=["LOW", "MEDIUM", "HIGH", "VERY_HIGH"],
    )

    assert metrics.accuracy == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.weighted_f1 == 1.0
    assert metrics.confusion_matrix == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]


def test_compute_classification_metrics_partial_prediction():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    assert metrics.accuracy == 0.75
    assert metrics.confusion_matrix == [
        [1, 1],
        [0, 2],
    ]
    assert 0.0 <= metrics.macro_f1 <= 1.0
    assert 0.0 <= metrics.weighted_f1 <= 1.0


def test_accuracy_function():
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)


def test_macro_f1_function():
    value = macro_f1([0, 1, 1], [0, 1, 0])

    assert 0.0 <= value <= 1.0


def test_weighted_f1_function():
    value = weighted_f1([0, 1, 1], [0, 1, 0])

    assert 0.0 <= value <= 1.0


def test_confusion_matrix_as_list():
    matrix = confusion_matrix_as_list(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        labels=[0, 1],
    )

    assert matrix == [
        [1, 1],
        [0, 2],
    ]


def test_metrics_accept_numpy_arrays():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 2, 2])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics.accuracy == pytest.approx(2 / 3)


def test_metrics_reject_none_y_true():
    with pytest.raises(ValueError, match="y_true cannot be None"):
        compute_classification_metrics(None, [0, 1])


def test_metrics_reject_none_y_pred():
    with pytest.raises(ValueError, match="y_pred cannot be None"):
        compute_classification_metrics([0, 1], None)


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics([0, 1], [0])


def test_metrics_reject_empty_inputs():
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_classification_metrics([], [])


def test_metrics_reject_multidimensional_y_true():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_classification_metrics([[0, 1]], [0, 1])


def test_metrics_reject_multidimensional_y_pred():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_classification_metrics([0, 1], [[0, 1]])


def test_confusion_matrix_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix_as_list([0, 1], [0])


def test_confusion_matrix_reject_empty_inputs():
    with pytest.raises(ValueError, match="cannot be empty"):
        confusion_matrix_as_list([], [])