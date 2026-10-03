import numpy as np
import pytest
from sklearn.metrics import f1_score

from llmornot.evaluate import accuracy, bootstrap_ci, bootstrap_indices, macro_f1, percentiles


def test_accuracy_hand_example():
    assert accuracy([0, 1, 2, 2], [0, 1, 1, 2]) == 0.75


def test_macro_f1_hand_example():
    # class 0: TP=1, true=1, pred=1 -> F1 = 1
    # class 1: TP=1, true=1, pred=2 -> F1 = 2*1/(1+2) = 2/3
    # class 2: TP=1, true=2, pred=1 -> F1 = 2*1/(2+1) = 2/3
    assert macro_f1([0, 1, 2, 2], [0, 1, 1, 2]) == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)


def test_macro_f1_counts_wrongly_predicted_class_as_zero():
    # class 3 never appears in y_true but is predicted once: its F1 is 0 and it is averaged in
    assert macro_f1([0, 0], [0, 3]) == pytest.approx((2 / 3 + 0) / 2)


def test_macro_f1_matches_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 77, 3000), rng.integers(0, 77, 3000)
    assert macro_f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred, average="macro"))


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 5, 500)
    y_pred = np.where(rng.random(500) < 0.8, y_true, (y_true + 1) % 5)  # ~80% correct
    idx = bootstrap_indices(500, n_boot=500, seed=7)
    point, lo, hi = bootstrap_ci(y_true, y_pred, accuracy, idx)
    assert lo < point < hi
    assert (lo, hi) == bootstrap_ci(y_true, y_pred, accuracy, bootstrap_indices(500, 500, seed=7))[1:]


def test_bootstrap_ci_narrows_with_more_data():
    def width(n):
        y_true = np.zeros(n, dtype=int)
        y_pred = (np.arange(n) % 5 == 0).astype(int)  # exactly 80% correct
        _, lo, hi = bootstrap_ci(y_true, y_pred, accuracy, bootstrap_indices(n, 500, seed=0))
        return hi - lo

    assert width(2000) < width(200)


def test_percentiles():
    times = np.arange(1, 101, dtype=float)  # 1..100 ms
    p = percentiles(times)
    assert p["p50_ms"] == pytest.approx(50.5)
    assert p["p95_ms"] == pytest.approx(95.05)
