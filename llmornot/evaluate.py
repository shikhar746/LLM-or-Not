"""Metrics, bootstrap confidence intervals and latency measurement.

Labels are integer class ids (0..76, the order of categories.json) everywhere in this module.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of items whose predicted class equals the true class."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Unweighted mean of per-class F1.

    Per class c:  precision = TP / predicted-as-c,  recall = TP / truly-c,
                  F1 = 2*TP / (predicted-as-c + truly-c)   (same thing, no division by zero issues).
    Averaged over every class that appears in y_true or y_pred (scikit-learn's default), so a
    class the model predicts but never sees correctly counts as F1 = 0 and drags the mean down.

    A prediction of -1 means "invalid output" (an LLM answer that is not one of the labels). It is
    always wrong: it lowers recall of the true class but is not a class of its own.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    valid = y_pred >= 0
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    tp = np.bincount(y_true[y_true == y_pred], minlength=n_classes)
    n_true = np.bincount(y_true, minlength=n_classes)
    n_pred = np.bincount(y_pred[valid], minlength=n_classes)
    present = (n_true + n_pred) > 0
    f1 = 2 * tp[present] / (n_true[present] + n_pred[present])
    return float(f1.mean())


def bootstrap_indices(n_items: int, n_boot: int = 2000, seed: int = 42) -> np.ndarray:
    """Resampled row indices, shape (n_boot, n_items).

    Generate these ONCE per test set and reuse them for every model ("paired bootstrap"): each
    model is then scored on exactly the same resampled test sets, so differences between models
    are not blurred by different random draws.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_items, size=(n_boot, n_items))


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: Metric,
    indices: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile (1 - alpha) confidence interval.

    Idea: pretend the test set is the population, draw many same-size samples from it with
    replacement, and see how much the metric moves. The middle 95% of those values is the CI.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = np.array([metric(y_true[idx], y_pred[idx]) for idx in indices])
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return metric(y_true, y_pred), float(lo), float(hi)


def latency_ms(predict_one: Callable[[str], object], texts: Sequence[str], warmup: int = 10) -> np.ndarray:
    """Wall-clock milliseconds per query, calling the model one message at a time.

    One-at-a-time mirrors live routing, where each customer message arrives on its own.
    The first `warmup` calls are run but not recorded (they include one-off costs such as
    loading weights into memory or JIT compilation).
    """
    for text in texts[:warmup]:
        predict_one(text)
    times = []
    for text in texts:
        start = time.perf_counter()
        predict_one(text)
        times.append((time.perf_counter() - start) * 1000)
    return np.array(times)


def percentiles(times_ms: np.ndarray) -> dict[str, float]:
    """p50 = typical query; p95 = the slow tail that 1 in 20 customers sees."""
    p50, p95 = np.percentile(times_ms, [50, 95])
    return {"p50_ms": float(p50), "p95_ms": float(p95)}
