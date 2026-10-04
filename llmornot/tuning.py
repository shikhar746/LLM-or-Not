"""Pick hyperparameters on the validation split, never on test."""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from llmornot.evaluate import accuracy, macro_f1
from llmornot.models.base import IntentClassifier


def grid_search(
    make_model: Callable[..., IntentClassifier],
    grid: dict[str, list],
    train_texts: Sequence[str],
    y_train: np.ndarray,
    val_texts: Sequence[str],
    y_val: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    """Fit one model per grid point on train, score it on validation.

    Returns the best parameters (highest validation macro-F1; ties go to the first grid point,
    so list the simpler / more regularised values first) and a table of every trial.
    """
    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[n] for n in names)):
        params = dict(zip(names, values))
        pred = make_model(**params).fit(train_texts, y_train).predict(val_texts)
        rows.append({**params, "val_accuracy": accuracy(y_val, pred), "val_macro_f1": macro_f1(y_val, pred)})
    trials = pd.DataFrame(rows)
    best = trials.loc[trials["val_macro_f1"].idxmax(), names].to_dict()
    return best, trials
