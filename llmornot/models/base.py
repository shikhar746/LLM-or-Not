"""The one interface every method implements, so the benchmark can treat them all the same."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class IntentClassifier(ABC):
    """fit / predict_proba / predict, with labels as integer class ids 0..n_classes-1.

    predict_proba returns an (n_texts, n_classes) array whose rows sum to 1. Methods that cannot
    give probabilities (the zero-shot LLM) return a one-hot row, or all zeros for an invalid answer.
    """

    name: str = "base"

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    @abstractmethod
    def fit(self, texts: Sequence[str], y: np.ndarray) -> "IntentClassifier":
        ...

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        """Most probable class per text; -1 where the model gave no valid answer (all-zero row)."""
        proba = self.predict_proba(texts)
        pred = proba.argmax(axis=1)
        pred[proba.max(axis=1) == 0] = -1
        return pred

    def params(self) -> dict:
        """Settings worth recording next to the results (chosen hyperparameters, model names)."""
        return {}
