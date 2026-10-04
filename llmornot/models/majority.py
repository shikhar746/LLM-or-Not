"""Baseline 1: always predict the most frequent training intent.

It has no skill at all, so any real method must beat it by a wide margin. It also exposes the
floor of the metrics: on Banking77's balanced test set (40 per intent) it scores 1/77 accuracy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from llmornot.models.base import IntentClassifier


class MajorityClass(IntentClassifier):
    name = "majority"

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "MajorityClass":
        self.majority_ = int(np.bincount(y, minlength=self.n_classes).argmax())
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        proba = np.zeros((len(texts), self.n_classes))
        proba[:, self.majority_] = 1.0
        return proba

    def params(self) -> dict:
        return {"majority_class_id": self.majority_}
