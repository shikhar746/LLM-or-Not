"""Baseline 2: TF-IDF features (word + character n-grams) into multinomial logistic regression.

Why two kinds of n-grams:
  word 1-2 grams   capture phrases such as "top up", "exchange rate", "not received"
  char 2-5 grams   (inside word boundaries) survive typos and word forms: "topup", "top-up",
                   "verifying" vs "verify" share most of their character chunks
TF-IDF down-weights words that appear in almost every message ("my", "card", "I").
sublinear_tf uses 1 + log(count), so a word repeated three times is not three times as important.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from llmornot.models.base import IntentClassifier


class TfidfLogReg(IntentClassifier):
    name = "tfidf_lr"

    def __init__(self, n_classes: int, C: float = 10.0, seed: int = 42):
        super().__init__(n_classes)
        self.C = C
        self.seed = seed

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "TfidfLogReg":
        features = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
        ])
        # C = inverse regularisation strength: larger C trusts the training data more.
        clf = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
        self.pipeline_ = Pipeline([("features", features), ("clf", clf)]).fit(list(texts), y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        # Expand to all n_classes columns in case a class was missing from the training data
        # (possible in the small learning-curve subsets later).
        proba = np.zeros((len(texts), self.n_classes))
        proba[:, self.pipeline_.classes_] = self.pipeline_.predict_proba(list(texts))
        return proba

    def params(self) -> dict:
        return {"C": self.C}
