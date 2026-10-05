"""Method 3: frozen sentence embeddings into logistic regression.

A sentence-transformers model turns each message into a fixed-length vector (384 numbers for
all-MiniLM-L6-v2) such that messages with similar meaning get nearby vectors, even when they share
no words ("my card hasn't come" ~ "still waiting for the card"). The encoder is NOT trained here
("frozen"); only a logistic regression on top of the vectors is fitted. That keeps training cheap
and isolates one question: do pretrained meaning-vectors beat word counts (TF-IDF)?
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from llmornot.models.base import IntentClassifier


def load_encoder(model_name: str, device: str | None = None):
    # Imported here so the rest of the package works without PyTorch installed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class EmbedLogReg(IntentClassifier):
    name = "embed_lr"

    def __init__(self, n_classes: int, encoder, C: float = 10.0, batch_size: int = 64, seed: int = 42):
        """`encoder` is anything with .encode(list_of_texts, ...) -> 2-D array (a SentenceTransformer
        in practice; a small fake in the tests). It is passed in, so a grid search loads it once."""
        super().__init__(n_classes)
        self.encoder = encoder
        self.C = C
        self.batch_size = batch_size
        self.seed = seed

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        # normalize_embeddings=True scales every vector to length 1, so only direction (meaning)
        # matters, not length. This is the standard way these models are compared.
        return np.asarray(self.encoder.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ))

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "EmbedLogReg":
        self.clf_ = LogisticRegression(C=self.C, max_iter=3000, random_state=self.seed)
        self.clf_.fit(self.embed(texts), y)
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        proba = np.zeros((len(texts), self.n_classes))
        proba[:, self.clf_.classes_] = self.clf_.predict_proba(self.embed(texts))
        return proba

    def params(self) -> dict:
        return {"C": self.C}
