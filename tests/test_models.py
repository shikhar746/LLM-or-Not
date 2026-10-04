import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from llmornot.models.embed_lr import EmbedLogReg
from llmornot.models.majority import MajorityClass
from llmornot.models.tfidf_lr import TfidfLogReg
from llmornot.tuning import grid_search

TEXTS = ["my card has not arrived", "where is my new card", "card still not here",
         "what is the exchange rate", "exchange rate for euros", "rate to convert dollars"]
Y = np.array([0, 0, 0, 1, 1, 1])


class FakeEncoder:
    """Stands in for a SentenceTransformer: same .encode() signature, no download."""

    def encode(self, texts, batch_size=32, normalize_embeddings=False, convert_to_numpy=True, show_progress_bar=False):
        X = HashingVectorizer(n_features=64, analyzer="char_wb", ngram_range=(2, 3)).transform(texts).toarray()
        return X / np.linalg.norm(X, axis=1, keepdims=True) if normalize_embeddings else X


def test_majority_predicts_most_frequent_class():
    m = MajorityClass(n_classes=3).fit(TEXTS, np.array([2, 2, 1, 2, 0, 0]))
    assert (m.predict(["anything", "else"]) == 2).all()
    assert np.allclose(m.predict_proba(["x"]).sum(axis=1), 1)


def test_tfidf_lr_learns_toy_task_and_fills_all_class_columns():
    m = TfidfLogReg(n_classes=5).fit(TEXTS, Y)  # classes 2-4 never seen in training
    proba = m.predict_proba(["has my card arrived", "euro exchange rate"])
    assert proba.shape == (2, 5) and np.allclose(proba.sum(axis=1), 1)
    assert m.predict(["has my card arrived", "euro exchange rate"]).tolist() == [0, 1]


def test_embed_lr_with_fake_encoder():
    m = EmbedLogReg(n_classes=2, encoder=FakeEncoder()).fit(TEXTS, Y)
    assert m.predict(["card not arrived", "exchange rate"]).tolist() == [0, 1]


def test_grid_search_returns_best_params_and_all_trials():
    best, trials = grid_search(lambda C: TfidfLogReg(2, C=C), {"C": [0.01, 10]}, TEXTS, Y, TEXTS, Y)
    assert len(trials) == 2 and set(best) == {"C"}
    assert trials["val_macro_f1"].max() == trials.loc[trials["C"] == best["C"], "val_macro_f1"].item()
