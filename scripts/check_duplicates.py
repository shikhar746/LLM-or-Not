"""Check for exact and near-duplicate overlap between Banking77 train and test.

Why: if a test query (or a trivial rewording of it) also sits in train, a model can score it by
memory rather than understanding, and test accuracy overstates real-world accuracy.

Method:
  exact      - identical after normalise() (lowercase, no punctuation, single spaces)
  near       - cosine similarity of character 3-5-gram TF-IDF vectors; each test item is compared
               with every train item and we keep its single closest match. There is no "correct"
               threshold, so we report several and write the pairs out for eyeballing.
               Note: char_wb n-grams ignore word order, so a pure reordering ("Why was X declined?"
               vs "X was declined. Why?") scores 1.0. This measures lexical overlap, not identity.

Outputs:
  results/duplicates_summary.csv  - one row per check
  results/test_nearest_train.csv  - every test item with its closest train item and similarity
                                    (later used to score models on a low-overlap test subset)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from llmornot.data import ROOT, load_raw, normalise

THRESHOLDS = [0.95, 0.90, 0.80]
RESULTS = ROOT / "results"


def main() -> None:
    train, test = load_raw()
    train["norm"] = train["text"].map(normalise)
    test["norm"] = test["text"].map(normalise)
    rows = []

    def add(check: str, n: int, of: int, label_agree: float | None = None) -> None:
        rows.append({"check": check, "count": n, "out_of": of, "pct": round(100 * n / of, 2),
                     "label_agreement_pct": None if label_agree is None else round(100 * label_agree, 1)})

    # 1. Duplicates inside each split (same normalised text appearing more than once).
    for name, df in [("train", train), ("test", test)]:
        dup = df["norm"].duplicated(keep="first")
        add(f"within_{name}_exact_duplicates", int(dup.sum()), len(df))
        conflicting = df.groupby("norm")["label"].nunique()
        add(f"within_{name}_texts_with_conflicting_labels", int((conflicting > 1).sum()), df["norm"].nunique())

    # 2. Exact overlap: test texts whose normalised form also appears in train.
    train_labels = train.groupby("norm")["label"].agg(set)
    in_train = test["norm"].isin(train_labels.index)
    agree = test[in_train].apply(lambda r: r["label"] in train_labels[r["norm"]], axis=1)
    add("test_exact_in_train", int(in_train.sum()), len(test), agree.mean() if in_train.any() else None)

    # 3. Near-duplicates via character n-gram TF-IDF cosine similarity.
    #    TF-IDF rows are L2-normalised, so a dot product equals cosine similarity.
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(pd.concat([train["norm"], test["norm"]]))
    sims = vec.transform(test["norm"]) @ vec.transform(train["norm"]).T  # sparse, 3080 x 10003
    best_idx = np.asarray(sims.argmax(axis=1)).ravel()
    best_sim = np.asarray(sims.max(axis=1).todense()).ravel()

    nearest = pd.DataFrame({
        "test_row": test.index,
        "test_text": test["text"],
        "test_label": test["label"],
        "train_row": best_idx,
        "train_text": train["text"].values[best_idx],
        "train_label": train["label"].values[best_idx],
        "similarity": best_sim.round(4),
    })
    nearest["same_label"] = nearest["test_label"] == nearest["train_label"]
    for t in THRESHOLDS:
        hit = nearest["similarity"] >= t
        add(f"test_near_dup_sim_ge_{t:.2f}", int(hit.sum()), len(test),
            nearest.loc[hit, "same_label"].mean() if hit.any() else None)

    RESULTS.mkdir(exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "duplicates_summary.csv", index=False)
    nearest.sort_values("similarity", ascending=False).to_csv(RESULTS / "test_nearest_train.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
