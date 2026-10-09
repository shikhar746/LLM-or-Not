"""Most-confused intent pairs, plus a CSV of 100 errors for hand review.

    python scripts/error_analysis.py --method tfidf_lr

Writes
  results/confused_pairs_<method>.csv  - (true intent, predicted intent, count), most frequent first
  results/error_review_<method>.csv    - 100 seeded random errors with an empty `tag` column to fill in:
                                           label_noise  the gold label looks wrong or the message fits both
                                           model_error  the gold label is right and the model is wrong
                                         The file is never overwritten once it exists (use --force),
                                         so your hand-tagging cannot be lost by re-running.
"""

from __future__ import annotations

import argparse

import pandas as pd

from llmornot.data import ROOT, load_config, make_splits

RESULTS = ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    splits = make_splits()
    labels = splits.labels
    pred = pd.read_csv(RESULTS / "predictions" / f"{args.method}.csv")
    test = splits.test.set_index("row_id")
    nearest = pd.read_csv(RESULTS / "test_nearest_train.csv").set_index("test_row")

    errors = pred[pred.y_true != pred.y_pred].copy()
    errors["true_label"] = errors.y_true.map(lambda i: labels[i])
    errors["predicted_label"] = errors.y_pred.map(lambda i: "INVALID" if i < 0 else labels[i])
    print(f"{args.method}: {len(errors)} errors out of {len(pred)} predictions")

    pairs = (errors.groupby(["true_label", "predicted_label"]).size()
                   .rename("count").sort_values(ascending=False).reset_index())
    pairs.to_csv(RESULTS / f"confused_pairs_{args.method}.csv", index=False)
    print(pairs.head(10).to_string(index=False))

    out = RESULTS / f"error_review_{args.method}.csv"
    if out.exists() and not args.force:
        print(f"{out.name} exists; not overwriting (use --force).")
        return
    sample = errors.sample(min(args.n, len(errors)), random_state=load_config()["seed"]).sort_values("row_id")
    review = pd.DataFrame({
        "row_id": sample.row_id,
        "text": test.loc[sample.row_id, "text"].values,
        "true_label": sample.true_label,
        "predicted_label": sample.predicted_label,
        "nearest_train_text": nearest.loc[sample.row_id, "train_text"].values,
        "nearest_train_label": nearest.loc[sample.row_id, "train_label"].values,
        "tag": "",    # fill in: label_noise / model_error
        "notes": "",
    })
    review.to_csv(out, index=False)
    print(f"Wrote {out.name}: {len(review)} errors to tag by hand.")


if __name__ == "__main__":
    main()
