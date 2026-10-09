"""Fill the result tables in README.md from results/*.csv. No number in the README is typed by hand.

Replaces the text between <!-- NAME:START --> and <!-- NAME:END --> markers. Methods without
results yet appear as [placeholder].

    python scripts/make_report.py
"""

from __future__ import annotations

import re

import pandas as pd

from llmornot.data import ROOT

RESULTS = ROOT / "results"
README = ROOT / "README.md"
METHODS = ["majority", "tfidf_lr", "embed_lr", "llm_zeroshot"]
DISPLAY = {"majority": "Majority class", "tfidf_lr": "TF-IDF + LR",
           "embed_lr": "Sentence embeddings + LR", "llm_zeroshot": "LLM zero-shot"}
P = "[placeholder]"


def read(name: str) -> pd.DataFrame:
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["method"])


def ci(row: pd.Series, metric: str) -> str:
    return f"{row[metric]:.3f} ({row[metric + '_lo']:.3f}–{row[metric + '_hi']:.3f})"


def metric_row(metrics: pd.DataFrame, method: str, eval_set: str) -> pd.Series | None:
    hit = metrics[(metrics.method == method) & (metrics.eval_set == eval_set)]
    return None if hit.empty else hit.iloc[0]


def machine(lat: pd.Series) -> str:
    if lat["device"] == "ollama":
        return "Ollama" + (f", {lat['gpu']}" if isinstance(lat.get("gpu"), str) else "")
    hw = lat["gpu"] if str(lat["device"]).startswith("cuda") and isinstance(lat.get("gpu"), str) else lat["cpu"]
    return f"{hw} ({lat['device']})"


def main_table(metrics, latency, costs) -> str:
    lines = ["| Method | Accuracy (95% CI) | Macro-F1 (95% CI) | Invalid outputs | p50 / p95 latency (ms) "
             "| Compute-seconds per 1k queries | INR per 1k queries | Latency measured on |",
             "|---|---|---|---|---|---|---|---|"]
    for m in METHODS:
        r = metric_row(metrics, m, "sample500")
        lat = latency[latency.method == m]
        cost = costs[costs.method == m]
        if r is None or lat.empty:
            lines.append(f"| {DISPLAY[m]} | {P} | {P} | {P} | {P} | {P} | {P} | {P} |")
            continue
        lat, cost = lat.iloc[0], cost.iloc[0]
        inr = P if pd.isna(cost["inr_per_1k"]) else f"{cost['inr_per_1k']:.4f}"
        lines.append(f"| {DISPLAY[m]} | {ci(r, 'accuracy')} | {ci(r, 'macro_f1')} | {r['invalid_rate']:.1%} "
                     f"| {lat['p50_ms']:.2f} / {lat['p95_ms']:.2f} | {cost['compute_seconds_per_1k']:.2f} "
                     f"| {inr} | {machine(lat)} |")
    return "\n".join(lines)


def overlap_table(metrics) -> str:
    sets = [("test_full", "Full test (3,080)"), ("test_low_overlap", "Low-overlap test"),
            ("sample500", "500 sample"), ("sample500_low_overlap", "Low-overlap 500 sample")]
    lines = ["| Method | " + " | ".join(f"{label} accuracy" for _, label in sets) + " |",
             "|---|" + "---|" * len(sets)]
    for m in METHODS:
        cells = []
        for key, _ in sets:
            r = metric_row(metrics, m, key)
            cells.append("n/a (not run on this set)" if r is None and m == "llm_zeroshot" and key.startswith("test_")
                         else P if r is None else f"{ci(r, 'accuracy')} n={int(r['n'])}")
        lines.append(f"| {DISPLAY[m]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def observations(metrics) -> str:
    """Plain sentences computed from the CSVs, so even the prose numbers come from a script."""
    out = []
    for m in METHODS:
        full, low = metric_row(metrics, m, "test_full"), metric_row(metrics, m, "test_low_overlap")
        s500, s500_low = metric_row(metrics, m, "sample500"), metric_row(metrics, m, "sample500_low_overlap")
        if full is not None and low is not None and m != "majority":
            out.append(f"- {DISPLAY[m]}: accuracy {full['accuracy']:.3f} on the full test set vs "
                       f"{low['accuracy']:.3f} on the low-overlap subset "
                       f"({100 * (low['accuracy'] - full['accuracy']):+.1f} points).")
        if full is not None and s500 is not None and m != "majority":
            out.append(f"- {DISPLAY[m]}: accuracy {s500['accuracy']:.3f} on the 500 sample vs {full['accuracy']:.3f} "
                       f"on the full test set ({100 * (s500['accuracy'] - full['accuracy']):+.1f} points): "
                       "a reminder that a 500-item sample has its own luck; compare methods within one set only.")
        if m == "llm_zeroshot" and s500 is not None and s500_low is not None:
            out.append(f"- {DISPLAY[m]}: accuracy {s500['accuracy']:.3f} on the 500 sample vs "
                       f"{s500_low['accuracy']:.3f} on its low-overlap part.")
    return "\n".join(out) if out else P


def duplicates_table() -> str:
    d = read("duplicates_summary.csv")
    if d.empty:
        return P
    lines = ["| Check | Count | Out of | % | Same label (%) |", "|---|---|---|---|---|"]
    for _, r in d.iterrows():
        agree = "" if pd.isna(r["label_agreement_pct"]) else f"{r['label_agreement_pct']:.1f}"
        lines.append(f"| `{r['check']}` | {r['count']} | {r['out_of']} | {r['pct']:.2f} | {agree} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        raise SystemExit(f"README.md has no {name} markers")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> None:
    metrics, latency, costs = read("metrics.csv"), read("latency.csv"), read("costs.csv")
    text = README.read_text()
    text = replace_block(text, "MAIN_TABLE", main_table(metrics, latency, costs))
    text = replace_block(text, "OVERLAP_TABLE", overlap_table(metrics))
    text = replace_block(text, "OBSERVATIONS", observations(metrics))
    text = replace_block(text, "DUPLICATES", duplicates_table())
    README.write_text(text)
    print("README.md tables updated from results/")


if __name__ == "__main__":
    main()
