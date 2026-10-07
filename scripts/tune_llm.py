"""Choose the zero-shot LLM (model, prompt) on VALIDATION data. Run on the machine with Ollama.

Uses a fixed stratified validation subset (tuning_per_class items per intent, 231 by default) so
every candidate is scored on identical messages. Writes results/llm_tuning.csv; with
`model: auto` / `prompt: auto` in configs/models.yaml the benchmark then uses the best row.

    python scripts/tune_llm.py                      # every installed candidate model
    python scripts/tune_llm.py --models qwen2.5:3b  # only these
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yaml

from llmornot.data import ROOT, make_splits
from llmornot.evaluate import accuracy, macro_f1
from llmornot.llm_client import OllamaClient
from llmornot.models.llm_zeroshot import LLMZeroShot

RESULTS = ROOT / "results"


def tuning_subset(val: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    return (val.groupby("label").sample(n=per_class, random_state=seed)
               .sort_values("row_id").reset_index(drop=True))


def resolve_installed(candidates: list[str], installed: list[str]) -> list[str]:
    """Match "qwen2.5:3b" with an installed "qwen2.5:3b" and "phi3.5" with "phi3.5:latest"."""
    found = []
    for c in candidates:
        for name in installed:
            if name == c or name == f"{c}:latest":
                found.append(name)
                break
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="override the candidate list in configs/models.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))["llm_zeroshot"]
    seed = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["seed"]
    client = OllamaClient(cfg["host"])
    client.check()
    models = resolve_installed(args.models or cfg["candidate_models"], client.installed_models())
    if not models:
        raise SystemExit(f"None of {args.models or cfg['candidate_models']} is installed. Try: ollama pull qwen2.5:3b")

    splits = make_splits()
    label_id = {label: i for i, label in enumerate(splits.labels)}
    subset = tuning_subset(splits.val, cfg["tuning_per_class"], seed)
    y = subset["label"].map(label_id).to_numpy()
    print(f"Tuning on {len(subset)} validation messages; models: {models}")

    rows = []
    for model in models:
        for prompt in cfg["candidate_prompts"]:
            llm = LLMZeroShot(splits.labels, model, prompt, cfg["options"], ROOT / cfg["cache_dir"],
                              client=client, keep_alive=cfg["keep_alive"])
            pred = llm.predict(subset["text"].tolist())
            resp = llm.last_responses
            latency = np.array([r["wall_ms"] - r["load_ms"] for r in resp])
            rows.append({
                "model": model, "prompt": prompt, "n": len(subset),
                "val_accuracy": accuracy(y, pred), "val_macro_f1": macro_f1(y, pred),
                "invalid_rate": float(np.mean(pred == -1)),
                "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in resp])),
                "p50_ms": float(np.percentile(latency, 50)),
            })
            print(rows[-1])

    table = pd.DataFrame(rows).sort_values(["val_macro_f1", "p50_ms"], ascending=[False, True])
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "llm_tuning.csv", index=False)
    print("\nBest on validation:", table.iloc[0][["model", "prompt", "val_macro_f1"]].to_dict())


if __name__ == "__main__":
    main()
