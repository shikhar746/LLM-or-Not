"""Run the MVP benchmark and write results/*.csv plus the accuracy-vs-latency plot.

    python scripts/run_benchmark.py                                  # every method
    python scripts/run_benchmark.py --methods majority tfidf_lr      # a subset
    python scripts/run_benchmark.py --methods llm_zeroshot --offline # LLM replayed from llm_cache/

Methods can be run separately (even on different machines): each run replaces only its own rows
in the CSVs. Every latency row records the machine it was measured on.

Evaluation sets (DECISIONS.md D3, D6):
  test_full               all 3,080 test items           (not the LLM: too slow/expensive)
  test_low_overlap        test items whose nearest train item has similarity < 0.90
  sample500               the fixed seeded 500-item test sample (every method)
  sample500_low_overlap   the low-overlap part of that sample
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from llmornot.costs import compute_seconds_per_1k, local_cost_per_1k
from llmornot.data import ROOT, load_config, make_splits
from llmornot.evaluate import accuracy, bootstrap_ci, bootstrap_indices, latency_ms, macro_f1, percentiles
from llmornot.tuning import grid_search

RESULTS = ROOT / "results"
METHODS = ["majority", "tfidf_lr", "embed_lr", "llm_zeroshot"]
DISPLAY = {"majority": "Majority class", "tfidf_lr": "TF-IDF + LR",
           "embed_lr": "Sentence embeddings + LR", "llm_zeroshot": "LLM zero-shot"}
LOW_OVERLAP_THRESHOLD = 0.90
N_BOOT = 2000


def machine_info(device: str) -> dict:
    cpu = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {"os": platform.platform(), "python": platform.python_version(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "gpu": gpu, "device": device}


def upsert(df: pd.DataFrame, path: Path, methods: list[str]) -> pd.DataFrame:
    """Replace this run's methods in an existing CSV, keep the other methods' rows."""
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old[~old["method"].isin(methods)], df], ignore_index=True)
    df["order"] = df["method"].map({m: i for i, m in enumerate(METHODS)})
    df = df.sort_values([c for c in ["order", "eval_set"] if c in df]).drop(columns="order")
    df.to_csv(path, index=False)
    return df


def build_and_fit(method: str, cfg: dict, splits, y_train, y_val, seed: int, offline: bool):
    """Tune on validation (if the method has a grid), refit on train + validation. Returns (model, device)."""
    n = len(splits.labels)
    train_val_texts = pd.concat([splits.train.text, splits.val.text]).tolist()
    y_train_val = np.concatenate([y_train, y_val])
    mcfg = cfg.get(method) or {}

    if method == "majority":
        from llmornot.models.majority import MajorityClass
        return MajorityClass(n).fit(train_val_texts, y_train_val), "cpu"

    if method == "tfidf_lr":
        from llmornot.models.tfidf_lr import TfidfLogReg
        make = lambda **p: TfidfLogReg(n, seed=seed, **p)  # noqa: E731
        device = "cpu"
    elif method == "embed_lr":
        from llmornot.models.embed_lr import EmbedLogReg, load_encoder
        encoder = load_encoder(mcfg["model_name"], mcfg.get("device"))
        make = lambda **p: EmbedLogReg(n, encoder, batch_size=mcfg["batch_size"], seed=seed, **p)  # noqa: E731
        device = str(encoder.device)
    elif method == "llm_zeroshot":
        return build_llm(mcfg, splits.labels, offline), "ollama"
    else:
        raise ValueError(method)

    best, trials = grid_search(make, mcfg["grid"], splits.train.text.tolist(), y_train,
                               splits.val.text.tolist(), y_val)
    trials.to_csv(RESULTS / f"tuning_{method}.csv", index=False)
    print(f"  {method}: best on validation {best}")
    return make(**best).fit(train_val_texts, y_train_val), device


def build_llm(mcfg: dict, labels: list[str], offline: bool):
    from llmornot.llm_client import OllamaClient
    from llmornot.models.llm_zeroshot import LLMZeroShot

    model, prompt = mcfg["model"], mcfg["prompt"]
    if "auto" in (model, prompt):
        tuning = RESULTS / "llm_tuning.csv"
        if not tuning.exists():
            raise SystemExit("model/prompt is 'auto' but results/llm_tuning.csv is missing: run scripts/tune_llm.py first.")
        best = pd.read_csv(tuning).sort_values(["val_macro_f1", "p50_ms"], ascending=[False, True]).iloc[0]
        model = best["model"] if model == "auto" else model
        prompt = best["prompt"] if prompt == "auto" else prompt
    client = None
    if not offline:
        client = OllamaClient(mcfg["host"])
        client.check()
    print(f"  llm_zeroshot: model={model} prompt={prompt} ({'offline replay' if offline else 'live'})")
    return LLMZeroShot(labels, model, prompt, mcfg["options"], ROOT / mcfg["cache_dir"],
                       client=client, keep_alive=mcfg["keep_alive"])


def plot(metrics: pd.DataFrame, latency: pd.DataFrame, path: Path) -> None:
    """Accuracy on the shared 500-item sample vs p50 latency (log scale), with 95% CI bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed categorical order (validated palette): colour follows the method, never its rank.
    colours = {"majority": "#2a78d6", "tfidf_lr": "#eb6834", "embed_lr": "#1baf7a", "llm_zeroshot": "#eda100"}
    m = metrics[metrics["eval_set"] == "sample500"].merge(latency, on="method")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for _, r in m.iterrows():
        yerr = [[r["accuracy"] - r["accuracy_lo"]], [r["accuracy_hi"] - r["accuracy"]]]
        ax.errorbar(r["p50_ms"], r["accuracy"], yerr=yerr, fmt="o", ms=8, lw=1.5, capsize=0,
                    color=colours[r["method"]], label=DISPLAY[r["method"]])
        ax.annotate(DISPLAY[r["method"]], (r["p50_ms"], r["accuracy"]), xytext=(8, 6),
                    textcoords="offset points", fontsize=9, color="#0b0b0b")
    ax.set_xscale("log")
    ax.set_xlim(m["p50_ms"].min() / 3, m["p50_ms"].max() * 8)  # room for the direct labels
    ax.set_xlabel("p50 latency per query (ms, log scale)", color="#52514e")
    ax.set_ylabel("Accuracy on 500-query test sample", color="#52514e")
    ax.set_ylim(0, 1)
    ax.grid(True, which="major", color="#e4e3df", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Accuracy vs latency (bars: 95% bootstrap CI)", fontsize=11, color="#0b0b0b", loc="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    ap.add_argument("--offline", action="store_true", help="LLM: replay llm_cache/ only, never call Ollama")
    args = ap.parse_args()

    data_cfg = load_config()
    seed = data_cfg["seed"]
    model_cfg = yaml.safe_load(open(ROOT / "configs" / "models.yaml"))
    prices = yaml.safe_load(open(ROOT / "configs" / "costs.yaml"))["local"]
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "predictions").mkdir(exist_ok=True)

    splits = make_splits(data_cfg)
    label_id = {label: i for i, label in enumerate(splits.labels)}
    y_train = splits.train.label.map(label_id).to_numpy()
    y_val = splits.val.label.map(label_id).to_numpy()

    nearest_path = RESULTS / "test_nearest_train.csv"
    if not nearest_path.exists():
        raise SystemExit("results/test_nearest_train.csv missing: run scripts/check_duplicates.py first.")
    nearest = pd.read_csv(nearest_path).set_index("test_row")["similarity"]
    test = splits.test.assign(y=lambda d: d.label.map(label_id), max_sim=lambda d: d.row_id.map(nearest))
    sample_ids = set(splits.llm_sample.row_id)
    eval_sets = {
        "test_full": test,
        "test_low_overlap": test[test.max_sim < LOW_OVERLAP_THRESHOLD],
        "sample500": test[test.row_id.isin(sample_ids)],
        "sample500_low_overlap": test[test.row_id.isin(sample_ids) & (test.max_sim < LOW_OVERLAP_THRESHOLD)],
    }
    # One set of bootstrap indices per evaluation set, shared by every method (paired bootstrap).
    boot = {name: bootstrap_indices(len(df), N_BOOT, seed) for name, df in eval_sets.items()}

    metric_rows, latency_rows, cost_rows, params = [], [], [], {}
    for method in args.methods:
        print(f"[{method}]")
        model, device = build_and_fit(method, model_cfg, splits, y_train, y_val, seed, args.offline)
        params[method] = model.params()
        sample = eval_sets["sample500"]
        is_llm = method == "llm_zeroshot"

        # Predictions. The LLM is only run on the 500-item sample (DECISIONS.md D3).
        scored = sample if is_llm else test
        pred = pd.Series(model.predict(scored.text.tolist()), index=scored.row_id.values)
        pd.DataFrame({"row_id": scored.row_id, "y_true": scored.y, "y_pred": pred.values}).to_csv(
            RESULTS / "predictions" / f"{method}.csv", index=False)

        for name, df in eval_sets.items():
            if is_llm and not name.startswith("sample500"):
                continue
            y_true, y_pred = df.y.to_numpy(), pred.loc[df.row_id].to_numpy()
            acc = bootstrap_ci(y_true, y_pred, accuracy, boot[name])
            f1 = bootstrap_ci(y_true, y_pred, macro_f1, boot[name])
            metric_rows.append({"method": method, "eval_set": name, "n": len(df),
                                "accuracy": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
                                "macro_f1": f1[0], "macro_f1_lo": f1[1], "macro_f1_hi": f1[2],
                                "invalid_rate": float(np.mean(y_pred == -1))})

        # Latency: one query at a time on the 500 sample texts.
        if is_llm:
            # Recorded when each call was made (also when replayed from cache), minus model-load time.
            times = np.array([r["wall_ms"] - r["load_ms"] for r in model.last_responses])
            tokens_in = float(np.mean([r["prompt_tokens"] for r in model.last_responses]))
            tokens_out = float(np.mean([r["completion_tokens"] for r in model.last_responses]))
            if model.chat.client is not None:
                params[method]["ollama_loaded"] = model.chat.client.running()
        else:
            times = latency_ms(lambda t: model.predict([t]), sample.text.tolist())
            tokens_in = tokens_out = None
        latency_rows.append({"method": method, "n": len(times), **percentiles(times),
                             "mean_ms": float(times.mean()), **machine_info(device)})

        # Cost per 1,000 queries (DECISIONS.md D9). INR only if an hourly price is configured.
        mean_s = float(times.mean()) / 1000
        on_gpu = device.startswith("cuda") or is_llm
        price = prices["gpu_price_per_hour"] if on_gpu else prices["cpu_price_per_hour"]
        cost_rows.append({"method": method, "hardware": "gpu" if on_gpu else "cpu",
                          "compute_seconds_per_1k": compute_seconds_per_1k(mean_s),
                          "inr_per_1k": None if price is None else local_cost_per_1k(mean_s, price),
                          "mean_input_tokens": tokens_in, "mean_output_tokens": tokens_out})
        sample_row = next(r for r in metric_rows if r["method"] == method and r["eval_set"] == "sample500")
        print(f"  sample500 accuracy={sample_row['accuracy']:.4f}  p50={latency_rows[-1]['p50_ms']:.2f} ms")

    metrics = upsert(pd.DataFrame(metric_rows), RESULTS / "metrics.csv", args.methods)
    latency = upsert(pd.DataFrame(latency_rows), RESULTS / "latency.csv", args.methods)
    upsert(pd.DataFrame(cost_rows), RESULTS / "costs.csv", args.methods)
    params_path = RESULTS / "params.json"
    all_params = json.loads(params_path.read_text()) if params_path.exists() else {}
    params_path.write_text(json.dumps({**all_params, **params}, indent=2, default=str))
    plot(metrics, latency, RESULTS / "accuracy_vs_latency.png")
    print("Wrote results/metrics.csv, latency.csv, costs.csv, params.json, accuracy_vs_latency.png")


if __name__ == "__main__":
    main()
