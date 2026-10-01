# MVP plan

Scope: methods 1 (majority), 2 (TF-IDF + LR), 3 (sentence embeddings + LR), 5 (LLM zero-shot).
Output: accuracy / macro-F1 / latency / cost table with CIs, README with methodology, results, limitations.

## Repo layout

```
llm-or-not/
├── llmornot/
│   ├── data.py            # load Banking77 (pinned revision), dedup check, splits, 500-query sample
│   ├── evaluate.py        # accuracy, macro-F1, paired bootstrap CIs, latency timing
│   ├── costs.py           # cost per 1k queries (API tokens x price, or CPU/GPU-seconds)
│   ├── labels.py          # the 77 label names + short descriptions, output parsing
│   └── models/
│       ├── base.py        # shared interface: fit / predict_proba / predict
│       ├── majority.py
│       ├── tfidf_lr.py
│       ├── embed_lr.py
│       └── llm_zeroshot.py  # constrained output, JSONL cache, invalid-output count
├── configs/               # one YAML per method (hyperparameter grids, model names, prices)
├── prompts/               # prompt templates, under version control
├── scripts/
│   ├── check_duplicates.py
│   └── run_benchmark.py   # writes results/*.csv and plots
├── results/               # CSVs + plots (generated, committed)
├── tests/                 # pytest: label parsing, metrics, cost calculator
├── DECISIONS.md
├── requirements.txt       # pinned versions
└── README.md
```

## Steps

| # | Step | What you should understand afterwards |
|---|------|------------------------------------|
| 0 | Repo skeleton, pinned environment, this plan | Why pin versions and seeds |
| 1 | Data: load, exact + near-duplicate check train↔test, stratified validation split, seeded 500 sample | Leakage, stratification |
| 2 | Metrics + bootstrap CIs + cost calculator, with tests | What a CI means; paired bootstrap |
| 3 | Shared model interface; majority + TF-IDF/LR tuned on validation | Why a dumb baseline matters; char n-grams |
| 4 | Sentence embeddings + LR | What an embedding is; frozen encoder + linear head |
| 5 | LLM zero-shot: label descriptions, prompt, constrained output, JSONL cache; prompt tuned on validation only | Constrained decoding; why cache; invalid outputs |
| 6 | `run_benchmark.py`: full table + plot, latency p50/p95, cost | Measuring latency honestly |
| 7 | README: methodology, results from CSV, limitations | Reporting a negative result plainly |
