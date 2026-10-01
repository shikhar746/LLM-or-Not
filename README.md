# LLM-or-Not

**The cost–accuracy frontier for customer-intent routing.**

A bank must route each customer message to one of 77 intents. Which approach gives the best
accuracy per rupee and per millisecond — and when is an LLM actually worth it?

The end product is a decision backed by evidence, not a chatbot.

## Status

Work in progress (MVP). See [docs/PLAN.md](docs/PLAN.md) for the plan and
[DECISIONS.md](DECISIONS.md) for every design choice and why it was made.

## Results

| Method | Accuracy (95% CI) | Macro-F1 (95% CI) | p50 / p95 latency (ms) | Cost per 1k queries |
|---|---|---|---|---|
| Majority class | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| TF-IDF + LR | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| Sentence embeddings + LR | [placeholder] | [placeholder] | [placeholder] | [placeholder] |
| LLM zero-shot | [placeholder] | [placeholder] | [placeholder] | [placeholder] |

Every number in this README will be produced by `scripts/run_benchmark.py`. Nothing is estimated.

## Data

[Banking77](https://huggingface.co/datasets/PolyAI/banking77) (PolyAI), CC BY 4.0:
13,083 customer-service queries, 77 intents, 10,003 train / 3,080 test.
