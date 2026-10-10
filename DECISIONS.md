# Decisions

Each entry: what was decided, why, and what would change the decision.

## D1. Primary metrics: accuracy and macro-F1, with bootstrap 95% CIs
- **Why accuracy:** it is what a routing desk feels — the share of messages sent to the right queue.
- **Why macro-F1 as well:** it weights all 77 intents equally, so a model cannot look good by
  doing well only on frequent intents.
- **Why bootstrap CIs:** test sets are finite; a 1-point gap may be noise. Resampling the test
  items (with the *same* resampled indices for every model) shows how much a score could move.

## D2. Test set is touched once per final configuration
- All tuning (hyperparameters, prompt wording) uses a stratified validation split carved from train.
- **Why:** tuning on test silently overfits to it and inflates every number we report.

## D3. LLM evaluated on a fixed, seeded 500-query test sample (MVP)
- **Why:** LLM calls are slow (local) or cost money (API). 500 items keeps the run affordable.
- **Consequence:** every other method is *also* scored on the same 500 items, so the comparison is
  like-for-like. Methods 1–3 are additionally scored on the full 3,080-item test set.
- **Trade-off:** CIs on 500 items are wider (roughly ±4 accuracy points near 50%, narrower near 90%).

## D4. Pinned environment, one seed in one place
- Exact versions in `requirements.txt`; the global seed lives in `configs/data.yaml`.
- **Why:** a different scikit-learn or numpy version can change results slightly. Pinning plus a
  fixed seed means anyone re-running the scripts gets the same numbers.

## D5. Validation split: 10% of train, stratified (1,001 rows; 3-19 per intent)
- **Why 10%:** keeps 9,002 rows for training while giving enough rows to compare hyperparameter
  settings. Stratified so every intent appears in validation.
- **Weakness:** some intents have only 3 validation examples, so per-intent validation scores are
  noisy. We only use validation for overall accuracy / macro-F1 when choosing settings.

## D6. Train/test overlap: keep the official test set, also report a low-overlap subset
Measured by `scripts/check_duplicates.py` (see `results/duplicates_summary.csv`):
- Exact overlap after normalisation is small: 11 of 3,080 test items (0.36%).
- Lexical near-duplicates are common: 369 test items (11.98%) have a train item with char-n-gram
  similarity >= 0.95, and 700 (22.73%) >= 0.90. Most are word reorderings or one-word edits of a
  train sentence (Banking77 was written by people paraphrasing each other).
- **Decision:** headline results use the full official test set, so they are comparable with
  published Banking77 numbers. We additionally report every method on the test items whose
  closest train item has similarity < 0.90 (the "low-overlap subset").
- **Why this matters for the research question:** TF-IDF matches surface words, so it should
  benefit most from near-duplicates; a zero-shot LLM sees no training data, so it cannot. Scoring
  on the low-overlap subset tests whether the TF-IDF advantage (if any) survives.
- **Threshold 0.90 is a judgement call:** a random sample of pairs above it were reorderings or
  near-identical rewrites. It is reported, not tuned.
- Near-duplicate pairs whose labels disagree (e.g. "How long does a card delivery take?" =
  `card_arrival` in test vs `card_delivery_estimate` in train) hint at label noise or genuinely
  overlapping intents. These feed the error analysis later.

## D7. Paired percentile bootstrap, 2,000 resamples
- Resample test items with replacement 2,000 times; the CI is the 2.5th-97.5th percentile of the
  metric across resamples.
- **Paired:** the same 2,000 resampled index sets are used for every model, so all models are
  judged on identical draws (needed for fair model-vs-model comparisons later).
- **Why 2,000:** enough for stable 2.5%/97.5% percentiles; costs seconds, not minutes.
- macro-F1 is written by hand (vectorised with `np.bincount`) so 2,000 resamples are fast; a test
  checks it equals scikit-learn's `f1_score(average="macro")`.

## D8. Latency: one query at a time, wall-clock, after warm-up
- Each message is classified on its own, as in live routing. We report p50 and p95.
- The first 10 calls are discarded (one-off model loading, caches).
- Batch throughput would flatter the fast models further; it is not what a single customer feels.
- Measured on the author's laptop (RTX 3050 4 GB, 16 GB RAM), not the development container.

## D9. Cost of local models = compute time x an hourly machine price
- A local model has no per-token bill, but it is not free: it occupies a machine.
- We report compute-seconds per 1,000 queries (always measurable) and convert to INR only with an
  hourly price written in `configs/costs.yaml`. Until that price is chosen, the cost column says so
  instead of showing a guess.

## D10. Hyperparameters: grid search on validation, then refit on train + validation
- Each grid point is fitted on train (9,002) and scored on validation (1,001); the best by
  validation macro-F1 wins (ties go to the stronger regularisation, listed first).
- The winner is refit on all 10,003 labelled rows before the single test evaluation, so the final
  model uses all labelled data. Every trial is saved to `results/tuning_<method>.csv`.
- **Why macro-F1 for selection:** the test set is balanced across intents, so the selection
  criterion should weigh every intent equally too.

## D11. TF-IDF features: word 1-2 grams + character 2-5 grams, sublinear TF, logistic regression
- Word n-grams catch phrases ("top up", "exchange rate"); character n-grams catch typos and word
  forms ("topup" / "top-up", "verify" / "verifying"). Combining both is a standard strong text baseline.
- Logistic regression: fast, linear, gives probabilities (needed later for calibration/abstention).
- Only C (regularisation strength) is tuned; the n-gram ranges are fixed up front to keep the
  search small and the baseline honest rather than over-tuned.

## D12. Sentence encoder: all-MiniLM-L6-v2, frozen, L2-normalised vectors
- 22M parameters, 384-dimensional vectors; fast on a laptop CPU and tiny on a 4 GB GPU.
- Frozen (not fine-tuned) on purpose: fine-tuning an encoder is method 4 (Phase 2). Here we test
  whether general-purpose meaning-vectors plus a linear classifier beat word-count features.
- Vectors are normalised to length 1 (the model's intended use); C is tuned on validation.
- The encoder is passed into the model, so the grid search loads it once; tests use a fake
  encoder so they run without downloading anything.
- Limitation: the model is fetched from Hugging Face by name without a pinned revision.

## D13. LLM backend: local Ollama, small model that fits a 4 GB GPU
- Chosen by the project owner: free, private, no API budget needed.
- Hardware: RTX 3050 with 4 GB VRAM. A 4-bit 3-4B model (~2-2.5 GB) fits fully in VRAM; a 7-8B
  model (~4.7 GB) would spill onto the CPU and run several times slower.
- Candidates (only those installed are tried): qwen2.5:3b, llama3.2:3b, gemma2:2b, phi3.5.
  The final model is picked on validation by `scripts/tune_llm.py`, never on test.

## D14. Prompt design
- Two variants, both under version control in `prompts/`:
  `zeroshot_names` (just the 77 label names) and `zeroshot_descriptions` (name + one-line
  description). Comparing them on validation shows whether the descriptions earn their extra tokens.
- Descriptions were written from label names and a few training examples per intent; the test set
  was never read. Dataset quirks in label strings (`Refund_not_showing_up`, `reverted_card_payment?`)
  are kept so outputs map 1:1 to gold labels.
- Labels are listed alphabetically so related intents (card_*, top_up_*) are adjacent.
- The long label list is the system message and comes first; the customer message comes last.
  The prefix is identical for every query, so Ollama can reuse its computation for it, which
  cuts latency.
- No "think step by step": reasoning text would multiply output tokens and latency, and the point
  of this method is the cheapest possible LLM routing. (A reasoning variant could be a later ablation.)

## D15. Constrained output, strict parsing, invalid outputs counted
- Ollama's `format` gets a JSON schema whose `intent` field is an enum of the 77 labels, so the model
  can only emit a valid label (needs Ollama >= 0.5; checked at start-up).
- The parser still accepts only an exact label or a trivially normalised one (case, spaces,
  edge punctuation). No fuzzy guessing: a guessed label could hide model errors.
- Unparseable answers get class -1: scored as wrong, and the invalid rate is reported separately.

## D16. Decoding options and caching
- temperature 0 and a fixed seed: answers are repeatable.
- `num_ctx` 4096 set explicitly: Ollama's default context can be smaller than our prompt and
  truncates it silently. A guard raises an error if a prompt comes close to the limit.
- Every response is appended to `llm_cache/<model>.jsonl`, keyed by a hash of model + messages +
  schema + options. The cache is committed, so the benchmark replays without calling the model.
- Latency for the LLM is the wall time recorded when the call was made, minus Ollama's reported
  model-load time (a one-off start-up cost, like the warm-up calls skipped for other models).

## D17. The README's numbers are generated, never typed
- `scripts/make_report.py` rewrites the result tables (and the short observation sentences)
  between `<!-- NAME:START/END -->` markers from `results/*.csv`. Unmeasured cells stay `[placeholder]`.
- Each latency row records the machine it was measured on, and the README shows it, because
  methods may be run on different machines (no GPU or Ollama in the development container).

## D18. Error review protocol
- 100 errors per method, sampled with the global seed, written with the nearest training message
  next to each, to make label noise easy to spot.
- Two tags only: `label_noise` (gold label wrong, or the message honestly fits both intents) and
  `model_error`. Two tags keep the hand-review fast and the result easy to state.
- The review file is never overwritten once created, so hand-tagging cannot be lost.
