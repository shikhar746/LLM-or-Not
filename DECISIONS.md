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
