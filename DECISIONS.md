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
