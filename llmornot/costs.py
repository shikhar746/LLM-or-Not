"""Cost per 1,000 queries.

Two kinds of cost, kept separate so neither hides inside the other:
  API models   - you pay per token: (input tokens x input price) + (output tokens x output price)
  local models - you pay for the machine's time: compute-seconds x price of that machine per hour

Prices are always passed in (from configs/costs.yaml), never hard-coded here.
"""

from __future__ import annotations

PER_MILLION = 1_000_000
QUERIES = 1_000
SECONDS_PER_HOUR = 3_600


def api_cost_per_1k(
    mean_input_tokens: float,
    mean_output_tokens: float,
    price_input_per_m: float,
    price_output_per_m: float,
) -> float:
    """Cost of 1,000 queries given real mean token counts and prices per 1M tokens."""
    if min(mean_input_tokens, mean_output_tokens, price_input_per_m, price_output_per_m) < 0:
        raise ValueError("token counts and prices must be non-negative")
    per_query = (mean_input_tokens * price_input_per_m + mean_output_tokens * price_output_per_m) / PER_MILLION
    return per_query * QUERIES


def compute_seconds_per_1k(mean_seconds_per_query: float) -> float:
    """Machine-seconds needed for 1,000 queries, processed one at a time."""
    if mean_seconds_per_query < 0:
        raise ValueError("time must be non-negative")
    return mean_seconds_per_query * QUERIES


def local_cost_per_1k(mean_seconds_per_query: float, price_per_hour: float) -> float:
    """Cost of 1,000 queries on hardware that costs `price_per_hour` (e.g. a cloud VM's rate)."""
    if price_per_hour < 0:
        raise ValueError("price must be non-negative")
    return compute_seconds_per_1k(mean_seconds_per_query) / SECONDS_PER_HOUR * price_per_hour
