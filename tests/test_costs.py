import pytest

from llmornot.costs import api_cost_per_1k, compute_seconds_per_1k, local_cost_per_1k


def test_api_cost_hand_example():
    # 2,000 input tokens at 10 per 1M + 5 output tokens at 40 per 1M
    # per query = 0.02 + 0.0002 = 0.0202  ->  per 1k queries = 20.2
    assert api_cost_per_1k(2000, 5, 10, 40) == pytest.approx(20.2)


def test_api_cost_zero_when_free():
    assert api_cost_per_1k(2000, 5, 0, 0) == 0


def test_local_cost_hand_example():
    # 0.36 s per query -> 360 s per 1k = 0.1 h; at 50 per hour -> 5.0
    assert compute_seconds_per_1k(0.36) == pytest.approx(360)
    assert local_cost_per_1k(0.36, 50) == pytest.approx(5.0)


@pytest.mark.parametrize("args", [(-1, 5, 1, 1), (1, 5, -1, 1)])
def test_api_cost_rejects_negative(args):
    with pytest.raises(ValueError):
        api_cost_per_1k(*args)


def test_local_cost_rejects_negative():
    with pytest.raises(ValueError):
        local_cost_per_1k(1.0, -5)
