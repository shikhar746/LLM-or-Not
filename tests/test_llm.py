import json

import numpy as np
import pytest

from llmornot.llm_client import CachedChat, OllamaClient
from llmornot.models.llm_zeroshot import LLMZeroShot

LABELS = ["card_arrival", "exchange_rate", "top_up_failed"]
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 32}


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


class FakeOllama:
    """Answers like Ollama's /api/chat: card -> card_arrival, rate -> exchange_rate, else garbage."""

    def __init__(self, prompt_tokens=900):
        self.calls = 0
        self.prompt_tokens = prompt_tokens

    def get(self, url, timeout=None):
        return FakeResponse({"version": "0.6.2"} if url.endswith("/api/version") else {"models": [{"name": "fake:1b"}]})

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        text = json["messages"][-1]["content"]
        intent = "card_arrival" if "card" in text else "exchange_rate" if "rate" in text else "no idea"
        return FakeResponse({"message": {"content": '{"intent": "%s"}' % intent},
                             "prompt_eval_count": self.prompt_tokens, "eval_count": 8,
                             "load_duration": 2_000_000, "total_duration": 50_000_000})


def make_model(tmp_path, fake):
    client = OllamaClient(session=fake)
    return LLMZeroShot(LABELS, "fake:1b", "zeroshot_descriptions", OPTIONS, tmp_path, client=client)


def test_predictions_and_invalid_outputs(tmp_path):
    m = make_model(tmp_path, FakeOllama())
    assert m.predict(["where is my card", "what is the rate", "hello"]).tolist() == [0, 1, -1]
    assert [r["parse"] for r in m.last_responses] == ["exact", "exact", "invalid"]


def test_cache_avoids_repeat_calls_and_survives_reload(tmp_path):
    fake = FakeOllama()
    m = make_model(tmp_path, fake)
    m.predict(["where is my card", "what is the rate"])
    m.predict(["where is my card"])
    assert fake.calls == 2 and m.chat.hits == 1
    # offline replay: a new model with NO client still answers from the JSONL file
    offline = LLMZeroShot(LABELS, "fake:1b", "zeroshot_descriptions", OPTIONS, tmp_path, client=None)
    assert offline.predict(["where is my card"]).tolist() == [0]
    with pytest.raises(RuntimeError, match="not in cache"):
        offline.predict(["never seen before"])


def test_cache_key_changes_with_prompt(tmp_path):
    fake = FakeOllama()
    make_model(tmp_path, fake).predict(["my card"])
    LLMZeroShot(LABELS, "fake:1b", "zeroshot_names", OPTIONS, tmp_path, client=OllamaClient(session=fake)).predict(["my card"])
    assert fake.calls == 2  # different prompt -> different key -> new call
    lines = (tmp_path / "fake_1b.jsonl").read_text().splitlines()
    assert len(lines) == 2 and {json.loads(l)["prompt"] for l in lines} == {"zeroshot_descriptions", "zeroshot_names"}


def test_request_constrains_output_to_labels(tmp_path):
    req = make_model(tmp_path, FakeOllama()).request("hi")
    assert sorted(req["format"]["properties"]["intent"]["enum"]) == sorted(LABELS)
    assert req["messages"][0]["role"] == "system" and req["messages"][1]["content"] == "hi"


def test_truncation_guard(tmp_path):
    m = make_model(tmp_path, FakeOllama(prompt_tokens=4090))
    with pytest.raises(RuntimeError, match="truncated"):
        m.predict(["my card"])


def test_proba_is_one_hot_or_zero(tmp_path):
    proba = make_model(tmp_path, FakeOllama()).predict_proba(["my card", "hello"])
    assert np.array_equal(proba.sum(axis=1), [1.0, 0.0])
