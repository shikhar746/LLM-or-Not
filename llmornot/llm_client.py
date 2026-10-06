"""Talk to a local Ollama server, and cache every response to JSONL.

Why cache: an LLM run takes minutes to hours on a laptop. With the cache committed to git, anyone
can re-run the benchmark and get identical predictions, token counts and recorded latencies
without calling the model again. A cache entry is keyed by a hash of EVERYTHING that affects the
answer (model, messages, output schema, options), so changing the prompt can never reuse a stale answer.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

MIN_OLLAMA_VERSION = (0, 5, 0)  # first version that accepts a JSON schema in `format`


def cache_key(request: dict) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 300, session=None):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()  # tests pass a fake session

    def version(self) -> tuple[int, ...]:
        resp = self.session.get(f"{self.host}/api/version", timeout=10)
        resp.raise_for_status()
        return tuple(int(x) for x in re.findall(r"\d+", resp.json()["version"])[:3])

    def installed_models(self) -> list[str]:
        resp = self.session.get(f"{self.host}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json()["models"]]

    def check(self) -> None:
        """Fail early, with a clear message, if Ollama is missing or too old."""
        try:
            v = self.version()
        except requests.ConnectionError as e:
            raise RuntimeError(f"Ollama is not reachable at {self.host}. Start it with `ollama serve`.") from e
        if v < MIN_OLLAMA_VERSION:
            raise RuntimeError(f"Ollama {v} is too old; structured outputs need >= {MIN_OLLAMA_VERSION}.")

    def chat(self, request: dict) -> dict:
        """One non-streaming /api/chat call. Returns the fields we keep in the cache."""
        start = time.perf_counter()
        resp = self.session.post(f"{self.host}/api/chat", json={**request, "stream": False}, timeout=self.timeout)
        wall_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        body = resp.json()
        return {
            "content": body["message"]["content"],
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
            "wall_ms": wall_ms,
            # Ollama reports durations in nanoseconds. load = time to load weights into memory,
            # a one-off startup cost that we subtract when reporting per-query latency.
            "load_ms": body.get("load_duration", 0) / 1e6,
            "server_total_ms": body.get("total_duration", 0) / 1e6,
        }


class CachedChat:
    """Wraps a client: return the cached response if this exact request was seen, else call and append."""

    def __init__(self, client: OllamaClient | None, cache_path: Path):
        self.client = client
        self.cache_path = Path(cache_path)
        self.cache: dict[str, dict] = {}
        self.hits = self.misses = 0
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self.cache[entry["key"]] = entry

    def __call__(self, request: dict, meta: dict | None = None) -> dict:
        key = cache_key(request)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.client is None:
            raise RuntimeError("Response not in cache and no Ollama client given (offline replay mode).")
        self.misses += 1
        entry = {"key": key, "model": request["model"], **(meta or {}), **self.client.chat(request)}
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a") as f:  # append one line per call: a crash loses nothing
            f.write(json.dumps(entry) + "\n")
        self.cache[key] = entry
        return entry
