"""Method 5: zero-shot LLM classification through a local Ollama model.

Zero-shot = the model sees no labelled examples, only the list of intent names (and, in one prompt
variant, a one-line description of each). fit() therefore learns nothing; it exists only so this
model fits the shared interface.

Output is constrained with a JSON schema whose "intent" field is an enum of the 77 labels. Ollama
turns the schema into a grammar, so at every step the model may only produce tokens that keep the
output valid. We still parse defensively and count anything unusable as invalid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from llmornot.labels import INVALID, build_system_prompt, parse_label
from llmornot.llm_client import CachedChat, OllamaClient
from llmornot.models.base import IntentClassifier


def safe_name(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


class LLMZeroShot(IntentClassifier):
    name = "llm_zeroshot"

    def __init__(
        self,
        labels: list[str],
        model: str,
        prompt: str,
        options: dict,
        cache_dir: str | Path,
        client: OllamaClient | None = None,
        keep_alive: str = "30m",
    ):
        super().__init__(len(labels))
        self.labels = labels
        self.model = model
        self.prompt = prompt
        self.options = options
        self.keep_alive = keep_alive
        self.system_prompt = build_system_prompt(prompt, labels)
        self.schema = {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": sorted(labels, key=str.lower)}},
            "required": ["intent"],
        }
        self.chat = CachedChat(client, Path(cache_dir) / f"{safe_name(model)}.jsonl")
        self.last_responses: list[dict] = []

    def fit(self, texts: Sequence[str], y: np.ndarray) -> "LLMZeroShot":
        return self  # zero-shot: nothing to learn

    def request(self, text: str) -> dict:
        # The long system prompt is identical for every query and comes first; only the short
        # customer message changes. Ollama can then reuse its computation for the shared prefix.
        return {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}],
            "format": self.schema,
            "options": self.options,
            "keep_alive": self.keep_alive,
        }

    def classify(self, text: str) -> dict:
        """One query -> cached response plus the parsed class id."""
        resp = self.chat(self.request(text), meta={"prompt": self.prompt, "text": text})
        ctx = self.options.get("num_ctx")
        if ctx and resp["prompt_tokens"] >= ctx - self.options.get("num_predict", 0):
            raise RuntimeError(f"Prompt used {resp['prompt_tokens']} tokens; num_ctx={ctx} may have truncated it.")
        class_id, how = parse_label(resp["content"], self.labels)
        return {**resp, "class_id": class_id, "parse": how}

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """One-hot rows (the LLM gives a single answer, not probabilities); all-zero row if invalid."""
        self.last_responses = [self.classify(t) for t in texts]
        proba = np.zeros((len(texts), self.n_classes))
        for row, r in enumerate(self.last_responses):
            if r["class_id"] != INVALID:
                proba[row, r["class_id"]] = 1.0
        return proba

    def params(self) -> dict:
        return {"model": self.model, "prompt": self.prompt, **self.options}
