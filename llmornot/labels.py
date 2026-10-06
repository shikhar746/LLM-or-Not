"""Label descriptions, prompt construction and parsing of LLM answers back into class ids."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
INVALID = -1


def load_descriptions(path: Path = PROMPTS_DIR / "label_descriptions.yaml") -> dict[str, str]:
    return yaml.safe_load(path.read_text())


def build_system_prompt(prompt_name: str, labels: list[str], descriptions: dict[str, str] | None = None) -> str:
    """Fill a template from prompts/ with the label list.

    Labels are listed alphabetically so related intents sit together (card_*, top_up_*, transfer_*),
    which makes the near-identical ones easier to tell apart. The order is fixed for every query.
    """
    template = (PROMPTS_DIR / f"{prompt_name}.txt").read_text()
    ordered = sorted(labels, key=str.lower)
    if "descriptions" in prompt_name:
        descriptions = descriptions or load_descriptions()
        lines = [f"- {label}: {descriptions[label]}" for label in ordered]
    else:
        lines = [f"- {label}" for label in ordered]
    return template.replace("{label_block}", "\n".join(lines))


def _canonical(text: str) -> str:
    """Forgiving form for matching: lowercase, spaces/hyphens -> underscores, edge punctuation dropped."""
    text = re.sub(r"[\s\-]+", "_", text.strip().lower())
    return text.strip("\"'`.,:;!?()[]{}")


def parse_label(raw: str, labels: list[str]) -> tuple[int, str]:
    """Turn a raw LLM answer into (class_id, how).

    how = "exact"      the answer (or its "intent" JSON field) is exactly a label
          "normalised" it matches a label after lowercasing / fixing spaces and punctuation
          "invalid"    anything else -> class_id = -1, scored as wrong and counted separately

    We do NOT guess from partial matches ("card" -> card_arrival?). A wrong guess would hide
    model errors; counting them as invalid keeps the failure visible.
    """
    answer = raw.strip()
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, dict) and isinstance(parsed.get("intent"), str):
            answer = parsed["intent"].strip()
        elif isinstance(parsed, str):
            answer = parsed.strip()
    except json.JSONDecodeError:
        pass

    if answer in labels:
        return labels.index(answer), "exact"
    by_canonical = {_canonical(label): i for i, label in enumerate(labels)}
    key = _canonical(answer)
    if key in by_canonical:
        return by_canonical[key], "normalised"
    return INVALID, "invalid"
