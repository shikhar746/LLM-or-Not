"""Load Banking77, verify it, and build the splits every experiment uses.

Splits:
  train      - official train minus the validation rows (used to fit models)
  val        - stratified slice of official train (used for ALL tuning: hyperparameters, prompts)
  test       - official test set, untouched until a configuration is final
  llm_sample - fixed, seeded, stratified subset of test, scored by every method so the LLM
               comparison is like-for-like
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

# The Hugging Face dataset PolyAI/banking77 is built from these files. We pin the exact git commit
# and check SHA-256 hashes, so a silent upstream change cannot alter our results.
_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
_BASE = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{_COMMIT}/banking_data"
FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
    "categories.json": "53261da888122daf2d120d925458631d9619e15d82e56052e7a42e535ce32b63",
}


def load_config() -> dict:
    with open(ROOT / "configs" / "data.yaml") as f:
        return yaml.safe_load(f)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(raw_dir: Path = RAW_DIR) -> None:
    """Download the raw files once and check their hashes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = raw_dir / name
        if not path.exists():
            resp = requests.get(f"{_BASE}/{name}", timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"{name}: hash mismatch (got {actual}). Delete it and re-download.")


def load_labels(raw_dir: Path = RAW_DIR) -> list[str]:
    """The 77 intent names, in the dataset's official order (index = class id)."""
    download(raw_dir)
    return json.loads((raw_dir / "categories.json").read_text())


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official train and test sets as DataFrames with columns: text, label (str)."""
    download(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv").rename(columns={"category": "label"})
    test = pd.read_csv(raw_dir / "test.csv").rename(columns={"category": "label"})
    for df in (train, test):
        df["text"] = df["text"].str.strip()  # a few raw texts start with a stray newline
    return train, test


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    llm_sample: pd.DataFrame
    labels: list[str]


def make_splits(config: dict | None = None, raw_dir: Path = RAW_DIR) -> Splits:
    cfg = config or load_config()
    labels = load_labels(raw_dir)
    train_full, test = load_raw(raw_dir)

    # Keep the original row number as an id, so any item can be traced back to the raw file.
    train_full = train_full.reset_index(names="row_id")
    test = test.reset_index(names="row_id")

    # Stratified = every intent keeps the same share in train and val as in the full train set.
    train, val = train_test_split(
        train_full,
        test_size=cfg["val_fraction"],
        stratify=train_full["label"],
        random_state=cfg["seed"],
    )
    _, llm_sample = train_test_split(
        test,
        test_size=cfg["llm_sample_size"],
        stratify=test["label"],
        random_state=cfg["seed"],
    )
    return Splits(
        train=train.reset_index(drop=True),
        val=val.reset_index(drop=True),
        test=test,
        llm_sample=llm_sample.sort_values("row_id").reset_index(drop=True),
        labels=labels,
    )


_PUNCT = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Used only for duplicate detection."""
    return re.sub(r"\s+", " ", text.lower().translate(_PUNCT)).strip()
