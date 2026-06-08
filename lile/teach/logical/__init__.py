"""Logical-task loader for the RLVR loop.

Pinned dataset at ``tasks_v0.json`` (mirrors ``arc_agi_3/`` and
``humaneval/``). Each task is one verifiable reasoning prompt across one
of ten domains (prop_logic, syllogism, arith, sequence, set_ops,
bool_eval, kinship, parity, counting, ordering). See
``tasks_v0.json#/schema`` for the per-task fields.

The verifier at ``lile/objectives/verifiers/_logical.py`` looks tasks up
by prompt-hash, so the prompt string itself is the cache key — keep the
prompts byte-stable across edits if you want grades to remain comparable
across runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_DEFAULT_PATH = Path(__file__).parent / "tasks_v0.json"


def load_tasks(path: Path | None = None) -> list[dict[str, Any]]:
    """Return the flat list of task dicts from ``tasks_v0.json``."""
    src = Path(path) if path is not None else _DEFAULT_PATH
    raw = json.loads(src.read_text(encoding="utf-8"))
    tasks = raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw
    if not isinstance(tasks, list):
        raise ValueError(f"{src}: expected list of tasks, got {type(tasks).__name__}")
    return tasks


def get_split(
    tasks: list[dict[str, Any]] | None = None,
    train_ratio: float = 0.7,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/held-out split, stratified by domain.

    Within each domain, the first ``max(1, round(train_ratio * n))`` tasks
    (sorted by ``task_id``) go to train; the rest are held-out. This keeps
    the split stable across runs even when tasks are added, *provided* new
    tasks land at the end of their domain group. Both buckets are guaranteed
    non-empty when a domain has ≥ 2 tasks; small domains may degenerate to
    all-train or all-held-out — log the resulting per-domain counts.
    """
    if tasks is None:
        tasks = load_tasks()
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        by_domain.setdefault(t["domain"], []).append(t)

    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for domain in sorted(by_domain):
        items = sorted(by_domain[domain], key=lambda t: t["task_id"])
        n = len(items)
        # round() preserves the 70/30 intent. With n=3, round(2.1)=2 → 2 train, 1 held-out.
        # With n=2, round(1.4)=1 → 1 train, 1 held-out (still ≥ 1 each).
        n_train = max(1, min(n - 1, round(train_ratio * n))) if n >= 2 else n
        train.extend(items[:n_train])
        heldout.extend(items[n_train:])
    return train, heldout
