"""CPU-only contract tests for the logical-task verifier
(``lile.objectives.verifiers._logical``) backed by the pinned corpus at
``lile/teach/logical/tasks_v0.json``.

Covers:

- Registry dispatch through ``verify("logical", ...)`` (smoke).
- Per-mode comparison (exact, numeric, set, bool).
- Unknown-prompt routing → ``None`` (verifier abstains).
- Empty / no-answer candidate → ``False``.
- Trailing-``Answer:`` regex picks the *last* answer line when the model
  meanders before committing.
- ``get_split`` produces a non-empty heldout per domain at 70/30.
"""

from __future__ import annotations

import pytest

from lile.objectives.verifiers import VERIFIERS, verify
from lile.teach.logical import get_split, load_tasks

pytestmark = pytest.mark.cpu_only


def _task_by_id(tid: str) -> dict:
    for t in load_tasks():
        if t["task_id"] == tid:
            return t
    raise KeyError(tid)


def test_registry_exposes_logical():
    assert "logical" in VERIFIERS, "logical verifier failed to register on import"


def test_exact_mode_matches_canonical_answer():
    t = _task_by_id("logical/prop_logic/0")  # expects "contradiction"
    assert verify("logical", t["prompt"], "Reasoning... Answer: contradiction") is True
    assert verify("logical", t["prompt"], "Answer: tautology") is False


def test_numeric_mode_tolerates_format():
    t = _task_by_id("logical/arith/0")  # expects 180
    assert verify("logical", t["prompt"], "Answer: 180") is True
    assert verify("logical", t["prompt"], "Answer: 180.0") is True
    assert verify("logical", t["prompt"], "trash before Answer: 180 units") is True
    assert verify("logical", t["prompt"], "Answer: 120") is False


def test_set_mode_order_insensitive():
    t = _task_by_id("logical/set_ops/0")  # expects "3,4"
    assert verify("logical", t["prompt"], "Answer: 4,3") is True
    assert verify("logical", t["prompt"], "Answer: 3, 4") is True
    assert verify("logical", t["prompt"], "Answer: 4") is False


def test_bool_mode_normalizes_synonyms():
    t = _task_by_id("logical/bool_eval/0")  # expects 1
    assert verify("logical", t["prompt"], "Answer: 1") is True
    assert verify("logical", t["prompt"], "Answer: true") is True
    assert verify("logical", t["prompt"], "Answer: yes") is True
    assert verify("logical", t["prompt"], "Answer: 0") is False
    assert verify("logical", t["prompt"], "Answer: false") is False


def test_unknown_prompt_routes_to_none():
    # An out-of-corpus prompt must produce None, NEVER False — None is the
    # registry's "I cannot judge this" signal, and graders downstream skip
    # those rather than counting them as failures.
    assert verify("logical", "What is the capital of France?", "Answer: Paris") is None


def test_empty_candidate_is_false_not_none():
    # Empty/no-Answer candidate is a real failure (model produced nothing),
    # distinct from "I can't judge this" — must be False so RLVR routes
    # toward an unlike/correction objective.
    t = _task_by_id("logical/prop_logic/0")
    assert verify("logical", t["prompt"], "") is False
    assert verify("logical", t["prompt"], "Just thinking out loud, no answer.") is False


def test_last_answer_wins_when_model_meanders():
    # Models that change their mind mid-stream may emit multiple "Answer:"
    # lines. The verifier takes the LAST one — that's the model's final
    # commitment, not a hypothetical it walked back.
    t = _task_by_id("logical/prop_logic/0")  # expects "contradiction"
    candidate = (
        "Hmm, Answer: tautology? No wait, P and not-P is impossible.\n"
        "Answer: contradiction"
    )
    assert verify("logical", t["prompt"], candidate) is True


def test_get_split_produces_nonempty_buckets_per_domain():
    train, heldout = get_split()
    train_domains = {t["domain"] for t in train}
    heldout_domains = {t["domain"] for t in heldout}
    # Every domain should appear in both buckets given >= 2 tasks per domain.
    assert train_domains == heldout_domains, (
        "split skipped a domain: "
        f"only-train={train_domains - heldout_domains} only-heldout={heldout_domains - train_domains}"
    )
    # Reproducibility — re-splitting yields byte-identical buckets.
    train2, heldout2 = get_split()
    assert [t["task_id"] for t in train] == [t["task_id"] for t in train2]
    assert [t["task_id"] for t in heldout] == [t["task_id"] for t in heldout2]


def test_train_held_disjoint():
    train, heldout = get_split()
    train_ids = {t["task_id"] for t in train}
    heldout_ids = {t["task_id"] for t in heldout}
    assert not (train_ids & heldout_ids), "train/heldout overlap"
