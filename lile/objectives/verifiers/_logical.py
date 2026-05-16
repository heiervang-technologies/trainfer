"""Verifier for the pinned logical-task corpus at
``lile/teach/logical/tasks_v0.json``.

Each task carries its own answer-extraction regex + comparison mode, so
this verifier is a generic dispatcher: it hashes the prompt to find the
task, extracts the candidate's final answer per the task's ``extract``
pattern, then compares per the task's ``compare`` mode.

Compare modes
-------------
- ``exact``    — case-insensitive whitespace-stripped string equality.
- ``numeric``  — int / float comparison after stripping commas and units.
- ``set``      — split on commas; compare as a multiset of stripped tokens.
- ``bool``     — normalize to {true, false}; accepts {1, true, yes, y} and
                 {0, false, no, n} on each side.
- ``regex``    — ``expected`` is itself a regex; ``re.fullmatch`` on the
                 extracted answer.
"""
from __future__ import annotations

import hashlib
import logging
import re

from . import register

log = logging.getLogger(__name__)

_TASKS: dict[str, dict] | None = None  # task_id -> task dict
_BY_PROMPT: dict[str, str] = {}        # sha256(prompt) -> task_id

_DEFAULT_EXTRACT = re.compile(r"(?is)Answer\s*[:=]\s*([^\n]+)")
_TRUTHY = {"1", "true", "yes", "y", "t"}
_FALSY = {"0", "false", "no", "n", "f"}


def _load() -> None:
    """Lazy-load the pinned task set; idempotent."""
    global _TASKS
    if _TASKS is not None:
        return
    from lile.teach.logical import load_tasks
    _TASKS = {}
    for t in load_tasks():
        _TASKS[t["task_id"]] = t
        _BY_PROMPT[_hash(t["prompt"])] = t["task_id"]


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _extract(candidate: str, pattern: str | None) -> str | None:
    """Pull the final answer from ``candidate`` using the task's regex.

    Default pattern matches the trailing ``Answer:`` line we instruct the
    model to produce in every task prompt. Tasks can override by setting
    their own ``extract`` field if they need a stricter shape.
    """
    rx = re.compile(pattern, re.IGNORECASE | re.DOTALL) if pattern else _DEFAULT_EXTRACT
    matches = list(rx.finditer(candidate or ""))
    if not matches:
        return None
    # Last match wins — if the model meanders before answering, we take the
    # final ``Answer:`` line, not the first hypothetical.
    return matches[-1].group(1).strip()


def _norm_bool(s: str) -> str | None:
    s = s.strip().lower()
    if s in _TRUTHY:
        return "true"
    if s in _FALSY:
        return "false"
    return None


def _compare(got: str, expected: str, mode: str) -> bool:
    g = (got or "").strip()
    e = (expected or "").strip()
    if mode == "exact":
        return g.lower() == e.lower()
    if mode == "numeric":
        # Strip commas + trailing punctuation; tolerate '.' and minus sign.
        def _num(x: str) -> float | None:
            cleaned = re.sub(r"[^0-9.\-]", "", x)
            if not cleaned or cleaned in {".", "-", "-."}:
                return None
            try:
                return float(cleaned)
            except ValueError:
                return None
        ng = _num(g)
        ne = _num(e)
        return ng is not None and ne is not None and abs(ng - ne) < 1e-9
    if mode == "set":
        gs = {x.strip().lower() for x in re.split(r"[,\s]+", g) if x.strip()}
        es = {x.strip().lower() for x in re.split(r"[,\s]+", e) if x.strip()}
        return gs == es
    if mode == "bool":
        ng = _norm_bool(g)
        ne = _norm_bool(e)
        return ng is not None and ne is not None and ng == ne
    if mode == "regex":
        try:
            return bool(re.fullmatch(e, g, re.IGNORECASE | re.DOTALL))
        except re.error as exc:
            log.warning("logical verifier: bad regex %r — %s", e, exc)
            return False
    log.warning("logical verifier: unknown compare mode %r", mode)
    return False


def claims(prompt: str) -> bool:
    """True iff ``prompt`` matches a task in the pinned corpus by hash.

    A stricter alternative to substring sniffing — only known-cataloged
    prompts get routed to this verifier. Updating ``tasks_v0.json`` adds
    new claims automatically.
    """
    if not prompt or not isinstance(prompt, str):
        return False
    _load()
    return _hash(prompt) in _BY_PROMPT


@register("logical")
def verify(prompt: str, candidate: str) -> bool | None:
    """Look up the task, extract the answer, compare.

    Returns ``None`` if the prompt isn't in the corpus (caller should
    treat as "skip / can't judge", per registry conventions).
    """
    if not claims(prompt):
        return None
    task_id = _BY_PROMPT[_hash(prompt)]
    task = _TASKS[task_id]
    got = _extract(candidate, task.get("extract"))
    if got is None:
        return False  # model produced no extractable answer
    return _compare(got, task["expected"], task.get("compare", "exact"))


verify.claims = claims  # type: ignore[attr-defined]
