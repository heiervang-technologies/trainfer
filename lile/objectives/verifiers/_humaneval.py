"""HumanEval verifier — runs ALL unit tests via evalplus sandbox.

Claims rule: prompts that carry both a ``def`` signature and a ``>>>``
doctest-style sentinel — the canonical HumanEval docstring format.

Verify rule: parses a Python code block from the candidate, runs the
problem's base_input and plus_input tests via evalplus's trusted_exec
sandbox with a wall-clock timeout. Returns True only when ALL tests pass.

The problem set is loaded lazily from evalplus's cached HumanEval+ data.
Expected outputs are computed once (canonical solution run against inputs)
and cached by evalplus internally.

LOC budget: 200 lines including tests. If this file exceeds 200 LOC,
ABORT and pivot to _code.py "Expected output" format per campaign C-001
constraint 1.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import multiprocessing as _mp
import re

from . import register
# evalplus's untrusted_check uses multiprocessing.Process() without specifying
# a start method. Python 3.14 defaults to "spawn" on Linux, but evalplus's
# sandbox relies on fork semantics so the test runner inherits imported
# modules + sys.path. We scope the fork requirement to a sacrificial child
# process (via ProcessPoolExecutor with mp_context="fork") so the mutation
# doesn't propagate to the daemon's process-wide state.
_MP_CTX = _mp.get_context("fork")


_HUMANEVAL_DATA: dict | None = None
_HUMANEVAL_EXPECTED: dict | None = None
_TASK_BY_PROMPT: dict[str, str] = {}  # sha256(prompt) -> task_id


def _load_data():
    """Lazy-load HumanEval problems + expected outputs via evalplus.

    evalplus caches expected outputs on disk, so this is computed once
    per process.
    """
    global _HUMANEVAL_DATA, _HUMANEVAL_EXPECTED, _TASK_BY_PROMPT
    if _HUMANEVAL_DATA is not None:
        return
    from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
    from evalplus.evaluate import get_groundtruth

    _HUMANEVAL_DATA = get_human_eval_plus()
    _HUMANEVAL_EXPECTED = get_groundtruth(
        _HUMANEVAL_DATA,
        get_human_eval_plus_hash(),
        [],
    )
    # Build prompt -> task_id lookup
    for tid, pb in _HUMANEVAL_DATA.items():
        h = hashlib.sha256(pb["prompt"].encode("utf-8")).hexdigest()
        _TASK_BY_PROMPT[h] = tid


def _extract_code(candidate: str) -> str | None:
    """Extract the first Python fenced code block, or a bare def block."""
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", candidate, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # Bare def statement (no fence): grab from def to end.
    m = re.search(r"(def\s+\w+\(.*[\s\S]*)", candidate)
    if m:
        return m.group(1).strip()

    return None


def _match_task(prompt: str) -> str | None:
    """Look up task_id by prompt hash."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return _TASK_BY_PROMPT.get(h)


def claims(prompt: str) -> bool:
    """True when prompt looks like a HumanEval problem."""
    if not prompt or not isinstance(prompt, str):
        return False
    return bool(re.search(r"def \w+\(", prompt)) and ">>>" in prompt


def _run_check_correctness(problem: dict, code: str, expected: dict) -> dict:
    """Run check_correctness in a forked child.

    The child sets its own fork start method so evalplus's nested
    Process() calls inherit fork semantics. Does NOT touch the parent's
    multiprocessing state.
    """
    import multiprocessing
    # force=True is safe here: this runs in a freshly-forked child whose
    # multiprocessing state is independent of the parent. evalplus's nested
    # Process() calls need fork as the start method to inherit our imported
    # modules + sys.path.
    multiprocessing.set_start_method("fork", force=True)
    from evalplus.evaluate import check_correctness
    return check_correctness(
        dataset="humaneval",
        completion_id=0,
        problem=problem,
        solution=code,
        expected_output=expected,
        base_only=False,
        fast_check=False,
        min_time_limit=1.0,
        gt_time_limit_factor=4.0,
    )


@register("humaneval")
def verify(prompt: str, candidate: str) -> bool | None:
    if not claims(prompt):
        return None
    _load_data()

    code = _extract_code(candidate)
    if not code:
        return False

    task_id = _match_task(prompt)
    if task_id is None:
        return None  # Shouldn't happen in a well-configured loop

    problem = _HUMANEVAL_DATA[task_id]
    expected = _HUMANEVAL_EXPECTED[task_id]

    # Fork a sacrificial child for evalplus so its nested Process() calls
    # inherit fork semantics without mutating the parent's start method.
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=1, mp_context=_MP_CTX,
        ) as ex:
            future = ex.submit(_run_check_correctness, problem, code, expected)
            result = future.result(timeout=30)
    except concurrent.futures.TimeoutError:
        return False
    except Exception:
        return False

    base_status, _ = result.get("base", ("error", []))
    plus_status, _ = result.get("plus", ("error", []))
    return base_status == "pass" and plus_status == "pass"


verify.claims = claims  # type: ignore[attr-defined]


# ================================================================
# Tests (inline, <20 LOC)
def _test_claims():
    g = 'def add(a, b):\n    """Return a + b.\n    >>> add(1, 2)\n    3\n    """'
    assert claims(g) and not claims("capital?") and not claims("")
    print("  [OK] claims")

def _test_extract():
    assert _extract_code('```python\ndef f():\n    return 1\n```') == "def f():\n    return 1"
    assert _extract_code("no code") is None
    print("  [OK] extract")

def _test_verify():
    _load_data()
    tid = "HumanEval/0"
    if tid not in _HUMANEVAL_DATA:
        print("  [SKIP] verify: HumanEval/0 not cached")
        return
    pb = _HUMANEVAL_DATA[tid]
    assert verify(pb["prompt"], pb["canonical_solution"]) is True
    assert verify(pb["prompt"], "def has_close_elements(n, t):\n    return True\n") is False
    print("  [OK] verify")


if __name__ == "__main__":
    _test_claims()
    _test_extract()
    _test_verify()
    print("All verifier tests passed.")
