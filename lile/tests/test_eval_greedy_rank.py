"""CPU-only tests for the /v1/eval/greedy_rank route wiring.

Full end-to-end (model forward + queue + metrics) is exercised by the
R-001 experiment runner against a live daemon — not here. These tests pin
the schema, controller surface, and metrics counter so a rename or
signature drift breaks CI immediately.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cpu_only


def test_eval_request_schema_accepts_prompt_and_response():
    """Verify the Pydantic model shape without importing the full server graph."""
    # Lazy import so a torchless runner skips this test if pydantic is absent.
    try:
        from lile.server import EvalGreedyRankRequest
    except ImportError:
        pytest.skip("pydantic not available")

    req = EvalGreedyRankRequest(prompt="What is X?", response="Y")
    assert req.prompt == "What is X?"
    assert req.response == "Y"


def test_controller_has_submit_eval_greedy_rank():
    """``Controller`` must expose the submit method with the right signature."""
    import inspect

    try:
        from lile.controller import Controller
    except ImportError:
        pytest.skip("heavy deps not available")

    sig = inspect.signature(Controller.submit_eval_greedy_rank)
    params = list(sig.parameters.keys())
    assert params == ["self", "prompt", "response"]
    # return type is dict[str, Any]; not pinning because drift is low-value


def test_eval_greedy_rank_metrics_counter_is_importable():
    """The counter added by the claim PR must be importable and incrementable."""
    try:
        from lile import metrics
    except ImportError:
        pytest.skip("heavy deps not available")

    # Counter should exist and have initial value 0.
    assert hasattr(metrics, "_EVAL_GREEDY_RANK")
    # record_eval_greedy_rank should bump it by 1.
    val_before = metrics._EVAL_GREEDY_RANK._value.get()
    metrics.record_eval_greedy_rank()
    val_after = metrics._EVAL_GREEDY_RANK._value.get()
    assert val_after == val_before + 1


def test_task_handler_has_eval_greedy_rank_kind():
    """Controller._handle_task must branch on ``kind == 'eval_greedy_rank'``."""
    import ast
    from pathlib import Path

    controller_py = Path(__file__).parent.parent / "controller.py"
    src = controller_py.read_text(encoding="utf-8")
    tree = ast.parse(src)
    kinds = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op in node.ops:
                if isinstance(op, ast.Eq):
                    for val in ast.walk(node.left):
                        if isinstance(val, ast.Name) and val.id == "kind":
                            for comp_val in ast.walk(node.comparators[0]):
                                if isinstance(comp_val, ast.Constant):
                                    kinds.append(comp_val.value)
    assert "eval_greedy_rank" in kinds, (
        "Controller._handle_task missing branch for eval_greedy_rank"
    )
