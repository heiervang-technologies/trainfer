"""Unit tests for ``lile.state._install_matmul_lora_patch`` against a stub
``unsloth`` package.

The patch installer is the single load-bearing coupling between lile and the
ht-unsloth fork; ``lile/state.py:55-77``'s signature guard is what stops a
silent residual-application regression when upstream Unsloth shuffles its
kernel API. The live e2e proof lives in ``test_residual_live_path`` and
``test_family_compat`` — but both require a real Qwen3 model on a real GPU.

This test exercises the install path with a synthetic ``unsloth.kernels.utils``
module + a tiny linear stand-in for the LoRA matmul. No unsloth, no transformers,
no GPU.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytestmark = pytest.mark.cpu_only


def _make_stub_matmul_lora():
    """Return ``matmul_lora(X, W, W_quant, A, B, s, out=None)`` — the exact
    signature lile expects. Computes a deterministic ``X @ W.T`` and ignores
    the LoRA branch; the test setup attaches ``W._residual_delta`` to assert
    the patch's residual-addition contract.
    """

    def matmul_lora(X, W, W_quant, A, B, s, out=None):
        return X @ W.T

    return matmul_lora


def _install_stub_unsloth(monkeypatch, matmul_lora_impl):
    """Build a synthetic ``unsloth.kernels.utils`` with the given ``matmul_lora``
    and register it in ``sys.modules`` so ``import unsloth.kernels.utils`` works.

    Also installs a sibling re-export at ``unsloth.kernels.fast_lora`` so the
    cross-module rebind sweep has something to find — that's the load-bearing
    multi-binding contract from state.py:106-118.
    """
    unsloth = types.ModuleType("unsloth")
    kernels = types.ModuleType("unsloth.kernels")
    utils = types.ModuleType("unsloth.kernels.utils")
    fast_lora = types.ModuleType("unsloth.kernels.fast_lora")

    utils.matmul_lora = matmul_lora_impl
    # Simulate the "re-exported into another submodule" case (fast_lora pre-binds
    # matmul_lora at import time, so a naive patch of utils alone misses it).
    fast_lora.matmul_lora = matmul_lora_impl

    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.kernels", kernels)
    monkeypatch.setitem(sys.modules, "unsloth.kernels.utils", utils)
    monkeypatch.setitem(sys.modules, "unsloth.kernels.fast_lora", fast_lora)
    return unsloth, utils, fast_lora


def _fresh_install_state():
    """``lile.state`` caches its install via ``functools.cache``. Clear the
    cache so tests can drive the install function independently of any
    earlier-test side effects."""
    from lile import state as state_mod

    state_mod.install.cache_clear()
    return state_mod


def test_signature_match_installs_wrapper(monkeypatch):
    """A matmul_lora with the expected (X, W, W_quant, A, B, s, out) signature
    must be wrapped — and the wrapper must be tagged so a second install is
    idempotent."""
    original = _make_stub_matmul_lora()
    _, utils, fast_lora = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()

    state_mod._install_matmul_lora_patch()

    assert utils.matmul_lora is not original, "wrapper must replace the binding"
    assert getattr(utils.matmul_lora, "_lile_patched", False) is True
    assert utils.matmul_lora._lile_original is original
    # Cross-module re-export must also be rewritten.
    assert fast_lora.matmul_lora is utils.matmul_lora, (
        "fast_lora's pre-bound reference must be rebound too"
    )


def test_signature_mismatch_skips_install(monkeypatch, caplog):
    """A matmul_lora with a different signature must be left alone (no patch,
    no crash) — fast-path residual silently disabled, peft hook backstop still
    runs. We assert the warning fires so observability is preserved."""

    def odd_signature(inputs, weight):  # wrong arity + names
        return inputs @ weight.T

    _, utils, _ = _install_stub_unsloth(monkeypatch, odd_signature)
    state_mod = _fresh_install_state()

    with caplog.at_level("WARNING", logger="lile.state"):
        state_mod._install_matmul_lora_patch()

    assert utils.matmul_lora is odd_signature, (
        "drift detected — original must be preserved untouched"
    )
    assert any(
        "matmul_lora signature changed" in rec.message for rec in caplog.records
    ), "drift must surface as a WARNING, not silent"


def test_install_is_idempotent(monkeypatch):
    """Calling install() twice must produce exactly one wrapper layer —
    a second wrap would lose ``_lile_original`` and keep stacking forwards."""
    original = _make_stub_matmul_lora()
    _, utils, _ = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()

    state_mod._install_matmul_lora_patch()
    first = utils.matmul_lora
    state_mod._install_matmul_lora_patch()
    second = utils.matmul_lora

    assert first is second
    assert second._lile_original is original


def test_residual_is_added_when_attribute_present(monkeypatch):
    """End-to-end behavioral check: with the wrapper installed and a
    ``_residual_delta`` attribute set on the weight tensor, the wrapped call
    must return ``original(...) + X @ delta.T``.
    """
    original = _make_stub_matmul_lora()
    _, utils, _ = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()
    state_mod._install_matmul_lora_patch()

    torch.manual_seed(0)
    X = torch.randn(2, 4)
    W = torch.randn(3, 4)
    delta = torch.randn(3, 4)
    W._residual_delta = delta

    out = utils.matmul_lora(X, W, None, None, None, 1.0)
    expected = X @ W.T + X @ delta.T
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_residual_is_skipped_when_attribute_absent(monkeypatch):
    """No ``_residual_delta`` → behavior must be byte-identical to the
    underlying matmul_lora. The whole patch is a no-op on weights that
    never had a merge applied."""
    original = _make_stub_matmul_lora()
    _, utils, _ = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()
    state_mod._install_matmul_lora_patch()

    X = torch.randn(2, 4)
    W = torch.randn(3, 4)
    # No _residual_delta set.

    out = utils.matmul_lora(X, W, None, None, None, 1.0)
    torch.testing.assert_close(out, X @ W.T, rtol=0, atol=0)


def test_residual_is_skipped_when_attribute_is_none(monkeypatch):
    """Explicit ``W._residual_delta = None`` must be treated the same as
    "attribute absent" — no residual addition. ``getattr(W, ..., None)`` in
    state.py:87 covers both cases by design, so this is a regression guard
    against someone later switching to ``hasattr``."""
    original = _make_stub_matmul_lora()
    _, utils, _ = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()
    state_mod._install_matmul_lora_patch()

    X = torch.randn(2, 4)
    W = torch.randn(3, 4)
    W._residual_delta = None

    out = utils.matmul_lora(X, W, None, None, None, 1.0)
    torch.testing.assert_close(out, X @ W.T, rtol=0, atol=0)


def test_residual_passes_out_kwarg_through(monkeypatch):
    """Real Unsloth callers pass ``out=`` to reuse a pre-allocated buffer.
    The wrapper must forward it to the original kernel; the result + residual
    must equal the same math as the no-``out`` path on a fresh buffer."""
    captured = {}

    def kernel(X, W, W_quant, A, B, s, out=None):
        captured["out_is"] = out
        prod = X @ W.T
        if out is not None:
            out.copy_(prod)
            return out
        return prod

    _, utils, _ = _install_stub_unsloth(monkeypatch, kernel)
    state_mod = _fresh_install_state()
    state_mod._install_matmul_lora_patch()

    X = torch.randn(2, 4)
    W = torch.randn(3, 4)
    delta = torch.randn(3, 4)
    W._residual_delta = delta
    buf = torch.empty(2, 3)

    out = utils.matmul_lora(X, W, None, None, None, 1.0, out=buf)
    assert captured["out_is"] is buf, "wrapper must forward `out=` kwarg unchanged"
    expected = X @ W.T + X @ delta.T
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def test_residual_dtype_device_cast(monkeypatch):
    """The fast path skips a per-forward .to() syscall when the residual's
    dtype + device already match the kernel output. When they don't, the
    cast happens in-flight and the result is still correct."""

    def original(X, W, W_quant, A, B, s, out=None):
        return (X @ W.T).to(torch.float32)  # forces an output dtype != delta

    _, utils, _ = _install_stub_unsloth(monkeypatch, original)
    state_mod = _fresh_install_state()
    state_mod._install_matmul_lora_patch()

    X = torch.randn(2, 4, dtype=torch.float32)
    W = torch.randn(3, 4, dtype=torch.float32)
    delta = torch.randn(3, 4, dtype=torch.bfloat16)
    W._residual_delta = delta

    out = utils.matmul_lora(X, W, None, None, None, 1.0)
    assert out.dtype == torch.float32, "output dtype must follow original kernel"
    expected = (X @ W.T) + (X @ delta.to(torch.float32).T)
    torch.testing.assert_close(out, expected, rtol=5e-3, atol=5e-3)


def test_whitelist_excludes_this_file():
    """This file requires torch (imported at module scope) — it MUST NOT be
    in ``_TORCHLESS_OK``, otherwise pytest collection on a torchless runner
    will crash. The conftest's torch-missing branch will exclude it automatically
    when torch is absent; on a torchful runner (cpu_only bucket) it runs.

    Regression guard: if a future contributor adds this filename to the whitelist
    by mistake, the cpu_only suite on a true torchless install would crash at
    `import torch` during collection.
    """
    from pathlib import Path
    import lile.tests.conftest as conftest_mod

    assert Path(__file__).name not in conftest_mod._TORCHLESS_OK, (
        "test_matmul_lora_patch.py imports torch at module scope and MUST NOT be "
        "in _TORCHLESS_OK. Remove it from the whitelist in lile/tests/conftest.py."
    )
