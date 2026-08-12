"""C-5 regression: cross_entropy path must match the old log_softmax path.

The C-5 fix replaces:

    logprobs = F.log_softmax(shift_logits.float(), dim=-1)      # (B, T, V) float32
    token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(labels == -100, 0.0)

with:

    neg_logprobs = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100,
                                   reduction='none')
    token_logprobs = -neg_logprobs.reshape(B, T)

Both compute the same thing — negative log-likelihood of the target token at
each position — but cross_entropy fuses the softmax + gather internally and
never materializes the full (B, T, V) softmax output, saving ~592 MB per call.

This test proves numerical equivalence against random logits/labels at various
shapes, including edge cases (all-masked, single-token, large vocab).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.cpu_only


def _old_sequence_logprob_core(
    shift_logits: torch.Tensor, shift_labels: torch.Tensor
) -> torch.Tensor:
    """Original log_softmax + gather implementation (pre-C-5)."""
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logprobs = F.log_softmax(shift_logits.float(), dim=-1)
    token_logprobs = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logprobs = token_logprobs.masked_fill(~mask, 0.0)
    return token_logprobs.sum(dim=-1)


def _new_sequence_logprob_core(
    shift_logits: torch.Tensor, shift_labels: torch.Tensor
) -> torch.Tensor:
    """New cross_entropy implementation (C-5)."""
    B, T, V = shift_logits.shape
    flat_logits = shift_logits.reshape(B * T, V).float()
    flat_labels = shift_labels.reshape(B * T)
    neg_logprobs = F.cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction="none"
    )
    token_logprobs = -neg_logprobs.reshape(B, T)
    return token_logprobs.sum(dim=-1)


# ------------------------------------------------------------------ core equiv


@pytest.mark.parametrize(
    "B,T,V",
    [
        (1, 4, 10),  # tiny
        (2, 8, 100),  # small batch
        (1, 1, 50),  # single token
        (3, 16, 1000),  # medium vocab
        (2, 32, 32000),  # realistic vocab
        (1, 64, 152064),  # Qwen3 vocab — the real target
    ],
)
def test_numerical_equivalence(B: int, T: int, V: int):
    """Old and new implementations must agree to within float32 tolerance."""
    torch.manual_seed(42)
    logits = torch.randn(B, T, V)
    # Random labels with ~30% masked to -100.
    labels = torch.randint(0, V, (B, T))
    mask_positions = torch.rand(B, T) < 0.3
    labels[mask_positions] = -100

    old = _old_sequence_logprob_core(logits, labels)
    new = _new_sequence_logprob_core(logits, labels)

    torch.testing.assert_close(old, new, atol=1e-5, rtol=1e-5)


def test_all_masked():
    """When every label is -100, both paths should return 0."""
    logits = torch.randn(2, 8, 100)
    labels = torch.full((2, 8), -100, dtype=torch.long)

    old = _old_sequence_logprob_core(logits, labels)
    new = _new_sequence_logprob_core(logits, labels)

    assert (old == 0.0).all()
    assert (new == 0.0).all()


def test_no_masked():
    """When no labels are masked, both paths should still agree."""
    torch.manual_seed(7)
    logits = torch.randn(2, 8, 100)
    labels = torch.randint(0, 100, (2, 8))

    old = _old_sequence_logprob_core(logits, labels)
    new = _new_sequence_logprob_core(logits, labels)

    torch.testing.assert_close(old, new, atol=1e-5, rtol=1e-5)


def test_single_unmasked_token():
    """Edge case: only one real label per sample, rest masked."""
    logits = torch.randn(2, 8, 50)
    labels = torch.full((2, 8), -100, dtype=torch.long)
    labels[0, 3] = 7
    labels[1, 5] = 12

    old = _old_sequence_logprob_core(logits, labels)
    new = _new_sequence_logprob_core(logits, labels)

    torch.testing.assert_close(old, new, atol=1e-5, rtol=1e-5)
    # Both should be negative (log-probs are always <= 0).
    assert (old <= 0).all()


def test_output_is_negative():
    """Sum of log-probs must be non-positive for all non-trivial cases."""
    torch.manual_seed(99)
    logits = torch.randn(3, 16, 1000)
    labels = torch.randint(0, 1000, (3, 16))

    result = _new_sequence_logprob_core(logits, labels)
    assert (result <= 0).all(), f"log-prob sum should be <= 0, got {result}"


def test_gradient_flows():
    """The new implementation must allow gradients to flow through logits."""
    logits = torch.randn(2, 8, 100, requires_grad=True)
    labels = torch.randint(0, 100, (2, 8))
    labels[:, :3] = -100  # mask prompt

    result = _new_sequence_logprob_core(logits, labels)
    result.sum().backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    # Gradients should be zero at fully-masked positions (positions where
    # ALL samples have -100 labels don't contribute to loss).
    # But non-zero at positions with real labels.
    assert logits.grad.abs().sum() > 0


def test_gradient_equivalence():
    """Gradients through old and new paths must match."""
    torch.manual_seed(42)
    logits_old = torch.randn(2, 8, 100, requires_grad=True)
    logits_new = logits_old.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 100, (2, 8))
    labels[:, :3] = -100

    old = _old_sequence_logprob_core(logits_old, labels)
    new = _new_sequence_logprob_core(logits_new, labels)

    old.sum().backward()
    new.sum().backward()

    torch.testing.assert_close(logits_old.grad, logits_new.grad, atol=1e-5, rtol=1e-5)


# ------------------------------------------------------------------ bf16 dtype


def test_bf16_equivalence():
    """Both paths should agree when logits are bf16 (cast to float32 internally)."""
    torch.manual_seed(42)
    logits = torch.randn(2, 8, 100).bfloat16()
    labels = torch.randint(0, 100, (2, 8))
    labels[:, :3] = -100

    old = _old_sequence_logprob_core(logits, labels)
    new = _new_sequence_logprob_core(logits, labels)

    # bf16 has less precision, so wider tolerance.
    torch.testing.assert_close(old, new, atol=1e-3, rtol=1e-3)
