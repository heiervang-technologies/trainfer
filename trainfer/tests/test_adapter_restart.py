"""Repeated optimization must remain possible after consolidating an adapter."""

import pytest
import torch
from torch import nn

from trainfer.state import ModelState

pytestmark = pytest.mark.cpu_only


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})

    def forward(self, x):
        return self.lora_B["default"](self.lora_A["default"](x))


def test_reset_has_zero_output_and_can_learn_repeatedly():
    torch.manual_seed(11)
    model = Adapter()
    state = ModelState(model, None, "synthetic", 2, 2)
    x, target = torch.randn(8, 4), torch.randn(8, 3)
    for _ in range(3):
        state.reset_active_adapter()
        assert torch.count_nonzero(model(x)) == 0
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        before = torch.nn.functional.mse_loss(model(x), target).item()
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), target).backward()
        assert model.lora_B["default"].weight.grad.norm() > 0
        opt.step()
        assert torch.nn.functional.mse_loss(model(x), target).item() < before


def test_null_merge_does_not_create_dense_residual():
    model = nn.Sequential(Adapter())
    state = ModelState(model, None, "synthetic", 2, 2)
    state.reset_active_adapter()
    state.merge_active_into_residual()
    state.merge_active_into_residual()
    assert state.merged_deltas == {}


def test_reset_and_restore_invalidate_cast_decoder_adapters():
    state = ModelState(Adapter(), None, "synthetic", 2, 2)
    original = state.extract_active_adapter()
    for operation in (state.reset_active_adapter, lambda: state.load_active_adapter(original)):
        for p in state.model.parameters():
            p._fast_lora = p.detach().to(torch.bfloat16)
        operation()
        assert all(not hasattr(p, "_fast_lora") for p in state.model.parameters())
