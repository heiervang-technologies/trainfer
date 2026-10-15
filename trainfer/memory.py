"""Execution-time memory admission; estimates are not allocator guarantees.

Check under the model lock, immediately before work. Other GPU processes and
allocator fragmentation can invalidate any free-memory observation. In
particular, gradient/optimizer bytes are only a lower bound for training;
activation workspace must be budgeted separately for the chosen workload.
"""

from __future__ import annotations

import math

import torch

from .errors import MemoryAdmissionError


def cuda_capacity(device):
    device = torch.device(device)
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    return {"device": str(device), "total_bytes": total, "free_bytes": free,
            "allocated_bytes": allocated, "reserved_bytes": reserved,
            "available_bytes": free + max(0, reserved - allocated)}


class MemoryAdmission:
    def __init__(self, reserve_bytes=256 * 1024**2, training_workspace_bytes=0, inference_workspace_bytes=0):
        if min(reserve_bytes, training_workspace_bytes, inference_workspace_bytes) < 0:
            raise ValueError("memory budgets must be nonnegative")
        self.reserve_bytes = reserve_bytes
        self.training_workspace_bytes = training_workspace_bytes
        self.inference_workspace_bytes = inference_workspace_bytes
        self.last_report = None

    def check(self, device, required_bytes, *, operation, basis):
        capacity = cuda_capacity(device)
        if capacity is None:
            return None
        report = {**capacity, "operation": operation, "basis": basis,
                  "required_bytes": required_bytes, "reserve_bytes": self.reserve_bytes,
                  "admitted": required_bytes + self.reserve_bytes <= capacity["available_bytes"]}
        self.last_report = report
        if not report["admitted"]:
            raise MemoryAdmissionError(
                f"{operation} needs an estimated {required_bytes} additional bytes plus "
                f"{self.reserve_bytes} reserve; {capacity['available_bytes']} bytes available ({basis})"
            )
        return report

    def training(self, model, optimizer):
        params = [p for group in optimizer.param_groups for p in group["params"]]
        gradient = sum(p.numel() * p.element_size() for p in params if p.grad is None)
        # Bound built-in state allocations from the actual selected optimizer,
        # including lazy initialization. Plugins must provide their own policy.
        opt_name = type(optimizer).__name__
        new_state = 0
        known = opt_name in {"AdamW", "AdamW8bit", "Lion8bit"}
        if opt_name in {"AdamW8bit", "Lion8bit"}:
            optimizer.check_overrides()
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, p in enumerate(group["params"]):
                if optimizer.state.get(p):
                    continue
                n = p.numel()
                if opt_name == "AdamW":
                    new_state += 2 * n * p.element_size() + 4
                elif opt_name in {"AdamW8bit", "Lion8bit"}:
                    moments = 1 if opt_name == "Lion8bit" else 2
                    config = optimizer.get_config(group_index, parameter_index, group)
                    if config["optim_bits"] == 8 and n >= config["min_8bit_size"]:
                        new_state += moments * (n + 4 * math.ceil(n / 256) + 256 * 4)
                    else:
                        new_state += moments * n * 4
                    if config.get("max_unorm", 0) > 0:
                        new_state += 4
        required = gradient + new_state + self.training_workspace_bytes
        basis = "gradient + optimizer state + configured activation workspace"
        if not known:
            basis = "gradient + configured workspace; optimizer state estimate unavailable"
        report = self.check(next(model.parameters()).device, required, operation="train", basis=basis)
        if report is not None:
            report.update(gradient_bytes=gradient, new_optimizer_bytes=new_state,
                          workspace_bytes=self.training_workspace_bytes,
                          activation_estimate_supplied=self.training_workspace_bytes > 0,
                          optimizer_estimate_available=known)
        return report

    def residual_merge(self, state):
        resident = 0
        largest = 0
        for name, module in state.model.named_modules():
            if not hasattr(module, "lora_A") or "default" not in module.lora_A:
                continue
            a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
            if name + ".weight" not in state.merged_deltas and (
                not bool(torch.count_nonzero(a)) or not bool(torch.count_nonzero(b))
            ):
                continue
            elements = a.shape[1] * b.shape[0]
            resident += elements * 2  # canonical bf16 residual copied to GPU
            largest = max(largest, elements)
        # Include all new residuals even if old ones are resident: replacing
        # hooks may temporarily keep both. Workspace covers dense matmul,
        # float conversion, scaling, and adding an existing residual.
        required = resident + largest * 24
        return self.check(next(state.model.parameters()).device, required,
                          operation="residual_merge", basis="bf16 residual copies + dense per-layer workspace")

    def inference(self, model, input_ids, max_new_tokens):
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        config = getattr(model, "config", None)
        if hasattr(config, "get_text_config"):
            config = config.get_text_config()
        layers = getattr(config, "num_hidden_layers", None)
        heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        head_dim = getattr(config, "head_dim", None)
        hidden = getattr(config, "hidden_size", None)
        if head_dim is None and isinstance(hidden, int) and isinstance(heads, int) and heads > 0:
            head_dim = hidden // heads
        known = all(isinstance(v, int) and v > 0 for v in (layers, kv_heads, head_dim))
        parameter = next(model.parameters())
        element_bytes = max(2, parameter.element_size()) if parameter.is_floating_point() else 2
        # Full dense decoder KV cache, plus a 256-token growth allowance.
        # Hybrid/sliding-window caches may use less; backend scratch and prompt
        # logits/attention need the separately configured workspace allowance.
        kv_bytes = (2 * layers * kv_heads * head_dim * element_bytes * input_ids.shape[0]
                    * (input_ids.shape[1] + max_new_tokens + 256)) if known else 0
        required = kv_bytes + self.inference_workspace_bytes
        report = self.check(parameter.device, required, operation="inference",
                            basis="dense decoder KV estimate + configured inference workspace"
                            if known else "configured inference workspace; KV estimate unavailable")
        if report is not None:
            report.update(kv_bytes=kv_bytes, kv_estimate_available=known,
                          workspace_bytes=self.inference_workspace_bytes)
        return report
