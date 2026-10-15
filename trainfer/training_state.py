"""Portable training state: tensors and primitive values, loaded weights-only.

These helpers intentionally exclude model weights, queue cursors, trajectory
rewrites, and external learning policies. Optimizer parameter order is checked
before restoration to avoid silently assigning moments to different adapters.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported training-state value: {type(value).__name__}")


def rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(saved):
    random.setstate(saved["python"])
    torch.set_rng_state(saved["torch"].cpu())
    ns = saved["numpy"]
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), *ns[2:]))
    if saved["cuda"]:
        if not torch.cuda.is_available() or len(saved["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the training snapshot")
        torch.cuda.set_rng_state_all(saved["cuda"])


def export_training(engine):
    from .objectives.kto import _Z0_EMA

    return {
        "version": 1,
        "frozen_reference": engine.state.frozen_ref is not None,
        "parameters": [(n, list(p.shape)) for n, p in engine.state.model.named_parameters() if p.requires_grad],
        "config": {
            "lr": engine.lr,
            "grad_clip": engine.grad_clip,
            "per_objective": engine.per_objective,
            "per_objective_lr": engine.per_objective_lr,
            "optimizer_class": engine.optimizer_class,
            "default_watchlist": engine.default_watchlist,
        },
        "optimizers": {
            key: {
                "class": type(opt).__module__ + "." + type(opt).__qualname__,
                "state": cpu_tree(opt.state_dict()),
            }
            for key, opt in engine._opts.items()
        },
        "kto_ema": [(beta, value) for (model_id, beta), value in _Z0_EMA.items() if model_id == id(engine.state.model)],
        "rng": rng_state(),
    }


def validate_training(engine, saved):
    from .engine.train import TrainEngine

    if saved.get("version") != 1:
        raise ValueError("unsupported training-state version")
    names = [(n, list(p.shape)) for n, p in engine.state.model.named_parameters() if p.requires_grad]
    if saved["parameters"] != names:
        raise ValueError("trainable parameter names/order/shapes differ from snapshot")
    current = {
        "lr": engine.lr, "grad_clip": engine.grad_clip,
        "per_objective": engine.per_objective,
        "per_objective_lr": engine.per_objective_lr,
        "optimizer_class": engine.optimizer_class,
        "default_watchlist": engine.default_watchlist,
    }
    if saved["config"] != current:
        raise ValueError("optimizer configuration differs; use weights-only restore or matching configuration")
    if saved["frozen_reference"] != (engine.state.frozen_ref is not None):
        raise ValueError("frozen-reference configuration differs from training snapshot")
    if saved["rng"]["cuda"] and (
        not torch.cuda.is_available() or len(saved["rng"]["cuda"]) != torch.cuda.device_count()
    ):
        raise ValueError("CUDA device count differs from training snapshot")
    # Validate optimizer state on CPU parameters before touching live weights.
    # This neither duplicates the base model nor allocates another GPU optimizer.
    params = [torch.nn.Parameter(torch.empty(tuple(shape))) for _, shape in names]

    class Parameters:
        def parameters(self):
            return iter(params)

    from types import SimpleNamespace

    candidate = TrainEngine(SimpleNamespace(model=Parameters()), **saved["config"])
    for key, record in saved["optimizers"].items():
        opt = candidate._optimizer(key)
        if type(opt).__module__ + "." + type(opt).__qualname__ != record["class"]:
            raise ValueError(f"optimizer implementation differs for {key!r}")
        opt.load_state_dict(record["state"])
        # Most torch optimizers defer checking moment geometry until step().
        for parameter, values in opt.state.items():
            for field in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                value = values.get(field)
                if value is not None and value.shape != parameter.shape:
                    raise ValueError(f"optimizer {field} shape differs from parameter")


def restore_training(engine, saved):
    from .objectives.kto import _Z0_EMA

    engine.reset_optimizer()
    for key, record in saved["optimizers"].items():
        engine._optimizer(key).load_state_dict(record["state"])
    for beta, value in saved["kto_ema"]:
        _Z0_EMA[(id(engine.state.model), beta)] = value
    restore_rng(saved["rng"])
