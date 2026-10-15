"""Trusted, versioned optimizer factories; names never import request code."""

from dataclasses import dataclass
import logging
from typing import Callable

import torch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizerFactory:
    create: Callable
    version: str
    supports_per_objective: bool = True
    estimate_state_bytes: Callable | None = None


OPTIMIZERS: dict[str, OptimizerFactory] = {}


def register_optimizer(name, factory, *, version, supports_per_objective=True, estimate_state_bytes=None):
    """factory(params, *, lr, **kwargs) must return a torch Optimizer.

    Factories must also accept CPU parameters for snapshot preflight. Version
    changes invalidate resume, so bump it when state interpretation changes.
    Optional estimate_state_bytes(optimizer) returns future state allocation
    bytes, excluding already allocated state. Register only at process startup.
    """
    if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
        raise ValueError("optimizer name and version must be nonempty strings")
    if not callable(factory) or (estimate_state_bytes is not None and not callable(estimate_state_bytes)):
        raise TypeError("optimizer factory and state estimator must be callable")
    entry = OptimizerFactory(factory, version, supports_per_objective, estimate_state_bytes)
    if name in OPTIMIZERS and OPTIMIZERS[name] != entry:
        raise ValueError(f"optimizer {name!r} is already registered")
    OPTIMIZERS[name] = entry


def get_optimizer(name):
    try:
        return OPTIMIZERS[name]
    except KeyError:
        raise ValueError(f"unknown optimizer {name!r}; registered: {sorted(OPTIMIZERS)}") from None


def _legacy_8bit(name):
    def create(params, *, lr, **kwargs):
        try:
            import bitsandbytes as bnb
        except ImportError:
            # Preserve the historical default on torch-only installations.
            # Construction/runtime errors are not dependency absence and must
            # surface instead of silently changing optimization algorithms.
            log.warning("bitsandbytes unavailable; %s uses legacy torch AdamW fallback", name)
            return torch.optim.AdamW(params, lr=lr, **kwargs)
        return getattr(bnb.optim, name)(params, lr=lr, **kwargs)
    return create


def _sgd_state_bytes(optimizer):
    return sum(p.numel() * p.element_size()
               for group in optimizer.param_groups if group.get("momentum", 0)
               for p in group["params"] if not optimizer.state.get(p))


register_optimizer("adamw", torch.optim.AdamW, version="torch-adamw-v1")
register_optimizer("sgd", torch.optim.SGD, version="torch-sgd-v1", estimate_state_bytes=_sgd_state_bytes)
register_optimizer("adamw8bit", _legacy_8bit("AdamW8bit"), version="bnb-adamw8bit-legacy-v1", supports_per_objective=False)
register_optimizer("lion8bit", _legacy_8bit("Lion8bit"), version="bnb-lion8bit-legacy-v1", supports_per_objective=False)
