"""Explicit reduction contracts for normalization-preserving accumulation.

A registered objective must be separable: its loss on a slice times that
slice's denominator is an additive numerator. No cross-sample terms, running
state updates, or batch-dependent gating may be hidden in such an objective.
Registration is by callable identity so replacing an objective cannot inherit
an incompatible reduction. Plugins register their own contract explicitly.
"""

from dataclasses import dataclass
import math
from typing import Callable

from ..objectives import get_objective
from ..objectives.sft import sft_loss, weighted_sft_loss


_REDUCTIONS: dict[Callable, Callable] = {}


def register_reduction(objective: Callable, denominator: Callable) -> None:
    """Register denominator(samples, kwargs) -> positive finite scalar.

    Match the objective's exact local reduction, including denominator clamps.
    The callback must be pure and must not allocate model activations.
    """
    _REDUCTIONS[objective] = denominator


register_reduction(sft_loss, lambda samples, kwargs: len(samples))
register_reduction(weighted_sft_loss, lambda samples, kwargs: max(
    sum(float(s.get("weight", 1.0)) for s in samples), 1e-6,
))


@dataclass
class Microbatch:
    name: str
    objective: Callable
    samples: list
    kwargs: dict
    scale: float


def plan(spec: dict) -> list[Microbatch]:
    size = spec.get("microbatch_size")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("microbatch_size must be a positive integer")
    if spec.get("batch_objectives"):
        raise ValueError("microbatch accumulation requires explicit sidecar reduction support")
    entries = spec.get("objectives") or [{"name": spec["objective"]}]
    result = []
    for entry in entries:
        name = entry["name"]
        objective = get_objective(name)
        reduction = _REDUCTIONS.get(objective)
        if reduction is None:
            raise ValueError(f"objective {name!r} has no accumulation reduction contract")
        samples = entry.get("samples")
        if samples is None:
            samples = spec.get("samples", [])
        kwargs = entry.get("kwargs")
        if kwargs is None:
            kwargs = spec.get("kwargs", {})
        if not samples:
            raise ValueError("microbatch accumulation requires nonempty samples")
        weight = float(entry.get("weight", 1.0))
        denominator = float(reduction(samples, kwargs))
        if not math.isfinite(weight) or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("accumulation weight/denominator must be finite and denominator positive")
        for start in range(0, len(samples), size):
            chunk = samples[start:start + size]
            local = float(reduction(chunk, kwargs))
            if not math.isfinite(local) or local <= 0:
                raise ValueError("microbatch denominator must be finite and positive")
            scale = weight * local / denominator
            if not math.isfinite(scale):
                raise ValueError("microbatch scale must be finite")
            result.append(Microbatch(name, objective, chunk, dict(kwargs), scale))
    return result
