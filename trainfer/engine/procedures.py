"""Trusted stateful update procedures, executed as one serialized queue task."""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ProcedureFactory:
    create: Callable
    validate: Callable
    version: str


PROCEDURES: dict[str, ProcedureFactory] = {}


def register_procedure(name, factory, *, validate, version):
    """Register a factory() -> procedure with run(engine, spec), state_dict(),
    and load_state_dict(state). validate(spec) must be pure and reject invalid
    requests before admission. Factories/state loading cannot touch the live
    model; snapshot preflight instantiates them independently on CPU.

    run executes under the model lock, in the queue worker. It returns the
    usual {loss, components, skipped?} result. It may call engine.gradient_step
    repeatedly or implement a different algorithm over engine.state.model.
    It must not queue more work, make a second model, or retain autograd graphs.
    """
    if not isinstance(name, str) or not name or name == "gradient":
        raise ValueError("procedure name must be nonempty and cannot be 'gradient'")
    if not isinstance(version, str) or not version:
        raise ValueError("procedure version must be nonempty")
    if not callable(factory) or not callable(validate):
        raise TypeError("procedure factory and validator must be callable")
    entry = ProcedureFactory(factory, validate, version)
    if name in PROCEDURES and PROCEDURES[name] != entry:
        raise ValueError(f"procedure {name!r} already registered")
    PROCEDURES[name] = entry


def get_procedure(name):
    try:
        return PROCEDURES[name]
    except KeyError:
        raise ValueError(f"unknown update procedure {name!r}; registered: {sorted(PROCEDURES)}") from None
