"""Verifier registry — pass/fail (or graded) checks on candidate responses
where an objective ground-truth exists.

Consumed by TTRL-style pseudo-reward training (roadmap PR L) and future RL
workstreams. A verifier answers one question:

    Given ``prompt``, does ``candidate`` satisfy the domain-specific check?

Return values:

- ``True`` / ``1.0`` — candidate verifies cleanly
- ``False`` / ``0.0`` — candidate fails the check
- ``None`` — verifier is not applicable to this prompt (caller should skip;
  ``None`` is never coerced to False downstream)

Verifiers are adapters: they must never raise into the caller. The top-level
:func:`verify` dispatcher catches any exception and returns ``None`` so a
bad verifier can't take down the train loop.

Registering a custom verifier:

    from trainfer.objectives.verifiers import register

    @register("my_domain")
    def verify(prompt: str, candidate: str) -> bool | float | None:
        ...
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)

Verifier = Callable[[str, str], "bool | float | None"]

# Populated at module import by the seed modules below.
VERIFIERS: dict[str, Verifier] = {}


def register(domain: str) -> Callable[[Verifier], Verifier]:
    """Decorator that registers ``fn`` as the verifier for ``domain``."""

    def _wrap(fn: Verifier) -> Verifier:
        VERIFIERS[domain] = fn
        return fn

    return _wrap


def verify(domain: str, prompt: str, candidate: str) -> bool | float | None:
    """Dispatch to ``VERIFIERS[domain]``, swallowing adapter exceptions.

    Returns ``None`` when no verifier is registered for ``domain`` or the
    verifier itself raises — both indicate "can't judge this", never "fail".
    """
    fn = VERIFIERS.get(domain)
    if fn is None:
        return None
    try:
        return fn(prompt, candidate)
    except Exception as exc:
        log.warning("verifier %r raised %s — returning None", domain, exc)
        return None


def select(prompt: str) -> str | None:
    """Return the first registered verifier whose domain claims ``prompt``.

    Each seed verifier exposes a ``claims(prompt) -> bool`` sibling used here
    to let TTRL pick a verifier without the caller hard-coding domains.
    Custom verifiers without a ``claims`` attribute are skipped — register
    your own ``select`` if you need richer routing.

    Dispatch is insertion-order over ``VERIFIERS``; the first claim wins.
    Priority-sensitive verifiers (stricter claims that should pre-empt the
    seeds) must insert into ``VERIFIERS`` at the head rather than appending
    — e.g. ``VERIFIERS = {"my_strict": fn, **VERIFIERS}`` — or register
    their own ``select`` wrapper.
    """
    for domain, fn in VERIFIERS.items():
        claims = getattr(fn, "claims", None)
        if callable(claims) and claims(prompt):
            return domain
    return None


# Seed the registry. Order matters for :func:`select` — cheapest/strictest
# claim first, so "math" wins over "code" on ambiguous prompts.
from . import _math  # noqa: F401, E402
from . import _code  # noqa: F401, E402

# Logical-task verifier — pinned corpus at corpora/logical/tasks_v0.json;
# claims by prompt-hash so only catalogued prompts get routed here. Stdlib-
# only (regex + json), safe to import in any environment.
try:
    from . import _logical  # noqa: F401, E402
except Exception:  # pragma: no cover — defensive; corpus may be absent
    pass

# HumanEval verifier — wraps evalplus check_correctness in a fork-context
# child for sandboxed test execution. Importing here triggers the
# ``@register("humaneval")`` decorator. Wrapped in try/except because
# evalplus is an opt-in extra: slim environments (cpu_only test runs,
# torchless smoke) can still load the registry without it.
try:
    from . import _humaneval  # noqa: F401, E402
except Exception:  # pragma: no cover — defensive; evalplus is an extra
    pass


def load_plugins() -> list[str]:
    """Import out-of-tree verifiers advertised on the ``trainfer.verifiers``
    entry-point group, returning the domains they registered.

    This is the seam for benchmark-shaped verifiers that carry their own
    loaders, prompts, and task corpora and therefore do not belong in the
    daemon — the ARC-AGI-3 verifier in the ``cont`` repo is the reference
    consumer. A distribution opts in from its ``pyproject.toml``::

        [project.entry-points."trainfer.verifiers"]
        arc = "cont.teach.arc_agi_3.verifier"

    Importing the named module is what registers it; the ``@register``
    decorator does the rest. A plugin that fails to import is logged and
    skipped — a broken third-party verifier must never take down the
    daemon's registry.

    Called once from the server lifespan. Safe to call repeatedly: module
    import is cached and ``register`` is idempotent for a given domain.
    """
    from importlib.metadata import entry_points

    loaded: list[str] = []
    try:
        eps = entry_points(group="trainfer.verifiers")
    except Exception as exc:  # pragma: no cover — importlib.metadata edge
        log.warning("verifier plugin discovery failed: %s", exc)
        return loaded
    for ep in eps:
        before = set(VERIFIERS)
        try:
            ep.load()
        except Exception as exc:
            log.warning("verifier plugin %r failed to load: %s", ep.name, exc)
            continue
        loaded.extend(sorted(set(VERIFIERS) - before))
    if loaded:
        log.info("registered verifier plugins: %s", ", ".join(loaded))
    return loaded
