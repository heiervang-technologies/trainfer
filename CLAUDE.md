# Agent Instructions — agi (lile)

This repository hosts **lile**, the LiveLearn local LLM daemon. lile was relocated from `heiervang-technologies/ht-unsloth` on 2026-05-15 to give it independent versioning and a smaller blast radius.

## Where things live

- `lile/` — the daemon: `server.py`, `controller.py`, `state.py`, `engine/`, `objectives/`, `queue.py`, `snapshot.py`, etc.
- `lile/console/` — web console + launcher (`launch.py`, `dashboard.html`, `metrics.html`).
- `lile/teach/` — offline teachers, RLVR loop (`rlvr_loop.py`), ARC-AGI-3 runner, eval harness.
- `lile/tests/` — pytest suite. `cpu_only` marker for torchless tests; `gpu` marker for real-model tests; `eval` marker for `lile.teach.eval` harness.
- `compose.lile-dev.yaml` — dev compose (daemon + studio).
- `.claude/skills/lile/SKILL.md` — Claude Code skill that codifies house style and invariants for this codebase. **Read it before doing non-trivial work in `lile/`.**
- `OPERATING.md` — env vars, ports, data layout, snapshot lifecycle, cross-repo policy.
- `MIGRATION.md` — what moved from ht-unsloth on 2026-05-15, what stayed, what downstream consumers need to change.

## Cross-repo dependency

lile depends on `unsloth` from the heiervang fork:

```
unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@ht-2026-05-15
```

(see [`pyproject.toml`](pyproject.toml)). Bump intentionally when ht-unsloth syncs with upstream — the load-bearing coupling is the runtime monkeypatch in `lile/state.py:55-77` of `unsloth.kernels.utils.matmul_lora`, guarded by `LileMatmulRebindError` on signature drift.

The Studio frontend + the `studio/backend/routes/lile.py` HTTP proxy stay in ht-unsloth and talk to the daemon over `LILE_DAEMON_URL`. Don't add Python imports across that boundary; if you need to share types, copy them.

## Development

```bash
pip install -e .
python -m lile.console.launch
```

Tests:

```bash
pytest -m cpu_only       # default for CI without a GPU
pytest                   # full suite
```

## Commit Guidelines

- Conventional commits: `feat(lile):`, `fix(lile):`, `docs(lile):`, `test(lile):`.
- Keep commits focused and atomic.
- Reference issues / PRs in the body when relevant.

## Pull Request Guidelines

- All changes go through a PR.
- Reference the issue or describe the motivating problem.
- The agent assignee is `@marksverdhai` (the synthetic-twin bot, not the human `@marksverdhei`).
