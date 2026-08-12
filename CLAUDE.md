# Agent Instructions — trainfer

This repository hosts **trainfer**: train and infer at the same time, one mutable model, over an HTTP API. It is the serving substrate for continual learning.

trainfer was extracted from `heiervang-technologies/ht-unsloth` on 2026-05-15 (as `lile`, in a repo called `agi`), then renamed and split from the research half on 2026-08-12. See [`MIGRATION.md`](MIGRATION.md).

## Scope — what belongs here and what doesn't

**Here:** the daemon and its contracts. Serving, training, the commit cursor, objectives, the queue, snapshots, state, the console, verifiers that are cheap and stdlib-shaped.

**Not here:** anything that *drives* the daemon. Teachers, RLVR/RLAIF loops, benchmark runners, eval harnesses, campaign journals, research data, and the surveys/proofs live in [`cont`](https://github.com/heiervang-technologies/cont). `cont` depends on `trainfer`; the dependency never points the other way.

The seam for benchmark-shaped verifiers that carry their own corpora is the `trainfer.verifiers` entry-point group — see `trainfer/objectives/verifiers/__init__.py:load_plugins`. If you find yourself wanting to `import cont` from this package, that is the tool you actually want.

## Where things live

- `trainfer/` — the daemon: `server.py`, `controller.py`, `state.py`, `engine/`, `objectives/`, `queue.py`, `snapshot.py`, etc.
- `trainfer/console/` — web console + launcher (`launch.py`, `dashboard.html`, `metrics.html`).
- `trainfer/ttrl_mv.py` — idle-time TTRL majority-vote pseudo-label scheduler. Daemon runtime, imported by `controller.py`.
- `trainfer/objectives/verifiers/` — verifier registry, in-tree verifiers, `corpora/` (pinned task sets the in-tree verifiers own), and the plugin hook.
- `trainfer/tests/` — pytest suite. `cpu_only` marker for torchless tests; `gpu` marker for real-model tests.
- `compose.trainfer-dev.yaml` — dev compose (daemon + studio).
- `.claude/skills/trainfer/SKILL.md` — Claude Code skill that codifies house style and invariants for this codebase. **Read it before doing non-trivial work in `trainfer/`.**
- `OPERATING.md` — env vars, ports, data layout, snapshot lifecycle, cross-repo policy.
- `MIGRATION.md` — the ht-unsloth → agi → trainfer/cont history and every rename in it.

## Cross-repo dependency

trainfer depends on `unsloth` from the heiervang fork:

```
unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@ht-2026-05-15
```

(see [`pyproject.toml`](pyproject.toml)). Bump intentionally when ht-unsloth syncs with upstream — the load-bearing coupling is the runtime monkeypatch in `trainfer/state.py` of `unsloth.kernels.utils.matmul_lora`, guarded by `TrainferMatmulRebindError` on signature drift.

The Studio frontend + the `studio/backend/routes/lile.py` HTTP proxy stay in ht-unsloth and talk to the daemon over `LILE_DAEMON_URL`. Those names still say `lile`; the rename on that side is a pending follow-up. Don't add Python imports across that boundary; if you need to share types, copy them.

## Development

```bash
pip install -e .
python -m trainfer.console.launch
```

Tests:

```bash
pytest -m cpu_only       # default for CI without a GPU
pytest                   # full suite
```

## Commit Guidelines

- Conventional commits: `feat(trainfer):`, `fix(trainfer):`, `docs(trainfer):`, `test(trainfer):`.
- Keep commits focused and atomic.
- Reference issues / PRs in the body when relevant.

## Pull Request Guidelines

- All changes go through a PR.
- Reference the issue or describe the motivating problem.
- The agent assignee is `@marksverdhai` (the synthetic-twin bot, not the human `@marksverdhei`).
