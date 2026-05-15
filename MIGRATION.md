# Migration: ht-unsloth → agi (2026-05-15)

lile lived inside [`heiervang-technologies/ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth) until 2026-05-15, then moved to this repo. This doc records what changed, what stayed, and what a downstream consumer needs to update.

## What moved here

| Path in ht-unsloth | Path in agi |
|---|---|
| `lile/` | `lile/` |
| `lile_data/tutor_run_01/{cold_500_det,trained_500_det,trained_500_det_run2}.json` + `length_compression_finding.md` | `lile_data/tutor_run_01/...` (same) |
| `compose.lile-dev.yaml` | `compose.lile-dev.yaml` |
| `.claude/skills/lile/SKILL.md` | `.claude/skills/lile/SKILL.md` |
| Lile-only pyproject extras (`eval`, `dev`) + the `lile/tests`-scoped pytest config | `pyproject.toml` (clean restart, lile-only) |

History was preserved via `git filter-repo` from `ht-unsloth@lile-v3` (later squashed onto `ht` as commit `53757129`). Per-file blame inside `lile/` still points at the original authors.

## What stayed in ht-unsloth

- `unsloth/` — the upstream-tracking fork. agi consumes it as a pinned git dep.
- `studio/` — the React/FastAPI control plane. Talks to the lile daemon over HTTP (`LILE_DAEMON_URL`).
- The upstream Unsloth test suite under `tests/`.

## What got rewritten on the ht-unsloth side

[`ht-unsloth#53`](https://github.com/heiervang-technologies/ht-unsloth/pull/53) landed alongside the agi import:

- `studio/backend/routes/lile.py:_lile_base_url()` now requires `LILE_DAEMON_URL` (legacy `LILE_HOST` + `LILE_PORT` still resolve). It raises a loud `RuntimeError` if no daemon location is configured.
- `POST /api/lile/capsule/start` is no longer a subprocess spawn — `python -m lile.server` is not in that repo anymore. The endpoint now reports daemon reachability.
- `POST /api/lile/capsule/stop` is a no-op (`externally_managed`).
- Removed orphan helpers (`_data_dir`, `_probe_health`, `_spawned_pid`) and the lile-scoped pytest config.

## What a downstream consumer needs to do

If you ran lile from a ht-unsloth checkout:

1. **Clone agi instead** for the daemon code: `git clone https://github.com/heiervang-technologies/agi`.
2. **Install lile** from this repo: `pip install -e .` (pulls the right unsloth fork commit transitively).
3. **Start the daemon as before**: `python -m lile.console.launch`.
4. **Tell Studio where it is** — set `LILE_DAEMON_URL` (e.g. `http://127.0.0.1:8768`) in the environment of your Studio backend process.
5. **Snapshot files are forward-compatible.** `ServeConfig.data_dir` still defaults to `./lile_data/`; copy your existing snapshot directory across if you want to keep continuity.

If you didn't touch lile from ht-unsloth directly (e.g. only used the Studio chat UI), the only operational change is the `LILE_DAEMON_URL` env var.

## What the `matmul_lora` patch contract requires

lile's only Python-level coupling with unsloth is the runtime monkeypatch at [`lile/state.py:63-118`](lile/state.py) of `unsloth.kernels.utils.matmul_lora`. The patch installer guards against signature drift and is exercised in CI without a real Unsloth via [`lile/tests/test_matmul_lora_patch.py`](lile/tests/test_matmul_lora_patch.py).

When bumping the unsloth pin in `pyproject.toml`, run that test against the new combination — a signature mismatch silently disables the fast-path residual (the PEFT forward_pre_hook backstop still fires, but you lose Unsloth's fast kernels).

## Open follow-ups

- HT-CHANGELOG entry in ht-unsloth recording the relocation (owner: lile-architect).
- Studio frontend lifecycle UI strings ("Load" / capsule-status pill) reworded from "starting daemon" to "checking daemon reachability" (owner: lile-architect).
- Eventual ht release tag → re-pin agi's unsloth dep off the raw SHA.
- Workflow `@main` → version tag pins on the heiervang-technologies/core reusable workflows used by `.github/workflows/`.
