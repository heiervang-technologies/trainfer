# Migration history

Two moves matter if you have an older checkout.

## 2. `agi` → `trainfer` + `cont` (2026-08-12)

The `heiervang-technologies/agi` repo was split in two and both halves were renamed. The AGI framing is gone; the work is continual learning, and the two repos now say so.

| Was (`agi`) | Is now |
|---|---|
| `lile/` (daemon core, engine, objectives, queue, snapshot, state, console) | [`trainfer`](https://github.com/heiervang-technologies/trainfer) → `trainfer/` |
| `lile/teach/ttrl_mv.py` | `trainfer/ttrl_mv.py` (it is daemon runtime — `controller.py` imports it) |
| `lile/teach/logical/` | `trainfer/objectives/verifiers/corpora/logical/` (verifier-owned corpus) |
| `lile/teach/` (everything else) | [`cont`](https://github.com/heiervang-technologies/cont) → `teach/` |
| `lile/docs/research/`, `docs/research/` | `cont` → `docs/research/` |
| `autoresearch/` | `cont` → `autoresearch/` |
| `lile_data/`, `lile_data_nanbeige/` | `cont` → `data/`, `data_nanbeige/` |
| `AGENTS.md` (research-agent charter) | `cont` → `AGENTS.md` |

Both repos keep full `git` history — the split was done with `git filter-repo` path selection, and the rename landed as a single commit on top, so `git log --follow` works across it.

### Python-level renames

Everything named `lile` is now named `trainfer`, case-preserved:

| Was | Is now |
|---|---|
| `import lile.server` | `import trainfer.server` |
| `LileMatmulRebindError`, `LileError`, … | `TrainferMatmulRebindError`, `TrainferError`, … |
| `LILE_HOST`, `LILE_PORT`, `LILE_DEV_AUTORELOAD`, `LILE_PROXY_*`, `LILE_SIGTERM_*`, `LILE_AUTOSAVE_ON_EXIT`, `LILE_AUTOLOAD_ON_BOOT`, `LILE_AUTORELOAD_PATTERN` | same names with the `TRAINFER_` prefix |
| `./lile_data/` (default `ServeConfig.data_dir`) | `./trainfer_data/` |
| `compose.lile-dev.yaml` | `compose.trainfer-dev.yaml` |
| `pytest -m eval` | marker dropped — the eval harness lives in `cont` |
| `pip install -e '.[eval]'` | `pip install -e '.[verifiers]'` for the HumanEval verifier; the harness itself is `cont` |

Snapshot files are format-compatible. Rename (or re-point `ServeConfig.data_dir` at) your existing `lile_data/` directory and the daemon will pick it up.

### The verifier boundary

`lile/objectives/verifiers/__init__.py` used to hard-import `lile.teach.arc_agi_3.verifier` inside a `try/except`. Now that the ARC-AGI-3 runner lives in a different distribution, that import is gone and out-of-tree verifiers register through an entry-point group instead:

```toml
[project.entry-points."trainfer.verifiers"]
arc = "cont.teach.arc_agi_3.verifier"
```

`trainfer.objectives.verifiers.load_plugins()` is called once from the server lifespan and imports whatever is advertised there. A plugin that fails to import is logged and skipped — a broken third-party verifier can't take down the registry.

(ARC-AGI-3 keeps its name. It is the ARC Prize benchmark's name, not our framing.)

### Open follow-ups

- **`ht-unsloth` still says `lile`.** `studio/backend/routes/lile.py`, `studio/frontend/src/features/lile/`, the `/lile` Studio page, and `LILE_DAEMON_URL` are unchanged. Studio talks to the daemon over plain HTTP, so nothing is broken — but the names are now inconsistent and want a follow-up PR in that repo.
- Workflow `@main` → version-tag pins on the `heiervang-technologies/core` reusable workflows used by `.github/workflows/`.

## 1. `ht-unsloth` → `agi` (2026-05-15)

The daemon (then called `lile`) lived inside [`heiervang-technologies/ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth) until 2026-05-15, when it moved out to get independent versioning and a smaller blast radius. History was preserved via `git filter-repo` from `ht-unsloth@lile-v3` (later squashed onto `ht` as commit `53757129`), so per-file blame still points at the original authors.

### What stayed in ht-unsloth

- `unsloth/` — the upstream-tracking fork. trainfer consumes it as a pinned git dep.
- `studio/` — the React/FastAPI control plane. Talks to the daemon over HTTP.
- The upstream Unsloth test suite under `tests/`.

### What got rewritten on the ht-unsloth side

[`ht-unsloth#53`](https://github.com/heiervang-technologies/ht-unsloth/pull/53) landed alongside the extraction:

- `studio/backend/routes/lile.py:_lile_base_url()` requires `LILE_DAEMON_URL` (legacy `LILE_HOST` + `LILE_PORT` still resolve). It raises a loud `RuntimeError` if no daemon location is configured.
- `POST /api/lile/capsule/start` is no longer a subprocess spawn — the daemon module is not in that repo anymore. The endpoint reports daemon reachability.
- `POST /api/lile/capsule/stop` is a no-op (`externally_managed`).
- Removed orphan helpers (`_data_dir`, `_probe_health`, `_spawned_pid`) and the daemon-scoped pytest config.

## The `matmul_lora` patch contract

trainfer's only Python-level coupling with unsloth is the runtime monkeypatch at [`trainfer/state.py`](trainfer/state.py) of `unsloth.kernels.utils.matmul_lora`. The patch installer guards against signature drift and is exercised in CI without a real Unsloth via [`trainfer/tests/test_matmul_lora_patch.py`](trainfer/tests/test_matmul_lora_patch.py).

When bumping the unsloth pin in `pyproject.toml`, run that test against the new combination — a signature mismatch silently disables the fast-path residual (the PEFT `forward_pre_hook` backstop still fires, but you lose Unsloth's fast kernels).
