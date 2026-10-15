# Operating trainfer

Day-to-day knobs for running the trainfer daemon: env vars, ports, data layout, snapshot lifecycle.

For the *what* and *why* of trainfer, see [`trainfer/PLAN.md`](trainfer/PLAN.md) and [`trainfer/DESIGN.md`](trainfer/DESIGN.md). For the live-state contract and HTTP surface, see [`trainfer/README.md`](trainfer/README.md).

## Environment variables

| Env var | Purpose | Default | Read by |
|---|---|---|---|
| `TRAINFER_HOST` | Bind address for the daemon HTTP server. | `127.0.0.1` | `trainfer/config.py`, `trainfer/console/launch.py` |
| `TRAINFER_PORT` | Bind port for the daemon. | `8768` | `trainfer/config.py`, `trainfer/console/launch.py` |
| `TRAINFER_PROXY_BIND` | Bind address for the optional console proxy. | `127.0.0.1` | `trainfer/console/proxy.py` |
| `TRAINFER_PROXY_PORT` | Bind port for the console proxy. | `8766` | `trainfer/console/proxy.py` |
| `TRAINFER_AUTOLOAD_ON_BOOT` | If truthy, restore the `_autosave` snapshot at startup. | unset (auto) | `trainfer/config.py` |
| `TRAINFER_AUTOSAVE_ON_EXIT` | If truthy, snapshot to `_autosave` on graceful shutdown. | unset (auto) | `trainfer/config.py` |
| `TRAINFER_DEV_AUTORELOAD` | `1` → enable [`jurigged`](https://github.com/breuleux/jurigged) hot reload of function bodies. Requires `pip install -e .[dev]`. | unset | `trainfer/server.py`, `trainfer/dev/autoreload.py` |
| `TRAINFER_AUTORELOAD_PATTERN` | Glob the autoreload watcher should track. | `trainfer/**/*.py` | `trainfer/dev/autoreload.py` |
| `TRAINFER_SIGTERM_DEADLINE` | Wall-clock seconds before forced shutdown after `SIGTERM`. | per `ServeConfig.shutdown_deadline_s` | `trainfer/server.py` |
| `TRAINFER_SIGTERM_GRACE` | Extra grace after the deadline before `SIGKILL`-equivalent cleanup. | per `ServeConfig.shutdown_hard_stop_grace_s` | `trainfer/server.py` |
| `UNSLOTH_DISABLE_STATISTICS` | Set to `1` in tests and CI to silence Unsloth telemetry. | unset | upstream Unsloth |
| `HF_HUB_ENABLE_HF_TRANSFER` | Speed up HF model downloads. | unset (off) | transformers/huggingface_hub |

Studio consumers still read `LILE_DAEMON_URL` (and legacy `LILE_HOST` + `LILE_PORT`) from `studio/backend/routes/lile.py` in [`ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth) — point Studio at the daemon you start from this repo.

## Compute scheduling and commit dependencies

`--inference-burst N` (default 4) lets at most N consecutive inference operations
run while an update is waiting. Updates retain FIFO order; inference requests
retain order within their class. Scheduling occurs between complete generations
and update operations, so a long generation or backward pass is not preempted.
`/health.scheduler` reports operation counts, cumulative wait/hold time, and
waiting requests for each class. Set N lower to favor update progress, or higher
to favor inference under contention; benchmark your workload before tuning it.

Training requests are admitted as a complete set of chunks or rejected with
`queue_full`. An accepted request can still partially execute if a later step
fails: there is no automatic rollback. Its final token reports any chunk failure.
Clients should check dependency outcomes rather than infer success from the
completion cursor alone. `/v1/wait` and dependent chat return `task_failed`
(409), `not_found` (404), or `timeout` (504) instead of generating after an
unsatisfied dependency. Completion history retains 1024 metadata records;
older tokens return `commit_expired` (410). Tokens are process-local.

Shutdown closes admission and drains before autosave. If an update outlives both
the deadline and grace period, it stays tracked until it finishes; autosave is
skipped to avoid writing an inconsistent checkpoint. Queued, unstarted work is
resolved as dropped. A process supervisor may still need to terminate a stuck
CUDA operation; the engine never pretends that such an operation completed.

## Ports

| Port | Service | Default |
|---|---|---|
| `8768` | trainfer daemon — `/health`, `/v1/chat/completions`, `/v1/train`, `/v1/feedback`, `/v1/state/*` | configurable via `TRAINFER_PORT` |
| `8766` | Optional console proxy (browser-friendly demo + dashboard + metrics scraper) | configurable via `TRAINFER_PROXY_PORT` |

## Data layout

`ServeConfig.data_dir` defaults to `./trainfer_data/` (relative to the working directory). Inside:

- `trajectory.jsonl` — append-only event log (`train_step`, `inference`, `feedback`, `eval_point`, …). See [`trainfer/trajectory.py`](trainfer/trajectory.py).
- `snapshots/<name>/` — a manifest pointing to versioned tensor payloads, adapted base projections, and optional optimizer/RNG state. See [`trainfer/snapshot.py`](trainfer/snapshot.py).
- `daemon.log` — written by the legacy Studio capsule spawn path (still readable; no longer produced by the in-repo entrypoint).
- `rlvr_loop.jsonl` — RLVR scheduler events. See [`teach/rlvr_loop.py`](https://github.com/heiervang-technologies/cont/blob/main/teach/rlvr_loop.py) in the `cont` repo.
- `tutor_run_01/` — committed eval baselines (`cold_500_det.json`, `trained_500_det{,_run2}.json`, `length_compression_finding.md`). Used as the anchor for length-compression regressions.

Override the directory at the config layer (`ServeConfig(data_dir=...)`) or via the CLI (`python -m trainfer.server --data-dir ...`).

## Snapshot lifecycle

- Snapshots are taken through the compute queue (single-writer), so `/v1/state/snapshot/save` is serialized against training and merges.
- The reserved name `_autosave` is written automatically on graceful shutdown if `cfg.autosave_on_exit` is on, and reloaded at startup if `cfg.autoload_on_boot` is on.
- Saves publish the manifest only after all payloads are written and synced. An interrupted overwrite leaves the previous snapshot readable. Checksums and model/adapter/quantization compatibility are checked before restore. Adapter dtype is preserved.
- Snapshots include every adapted base projection, including packed quantization metadata and dense projections in mixed-precision models. This costs disk space but permits restoring an earlier base after consolidation. Restore stages rollback data on disk and copies projections in place, without a second GPU base.
- Default load is **weights-only**: it restores weights/residuals and resets optimizer history. Save with `{"name":"session","include_training":true}` and load with `{"name":"session","resume":true}` to resume optimizer moments, Python/NumPy/torch/CUDA RNG state, and the model's KTO running baseline. Resume requires matching optimizer configuration, parameter order/shapes, reference mode, and CUDA device count. Identical numerical behavior across different software/hardware is not promised.
- `--autosave-training-state` enables resumable autosaves and uses them on boot. Older weights-only autosaves still restore their weights. An invalid autosave now fails startup instead of silently serving a cold model; use `--no-autoload-on-boot` explicitly when a cold start is intended.
- The trajectory remains append-only. Its saved byte offset is provenance metadata, not a request to truncate the log. Queue tokens remain process-local and are not rewound by snapshot load.

## Compact consolidation and preview

`POST /v1/state/merge` without a body keeps the existing dense-residual strategy.
To estimate a compact merge without modifying the model:

```json
{"strategy":"cpu_requantize","preview":true}
```

To apply it, use `{"strategy":"cpu_requantize"}`. It prepares each projection
on CPU and stages candidate/rollback weights under `data_dir/consolidation/`.
The model lock remains held during this operation: it is a serving pause, not
an asynchronous weight copy. Packed GPU storage is overwritten in place; dense
projections remain in their original dtype. The adapter and optimizer restart
after a successful merge. A null update does not requantize the base again.

Reports include per-layer and aggregate error relative to the intended update,
retained update norm/cosine, unchanged weight fraction, and estimated CPU
workspace. This estimates weight error, not task accuracy. Applying rejects an
error norm at least as large as the update by default. Callers can supply
`max_relative_update_error` to choose a different explicit bound; preview without
a bound always reports the candidate. Even passing the default gate does not
prove behavioral quality. Small updates can be erased by quantization.

`max_cpu_workspace_bytes` defaults to 4 GiB and screens the estimated dense CPU
working tensors before allocation. Backend/BLAS workspace and allocator overhead
mean it is not a hard RSS limit. Temporary disk use includes old and new packed
weights (and any dense adapted projections). Use a sufficiently large local data
volume. The CPU backend was validated with bitsandbytes 0.50.2 and torch 2.10.0;
older backends without CPU 4-bit operators cannot use this strategy.

## Cross-repo policy

trainfer pins `unsloth` to a specific commit on the heiervang fork:

```toml
unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@<tag>
```

(see [`pyproject.toml`](pyproject.toml)). Current pin: `ht-2026-05-15`.

**Tag cadence (tied to upstream-sync).** ht-unsloth cuts a tag named `ht-YYYY-MM-DD` after every successful rebase of `ht` onto upstream Unsloth's `main` — that boundary is the natural "stable point" where upstream just shipped a compatible state, our patches replay clean, CI is green. trainfer bumps its pin on the same beat. Expected cadence: 2–4 weeks when upstream is hot (Gemma releases, transformers point releases), 6–8 weeks when quiet. Goal: trainfer never lives more than one tag behind the last upstream-compatible point.

**When to bump the pin:**
1. ht-unsloth pushes a new `ht-YYYY-MM-DD` tag. Confirm `matmul_lora`'s signature still matches `_EXPECTED_MATMUL_LORA_PARAMS` in [`trainfer/state.py`](trainfer/state.py) — if it drifted, `test_signature_mismatch_rejects_install` verifies that model loading fails instead of silently ignoring residuals.
2. Open a one-line PR bumping the `@<tag>` reference in `pyproject.toml` + `CLAUDE.md`. Keep `MIGRATION.md` at the historical SHA — that doc records what was current at relocation time, not the moving pin.
3. Never `@ht` or `@main` in the pin — those are moving refs and break reproducibility. Always a tagged name (`ht-YYYY-MM-DD`) or, in emergencies, a frozen SHA.

Bumping is a deliberate two-PR dance: (1) PR in `ht-unsloth` lands the upstream sync, (2) PR in `trainfer` bumps the pin and runs the full GPU test suite against the new combination.

## Smoke testing (no GPU)

A clean torchful-but-no-unsloth install is enough to run the cpu_only suite:

```bash
python -m venv .venv && . .venv/bin/activate
pip install --no-deps -e .
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install fastapi 'uvicorn[standard]' httpx 'pydantic>=2' \
            prometheus-client safetensors scipy transformers \
            pytest pytest-asyncio respx

pytest -m cpu_only
```

The conftest at [`trainfer/tests/conftest.py`](trainfer/tests/conftest.py) automatically skips GPU- and unsloth-dependent tests when those modules are missing, so the cpu_only bucket stays green on a minimal install.

Full GPU smoke (loads Qwen3-0.6B; ~90 s end-to-end):

```bash
pip install -e .                         # full unsloth git dep
python -m trainfer.tests.smoke_server        # in-process server, /v1/train + /v1/chat roundtrip
```
