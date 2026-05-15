# Operating lile

Day-to-day knobs for running the lile daemon: env vars, ports, data layout, snapshot lifecycle.

For the *what* and *why* of lile, see [`lile/PLAN.md`](lile/PLAN.md) and [`lile/DESIGN.md`](lile/DESIGN.md). For the live-state contract and HTTP surface, see [`lile/README.md`](lile/README.md).

## Environment variables

| Env var | Purpose | Default | Read by |
|---|---|---|---|
| `LILE_HOST` | Bind address for the daemon HTTP server. | `127.0.0.1` | `lile/config.py`, `lile/console/launch.py` |
| `LILE_PORT` | Bind port for the daemon. | `8768` | `lile/config.py`, `lile/console/launch.py` |
| `LILE_PROXY_BIND` | Bind address for the optional console proxy. | `127.0.0.1` | `lile/console/proxy.py` |
| `LILE_PROXY_PORT` | Bind port for the console proxy. | `8766` | `lile/console/proxy.py` |
| `LILE_AUTOLOAD_ON_BOOT` | If truthy, restore the `_autosave` snapshot at startup. | unset (auto) | `lile/config.py` |
| `LILE_AUTOSAVE_ON_EXIT` | If truthy, snapshot to `_autosave` on graceful shutdown. | unset (auto) | `lile/config.py` |
| `LILE_DEV_AUTORELOAD` | `1` → enable [`jurigged`](https://github.com/breuleux/jurigged) hot reload of function bodies. Requires `pip install -e .[dev]`. | unset | `lile/server.py`, `lile/dev/autoreload.py` |
| `LILE_AUTORELOAD_PATTERN` | Glob the autoreload watcher should track. | `lile/**/*.py` | `lile/dev/autoreload.py` |
| `LILE_SIGTERM_DEADLINE` | Wall-clock seconds before forced shutdown after `SIGTERM`. | per `ServeConfig.shutdown_deadline_s` | `lile/server.py` |
| `LILE_SIGTERM_GRACE` | Extra grace after the deadline before `SIGKILL`-equivalent cleanup. | per `ServeConfig.shutdown_hard_stop_grace_s` | `lile/server.py` |
| `UNSLOTH_DISABLE_STATISTICS` | Set to `1` in tests and CI to silence Unsloth telemetry. | unset | upstream Unsloth |
| `HF_HUB_ENABLE_HF_TRANSFER` | Speed up HF model downloads. | unset (off) | transformers/huggingface_hub |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `OPENROUTER_API_KEY` | Used by `lile/teach/` for offline teachers (Claude / GPT-OSS-120B / OpenRouter). Not needed by the daemon itself. | unset | `lile/teach/teacher_oss120b.py` etc. |

Studio consumers also read `LILE_DAEMON_URL` (and legacy `LILE_HOST` + `LILE_PORT`) from `studio/backend/routes/lile.py` in [`ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth) — point Studio at the daemon you start from this repo.

## Ports

| Port | Service | Default |
|---|---|---|
| `8768` | lile daemon — `/health`, `/v1/chat/completions`, `/v1/train`, `/v1/feedback`, `/v1/state/*` | configurable via `LILE_PORT` |
| `8766` | Optional console proxy (browser-friendly demo + dashboard + metrics scraper) | configurable via `LILE_PROXY_PORT` |

## Data layout

`ServeConfig.data_dir` defaults to `./lile_data/` (relative to the working directory). Inside:

- `trajectory.jsonl` — append-only event log (`train_step`, `inference`, `feedback`, `eval_point`, …). See [`lile/trajectory.py`](lile/trajectory.py).
- `snapshots/<name>/` — byte-exact checkpoints: model weights + optimizer state + bf16 residual + trajectory offset. See [`lile/snapshot.py`](lile/snapshot.py).
- `daemon.log` — written by the legacy Studio capsule spawn path (still readable; no longer produced by the in-repo entrypoint).
- `rlvr_loop.jsonl` — RLVR scheduler events. See [`lile/teach/rlvr_loop.py`](lile/teach/rlvr_loop.py).
- `tutor_run_01/` — committed eval baselines (`cold_500_det.json`, `trained_500_det{,_run2}.json`, `length_compression_finding.md`). Used as the anchor for length-compression regressions.

Override the directory at the config layer (`ServeConfig(data_dir=...)`) or via the CLI (`python -m lile.server --data-dir ...`).

## Snapshot lifecycle

- Snapshots are taken through the compute queue (single-writer), so `/v1/state/snapshot/save` is serialized against training and merges.
- The reserved name `_autosave` is written automatically on graceful shutdown if `cfg.autosave_on_exit` is on, and reloaded at startup if `cfg.autoload_on_boot` is on.
- Restore is byte-exact: model weights, optimizer state, bf16 residual, and the trajectory log offset all roll back together. There is no "partial" restore.

## Cross-repo policy

lile pins `unsloth` to a specific commit on the heiervang fork:

```toml
unsloth @ git+https://github.com/heiervang-technologies/ht-unsloth@<sha>
```

(see [`pyproject.toml`](pyproject.toml)).

**When to bump the pin:**
1. ht-unsloth rebases on upstream Unsloth and merges into `ht`. Confirm `matmul_lora`'s signature still matches `_EXPECTED_MATMUL_LORA_PARAMS` in [`lile/state.py`](lile/state.py:60) — if it drifted, the signature-guard test (`lile/tests/test_matmul_lora_patch.py::test_signature_mismatch_skips_install`) is the canary.
2. ht-unsloth cuts a release tag. Prefer `@<tag>` over `@<sha>` for readability once one exists.
3. Never `@ht` or `@main` in the pin — those are moving refs and break reproducibility.

Bumping is a deliberate two-PR dance: (1) PR in `ht-unsloth` lands the upstream sync, (2) PR in `agi` bumps the pin and runs the full GPU test suite against the new combination.

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

The conftest at [`lile/tests/conftest.py`](lile/tests/conftest.py) automatically skips GPU- and unsloth-dependent tests when those modules are missing, so the cpu_only bucket stays green on a minimal install.

Full GPU smoke (loads Qwen3-0.6B; ~90 s end-to-end):

```bash
pip install -e .                         # full unsloth git dep
python -m lile.tests.smoke_server        # in-process server, /v1/train + /v1/chat roundtrip
```
