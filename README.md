# agi — lile

Dedicated repository for **lile**, the LiveLearn local LLM daemon.

> One mutable model. Always serving. Always trainable. Any objective, any time, via API.

`lile` is a single-process FastAPI daemon that shares weights between inference and training, so feedback you send can land on the next inference request under a *typed* contract (not a best-effort). It is the load-bearing addition that used to live inside [`heiervang-technologies/ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth) and now lives here.

## Where to look first

| You want to… | Read |
|---|---|
| Get the daemon running | [`OPERATING.md`](OPERATING.md) — env vars, ports, data layout, smoke test |
| Understand what lile does and why | [`lile/PLAN.md`](lile/PLAN.md) (north-star spec) + [`lile/DESIGN.md`](lile/DESIGN.md) (one-pager) |
| See what's actually shipped | [`lile/STATUS.md`](lile/STATUS.md) — every claim is cited by a test |
| Learn the in-house vocabulary | [`lile/GLOSSARY.md`](lile/GLOSSARY.md) |
| Migrate from a ht-unsloth checkout | [`MIGRATION.md`](MIGRATION.md) |
| Touch the daemon HTTP surface | [`lile/README.md`](lile/README.md) |

## Layout

```
lile/                    # the daemon — server, training engine, objectives, queue, state
lile/console/            # web console (launch.py, dashboard.html, metrics.html, …)
lile/teach/              # offline teachers + RLVR loop + ARC-AGI-3 runner + eval harness
lile/tests/              # pytest suite (cpu_only + gpu markers)
compose.lile-dev.yaml    # dev-mode docker compose (daemon + studio)
.claude/skills/lile/     # Claude Code skill for working in this repo
pyproject.toml           # lile package + cross-repo unsloth dep
```

## Cross-repo dependency

lile consumes [`unsloth`](https://github.com/heiervang-technologies/ht-unsloth) (the heiervang fork) as a pinned git dependency. The single load-bearing coupling is a runtime monkeypatch of `unsloth.kernels.utils.matmul_lora` in [`lile/state.py`](lile/state.py); `LileMatmulRebindError` guards against upstream signature drift.

The HTTP-side companion (Studio frontend + `studio/backend/routes/lile.py` proxy) lives in `ht-unsloth` and talks to the lile daemon over `LILE_DAEMON_URL`.

## Quickstart

```bash
pip install -e .
python -m lile.console.launch    # default: Qwen3-8B on :8768
```

```bash
# OpenAI-compatible chat
curl -sS http://127.0.0.1:8768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}' | jq

# Train in the same process — returns commit_token N
curl -sS http://127.0.0.1:8768/v1/train \
  -H 'content-type: application/json' \
  -d '{"objective":"sft","samples":[{"prompt":"2+2?","response":"4."}]}' | jq

# Next chat that MUST see batch N
curl -sS http://127.0.0.1:8768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2?"}],"after_commit_token":N}' | jq
```

## Tests

```bash
pytest -m cpu_only           # no GPU required
pytest                       # full suite (GPU needed)
pytest -m "eval"             # eval harness (install with `pip install -e .[eval]`)
```
