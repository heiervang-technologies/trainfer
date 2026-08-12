# trainfer

**Train and infer at the same time.** A single-process FastAPI daemon that shares one set of weights between inference and training.

> One mutable model. Always serving. Always trainable. Any objective, any time, via API.

Feedback you POST lands on the next inference request under a *typed* contract, not a best-effort: `/v1/train` returns a `commit_token`, and `/v1/chat/completions` accepts `after_commit_token` to block until that training step is reflected in the forward pass. That contract is what makes continual learning something you can build on rather than hope for.

trainfer is the serving substrate. The continual-learning research that runs on top of it — teachers, RLVR/RLAIF loops, benchmark runners, the eval harness, campaign journals — lives in [`cont`](https://github.com/heiervang-technologies/cont) and drives this daemon over HTTP.

## Where to look first

| You want to… | Read |
|---|---|
| Get the daemon running | [`OPERATING.md`](OPERATING.md) — env vars, ports, data layout, smoke test |
| Understand what trainfer does and why | [`trainfer/PLAN.md`](trainfer/PLAN.md) (north-star spec) + [`trainfer/DESIGN.md`](trainfer/DESIGN.md) (one-pager) |
| See what's actually shipped | [`trainfer/STATUS.md`](trainfer/STATUS.md) — every claim is cited by a test |
| Learn the in-house vocabulary | [`trainfer/GLOSSARY.md`](trainfer/GLOSSARY.md) |
| Move off an older `lile` / `agi` checkout | [`MIGRATION.md`](MIGRATION.md) |
| Touch the daemon HTTP surface | [`trainfer/README.md`](trainfer/README.md) |

## Layout

```
trainfer/                     # the daemon — server, training engine, objectives, queue, state
trainfer/console/             # web console (launch.py, dashboard.html, metrics.html, …)
trainfer/ttrl_mv.py           # idle-time TTRL majority-vote pseudo-label scheduler
trainfer/objectives/verifiers # verifier registry + in-tree corpora + plugin entry point
trainfer/tests/               # pytest suite (cpu_only + gpu markers)
compose.trainfer-dev.yaml     # dev-mode docker compose (daemon + studio)
.claude/skills/trainfer/      # Claude Code skill for working in this repo
pyproject.toml                # trainfer package + cross-repo unsloth dep
```

## Cross-repo dependencies

trainfer consumes [`unsloth`](https://github.com/heiervang-technologies/ht-unsloth) (the heiervang fork) as a pinned git dependency. The single load-bearing coupling is a runtime monkeypatch of `unsloth.kernels.utils.matmul_lora` in [`trainfer/state.py`](trainfer/state.py); `TrainferMatmulRebindError` guards against upstream signature drift.

The HTTP-side companion (Studio frontend + `studio/backend/routes/lile.py` proxy) lives in `ht-unsloth` and talks to the daemon over `LILE_DAEMON_URL`. That side has **not** been renamed yet — see [`MIGRATION.md`](MIGRATION.md) for the pending follow-up.

[`cont`](https://github.com/heiervang-technologies/cont) depends on trainfer, never the other way round. Benchmark-shaped verifiers that carry their own corpora register into the daemon through the `trainfer.verifiers` entry-point group rather than being imported by it.

## Quickstart

```bash
pip install -e .
python -m trainfer.console.launch    # default: Qwen3-8B on :8768
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
```
