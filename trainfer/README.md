# trainfer — trainfer daemon

**One mutable model. Always serving. Always trainable. Any objective, any time, via API.**

`trainfer` is a single-process FastAPI daemon that shares weights between inference and training, so feedback you send can land on the next inference request under a *typed* contract (not a best-effort). It is built on top of [`heiervang-technologies/ht-unsloth`](https://github.com/heiervang-technologies/ht-unsloth), which it consumes as a pinned git dependency — see [`../pyproject.toml`](../pyproject.toml).

```bash
# Start the daemon — Qwen3.5-9B on :8768 by default (TRAINFER_PORT to override)
python -m trainfer.console.launch

# Or construct your own ServeConfig and call trainfer.server.serve(cfg) directly:
#   from trainfer.config import ServeConfig
#   from trainfer.server import serve
#   serve(ServeConfig(model="unsloth/Qwen3-8B-unsloth-bnb-4bit", port=8765))
# See trainfer/console/launch.py for a working example.

# Chat (OpenAI-compatible)
curl -sS http://127.0.0.1:8768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}' | jq

# Train in the same process — returns commit_token N
curl -sS http://127.0.0.1:8768/v1/train \
  -H 'content-type: application/json' \
  -d '{"objective":"sft","samples":[{"prompt":"2+2?","response":"4."}]}' | jq

# Next chat that MUST see batch N: pass after_commit_token
curl -sS http://127.0.0.1:8768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"2+2?"}],"after_commit_token":N}' | jq
```

The response body's `trainfer.commit_cursor` is guaranteed ≥ `after_commit_token`. That's the "post a batch, next inference sees it" promise as a contract.

From Studio (in the `ht-unsloth` repo), start the daemon as a *capsule* instead: open the `/lile` page → **Load** → pick a model. Studio proxies `/v1/*` through its `studio/backend/routes/lile.py` to a daemon at `LILE_DAEMON_URL`, which Studio expects you to start from this repo (`python -m trainfer.console.launch`). Studio no longer spawns the daemon itself since the 2026-05-15 relocation.

## Start here

| File | Read when you want to know… |
|---|---|
| **[PLAN.md](PLAN.md)** | …the full spec and the north star. Living design document. |
| **[DESIGN.md](DESIGN.md)** | …the load-bearing decisions and *why* — engine choice, merge strategy, commit cursor, primitives-as-contract. One-pager. |
| **[STATUS.md](STATUS.md)** | …what actually works today. Every claim is cited by the test or benchmark that produced it. |
| **[GLOSSARY.md](GLOSSARY.md)** | …what "Razin-safe", "commit cursor", "π-only objective", or any other in-house term means. |
| **[SUBMISSION.md](SUBMISSION.md)** | …the hackathon writeup: what's novel vs. existing unsloth/TRL, shipped vs. scoped-out. |
| **[console/README.md](console/README.md)** | …how to poke a running daemon from the browser (demo chat + dashboard + live Prometheus graphs). |
| **[docs/research/](docs/research/)** | …PR specs, proofs (Razin, unlike trajectory bound), surveys, research roadmap. |

For Studio integration (the `/lile` page, feedback modal, live charts), see [`studio/frontend/src/features/lile/`](https://github.com/heiervang-technologies/ht-unsloth/tree/ht/studio/frontend/src/features/lile) and [`studio/backend/routes/lile.py`](https://github.com/heiervang-technologies/ht-unsloth/blob/ht/studio/backend/routes/lile.py) in the `ht-unsloth` repo.

## HTTP surface (summary)

| Route | Purpose |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat (streaming + non-streaming). Accepts `after_commit_token` to block on a training commit. Returns `trainfer.{response_id, commit_cursor, latency_s}`. |
| `POST /v1/train` | Enqueue a training batch. Request body: `{objective, samples[], ...objective-specific params}`. Returns `commit_token`. |
| `POST /v1/train/memorize` | Greedy-memorize a single `(prompt, response)` pair via the `trainfer.memorize` SFT-until-greedy-matches loop. Returns `commit_token` and the final greedy-rank fraction. Used by the chat UI's implicit-OK auto-SFT flow and by the R-001..R-004 research items. |
| `POST /v1/eval/greedy_rank` | Evaluate greedy-rank fraction on a `(prompt, response)` pair under current weights, serialized through the compute queue so the returned commit_cursor is meaningful. Returns `{fraction, matched, total, commit_token}`. Research use (R-001..R-004). |
| `POST /v1/feedback` | Route feedback (`binary`, `rewrite`, `preferred`, `nl_critique`) on a prior `response_id` to the matching objective (KTO / weighted SFT / hinge / CoH or CCPD v2). Returns `commit_token`. |
| `POST /v1/wait` | Block until a given commit_token is applied. |
| `GET  /v1/state/trajectory/tail?limit=N` | Paginated JSONL trajectory events (`train_step`, `inference`, `feedback`, `eval_point`, …). Drives live dashboards. |
| `POST /v1/state/snapshot/{save,load,list}` | Byte-exact state checkpointing (model + optimizer + residual + trajectory offset). |
| `POST /v1/state/merge` | Fold `merged_deltas` into the live base (see DESIGN.md §6 for why this is the 4-bit safe path). |
| `GET  /health` | `{ok, model, queue_depth, commit_cursor, merges}`. |
| `GET  /metrics` | Prometheus scrape — request counts, queue depth, per-objective train steps + loss histograms. |

The server exposes only **primitives**; workflow logic (curricula, replay streams, eval loops, RLAIF-style critique loops) lives in caller scripts. See DESIGN.md §"Primitives as contract" for the boundary.

## Directory map

```
trainfer/
├── server.py              # FastAPI app: routes, SSE, lifecycle
├── controller.py          # Routes feedback → objective; owns the compute queue
├── engine/                # train.py + inference.py + replay.py (idle replay scheduler)
├── objectives/            # sft, ntp, kto, coh, hinge, kl (anchor), unlike, ccpd, safety
│   └── verifiers/         # Pluggable verifiers for CCPD v2 and friends
├── state.py               # ModelState: LoRA residual, mode_lock, frozen_ref
├── snapshot.py            # Byte-exact save/load
├── trajectory.py          # Append-only JSONL; offset-based tail + iter_with_offsets
├── queue.py               # ComputeQueue: single-writer commit cursor, wait_for
├── reasoning.py           # <think>…</think> streaming parser (Qwen3, R1, Magistral, gpt-oss)
├── logging_backends.py    # Optional fan-out to W&B / TensorBoard / MLflow / trackio
├── metrics.py             # Prometheus counters, histograms, gauges
├── middleware.py          # Request logging, error wrapping, CORS
├── commit_stream.py       # /v1/commits/stream SSE primitive
├── config.py              # ServeConfig: model, adapter, ports, feature flags
├── console/               # Stdlib-only proxy + HTML apps (demo / dashboard / metrics)
├── ttrl_mv.py             # Idle-time TTRL majority-vote pseudo-label scheduler
├── tests/                 # Invariant + E2E + concurrent-load + CCPD + replay tests
└── docs/                  # Design notes
```

Long-running clients — tutors, RLVR/RLAIF loops, benchmark runners, the eval
harness — are not part of the daemon. They live in the
[`cont`](https://github.com/heiervang-technologies/cont) repo and drive
trainfer over HTTP.

## Invariants you can rely on (and test)

1. **Commit cursor is monotone and contract-typed.** `tests/test_queue_cursor.py`, `tests/test_merge_and_e2e.py::test_controller_commit_cursor_e2e`.
2. **Concurrent train+infer is safe.** `tests/test_concurrent_load.py` — 10 chats + 10 trains, strict monotone cursor, `mode_lock` guards the Unsloth mode flip.
3. **4-bit merge is idempotent and applied live in the forward pass.** `tests/test_residual_live_path.py` — merged-adapter forward stays within 0.05 nats of trained-adapter forward; fingerprint unchanged across null second merge.
4. **Snapshot round-trip is byte-exact.** `tests/test_trajectory_snapshot.py`.
5. **Every objective produces a finite scalar loss with `requires_grad=True` on LoRA params.** `tests/smoke_objectives.py`, `tests/test_ccpd_e2e.py`, `tests/test_kl_scope.py`, …

See STATUS.md for the full test table with results.

## Running the console (optional)

```bash
# Daemon on 8768, proxy on 8766 (stdlib-only, no venv needed)
python -m trainfer.console.launch &           # daemon  → :8768
python trainfer/console/proxy.py              # proxy   → :8766, forwards /api/* → :8768
# → http://127.0.0.1:8766/          — demo chat
# → http://127.0.0.1:8766/dashboard — trajectory + commits
# → http://127.0.0.1:8766/metrics   — live Chart.js scraper on /api/metrics
```

Override ports with `TRAINFER_PORT` (upstream) and `TRAINFER_PROXY_PORT` (proxy).

## License

Apache 2.0 — see [`LICENSE`](../LICENSE) at the repo root. The Studio frontend that talks to trainfer lives in `ht-unsloth` and is AGPL-3.0 there; the trainfer daemon itself imposes no such restriction on consumers using its HTTP API.
