# Engine resource measurements

These are synthetic integration checks of the shared model engine. They do not
measure learning quality and do not replace the learning-policy research in
`cont`. Raw records include dependency checks, exact final adapter equality,
package versions, device occupancy, and the source commit.

## Mixed training and inference

[Raw RTX 3090 / Qwen3-0.6B records](rtx3090-qwen06b-mixed.json)

Each run restores identical weights, AdamW8bit state, and RNG, then executes
four SFT updates (four samples each, microbatch one) alongside six streaming
chats (24 generated tokens each). Half the chats depend on an update commit.
Three repetitions rotate the policy order. GPU work remains serialized against
one live model. The RLock control uses the current corrected engine, not an
old-main build with event-loop blocking or broken adapter reset.

| Policy | Mean run time | Mixed generated tokens/s | Mean independent-chat TTFT | Mean dependent-chat TTFT | Mean update completion latency | Maximum allocated GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unprioritized RLock | 5.651 s | 25.50 | 1.046 s | 3.790 s | 3.326 s | 830,998,528 |
| Inference burst 1 | 5.669 s | 25.40 | 1.545 s | 4.476 s | 2.427 s | 819,701,760 |
| Inference burst 4 | 5.647 s | 25.50 | 1.157 s | 3.625 s | 3.386 s | 842,295,296 |

Burst 4 improves chat responsiveness versus burst 1 in this workload at the
cost of update latency. The unprioritized control has slightly lower independent
chat TTFT than burst 4; bounded scheduling does not establish an improvement
over that control here. Its deterministic contention tests separately verify
bounded overtaking. Aggregate throughput differs by less than 1%, within the
observed run variation: **no throughput gain is established**. These results
justify exposing the scheduling trade-off; they do not identify a universal
optimal burst. Default burst 4 is a latency-biased starting point, while callers
prioritizing update freshness can choose 1.

TTFT is measured at the first generated token delivered to the streamer,
before text buffering, and includes queue/dependency waiting. Decode throughput
excluding the first token was approximately 41.9–42.2 tokens/s. Synchronised
training/inference mode changes totaled 0.085–0.088 seconds per run (ten mode
changes), about 1.5–1.6% of run time. Timing instrumentation applies equally to
all policies. Peak reserved allocation is separately retained in the raw JSON.

All nine runs completed all four updates and six chats, satisfied dependencies
at first-token delivery, and produced **bit-identical final adapters** across
all policy interleavings. Shared GPU occupancy is recorded before/after each
run; other workloads were left running. Three repetitions do not establish
production tail latency or performance on different model/sequence sizes.

Reproduce using the pinned HT Unsloth environment documented in OPERATING.md:

```bash
OMP_NUM_THREADS=4 UNSLOTH_DISABLE_STATISTICS=1 \
  TRAINFER_RESOURCE_REPORT=/path/on/bulk-disk/mixed-resources.json \
  python -m pytest trainfer/tests/test_mixed_resource_tradeoffs.py -qs
```

Set `HF_HOME`, `TMPDIR`, `TRITON_CACHE_DIR`, and `TORCHINDUCTOR_CACHE_DIR` to bulk
storage before running. The validation environment used torch 2.10.0+cu128,
Unsloth 2026.4.6 from the pinned HT fork, Transformers 5.5.0, bitsandbytes 0.50.2,
and four CPU threads. Avoid running another benchmark or model download at the
same time. Warm-up, snapshot restore, and model loading are excluded from run
latency; the integration test asserts the same correctness conditions anew.

The recorded source is `f536f51`, including the cached-decoder residual fix.
These runs measure updates and chats; consolidation pause and memory costs are
measured separately below and in the engine improvement ledger.

## Selected 14B capacity check

[Raw RTX 3090 / Qwen3-14B records](rtx3090-qwen14b-capacity.json)

Checkpoint `unsloth/Qwen3-14B-unsloth-bnb-4bit`, revision
`46105e245750aad3be7fd1d81c21cb03a0e438ed`, ran at engine source `f536f51`
with the same package environment as above. All model Parameters were CUDA
resident; Unsloth gradient checkpointing/offload remained enabled. Rank 8 /
alpha 16 across the default seven projections gives 32,112,640 trainable
parameters. Four samples per update, microbatch one, had actual input lengths
381, 387, 393, and 399; configured maximum context was 1024.

| Phase | Wall time | Peak allocated GPU bytes | Peak reserved GPU bytes |
| --- | ---: | ---: | ---: |
| First update | 8.117 s | 12,340,082,688 | 12,593,397,760 |
| Eight-token inference | 0.768 s | 11,829,072,384 | 12,664,700,928 |
| Second update | 7.461 s | 12,406,562,816 | 12,664,700,928 |

Both updates had finite loss and nonzero gradients. Before loading, the shared
24 GiB card had approximately 15.9 GiB free. This demonstrates a selected 14B
configuration fitting alongside existing workloads, not the largest possible
model, full-context capacity, or learning quality. Training used explicit 1 GiB
workspace and 1 GiB reserve budgets; inference workspace was 256 MiB.

Dense residual merge estimated 28,563,210,240 additional bytes and was rejected
before mutation. Full compact preview inspected 280 projections in **332.407 s**
with **1,024 bytes** additional peak GPU allocation and a largest planned CPU
tensor workspace of 4,278,910,976 bytes (not an RSS bound). Adapters were unchanged.
Relative update error was **2.13555**, so the default consolidation gate rejects
this candidate. Keeping the adapter is necessary here; compact storage does not
by itself establish that an early update survives requantization. The preview
holds the model lock throughout, so this is also a serving-pause cost.

Reproduce after explicitly downloading the pinned checkpoint to bulk storage:

```bash
OMP_NUM_THREADS=4 UNSLOTH_DISABLE_STATISTICS=1 \
  TRAINFER_CAPACITY_MODEL=/path/to/pinned/local/checkpoint \
  TRAINFER_CAPACITY_PREVIEW=1 \
  TRAINFER_CAPACITY_REPORT=/path/on/bulk-disk/capacity.json \
  python -m pytest trainfer/tests/test_capacity_resources.py -qs
```

## Consolidation with queued serving work

[Raw Qwen3-0.6B consolidation/serving record](rtx3090-qwen06b-consolidation-serving.json)
uses source `222a223`, the same GPU environment, rank 8 / alpha 16, maximum
sequence length 256, and three single-sample SFT updates at the controller's
default learning rate. Three eight-token chats and another update were admitted
while the merge held the shared model lock; one chat depended on that update.

The compact merge took 5.665 s across 196 projections and added 52,224 peak GPU
bytes. Independent chats delivered their first stream events after 5.889 s and
6.096 s; the dependent chat after 6.467 s. Those times include text buffering,
not just first-token delivery. The next update completed after 6.397 s with a
nonzero gradient; all work completed in 6.616 s with correct commits. This
quantifies the serving pause rather than claiming background consolidation.

The test explicitly allows relative update error up to 10 to exercise actual
application and subsequent serving/training. Observed error was 9.21887, which
the default gate would reject. This is deliberately permissive integration
coverage, not a recommended merge policy or evidence of useful learning.

```bash
OMP_NUM_THREADS=4 UNSLOTH_DISABLE_STATISTICS=1 \
  TRAINFER_CONSOLIDATION_REPORT=/path/on/bulk-disk/consolidation-serving.json \
  python -m pytest trainfer/tests/test_consolidation_serving.py -qs
```
