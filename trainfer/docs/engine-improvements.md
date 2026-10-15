# Engine improvement work

Scope: a unified, memory-efficient training/inference engine. Learning policies
and evidence of continual learning belong in `cont`.

## Acceptance checklist

- [x] Repeated train → consolidate → train cycles preserve a nonzero gradient
  path; model mutation and adapter toggles share the model lock.
- [x] Commit dependencies fail closed for unknown, timed-out, failed, and expired
  work; batch admission cannot silently accept a partial training request.
- [x] GPU work does not block the HTTP event loop; event publication remains
  ordered after task completion; shutdown does not race live model mutations.
- [x] Explicit scheduling controls balance inference latency with update work,
  preserving FIFO update ordering and avoiding starvation.
- [x] Consolidation has a bounded-memory alternative to dense GPU residuals,
  with quantitative preflight estimates and explicit approximation semantics.
- [ ] Memory admission, microbatch accumulation, and model/adapter configuration
  support training near capacity without changing the intended update silently.
- [x] Update procedures and optimizers have documented extension points beyond
  the built-in loss registry.
- [x] Snapshots distinguish weight restore from resumable optimization state,
  avoid stale files, and validate restore compatibility.
- [ ] A reproducible mixed-workload benchmark records memory peaks, inference
  latency/throughput, update throughput, switch overhead, and correctness.
- [ ] CPU and available GPU validation cover the changes; PR explains measured
  gains, costs, limitations, and decisions without claiming unmeasured benefits.

## Initial evidence and decisions

The baseline is origin/main at 1c6391f. The work uses an isolated worktree.
The existing local environment has no torch; the `cont` environment has torch
and can run CPU tensor regressions without installing into either repository.
An RTX 3090 is available, but its memory is shared with other workloads.

Code inspection found that reset zeros both LoRA factors, dense residuals are
copied to GPU after merge, synchronous queue handlers run on the asyncio loop,
and generation ignores failed/unknown/timed-out dependencies. These are
correctness and architectural findings, not GPU benchmark results.

Fix correctness before comparing performance. Preserve the existing dense
merge as an explicitly costed compatibility option while evaluating compact
consolidation; do not silently replace exact updates with lossy quantization.
Do not introduce learning policies or benchmark corpora from `cont`.

## Correctness and responsiveness changes

- Adapter reset now initializes A randomly and zeros B. This preserves a zero
  effective update while allowing B to receive gradients. Consolidation and
  explicit reset discard optimizer moments tied to the previous factors.
- Synchronous queue handlers execute in a thread. Completion callbacks publish
  commit events on the event loop after cursor advancement, avoiding unsafe
  cross-thread use of asyncio queues. Controller model operations and adapter
  toggles share the same reentrant lock.
- Batch admission reserves all chunks without suspension. A failure in an
  earlier chunk causes the final batch token to fail as well. No rollback is
  promised for already-executed steps.
- Dependency waits reject unknown, timed-out, and failed work. Streaming chat
  resolves its dependency before returning successful HTTP headers.
- The queue retains the last 1024 completion outcomes as metadata, without
  payloads, GPU results, or traceback references. Older tokens return HTTP 410
  (`commit_expired`), rather than guessing success after losing error evidence.
  Failed updates return HTTP 409 (`task_failed`). These are intentional wire
  contract changes; `/v1/wait` now reports timeout as HTTP 504.
- Shutdown closes admission before draining. Autosave happens after draining,
  under the model lock; it is skipped if an update outlives the shutdown grace.
  Unstarted tasks are dropped, but an active thread stays tracked and its token
  is not marked completed until the operation actually returns.

Validation so far: 310 CPU tests and 15 additional shutdown/combined-objective
tests pass using the existing torch CPU environment. These do not establish
GPU throughput or quantization quality. Real-model tests remain outstanding.

## Scheduling decision

Replace the unprioritized reentrant lock with a reentrant scheduler that lets
inference overtake updates for at most a configurable burst (default 4).
This preserves safe whole-operation boundaries and update FIFO order while
making the latency/progress trade-off explicit. It does not promise a hard
latency bound or interrupt a generation/backward pass. A deterministic real
thread contention test verifies FIFO within each class and that updates get a
turn even when multiple inference requests wait. Another tests timeout cleanup
and ownership. Health exposes wait/hold totals for subsequent mixed benchmarks.

The streaming generation worker now sends the terminal streamer signal on an
exception and propagates that exception to the consumer. Previously a failed
worker could leave the consumer blocked forever while holding the model lock.
A synthetic failing generation verifies stream termination and lock release.

## Validation update

- CPU tensor environment: 315 tests pass; torchless environment: 89 pass and
  3 skip. Additional shutdown/combined-objective tests: 15 pass.
- RTX 3090, torch 2.10.0+cu128, pinned Unsloth fork (2026.4.6), Unsloth Zoo
  2026.4.8, Transformers 5.5.0, bitsandbytes 0.50.2: real Qwen3-0.6B merge
  idempotence passes, as does three successive train/merge cycles with fresh
  optimizers and nonzero gradients/LoRA B updates after each reset.
- The installed bitsandbytes backend supports CPU NF4 quantization with nested
  statistics. This is feasibility evidence for CPU consolidation, not yet a
  validated implementation or a claim about retained model quality.
- Real concurrent-load script: 10 chats + 10 training updates passed in 4.6 s,
  final cursor 9; observed chat latency max 3.81 s / mean 2.39 s. This is a
  correctness smoke result on shared hardware, not a controlled speed comparison.
  The script now explicitly waits for all accepted updates before inspecting
  final counts. Its former assumption that all updates finish when chats finish
  depended on synchronous event-loop blocking. It now also has a pytest entry
  point marked `gpu`; previously invoking pytest on the file ran zero tests.

## Remaining implementation constraints

Compact consolidation must preserve quantization metadata and the live forward
path, with explicit approximation/error reporting. CPU NF4 staging is feasible
in the isolated validation environment; avoid creating a dense GPU model or
holding both complete GPU bases during a swap. Preview should measure update
error relative to the intended update, not merely relative to the much larger
base weights. Activation-weighted estimates require actual representative inputs.

Changing quantized base weights also changes snapshot semantics: an adapter-only
checkpoint can no longer identify a model by its original base reference alone.
Address base revisions and restore compatibility alongside consolidation, not
as a later documentation caveat. Include optimizer/RNG state for explicit resume,
while retaining intentional weights-only restore as a distinct operation.

Accumulation must preserve each objective's normalization; simply splitting
arbitrary losses is not equivalent. Expose update-procedure/optimizer extension
points and declare supported reductions rather than silently approximating
batch-dependent objectives. Benchmark memory and throughput over real mixed
workloads, including consolidation, using the same hardware/model/settings.

## Compact consolidation and snapshots: implementation/evidence

`cpu_requantize` stages one adapted projection at a time on CPU/disk and copies
the packed result into the existing GPU Parameter. It carries NF4/FP4 metadata
with each projection and preserves dense projections in dynamic 4-bit models.
Preview reports per-layer error and an aggregate error/update norm ratio without
mutating weights or RNG state. The default application gate requires error norm
strictly below update norm; callers may supply an explicit different bound.
Neither gate nor preview claims to predict task accuracy.

The CPU tensor estimate defaults to a 4 GiB ceiling per projection. This bounds
the planned dense working tensors, not process RSS or backend scratch space.
Consolidation pauses model execution while preparing/applying the candidate.
Disk staging retains both old and new projection weights for rollback.

Real Qwen3-0.6B, RTX 3090, three SFT steps, four CPU threads: explicitly allowing
relative error up to 2.0 yielded error/update norm **1.4132**, additional peak
GPU allocation **52,224 bytes**, and consolidation wall time **8.53 seconds**.
Target sum log-probability changed **−19.55 → −44.82**. This demonstrates a
memory-efficient mechanism and a harmful early consolidation, not retained
learning quality. The default gate now rejects this case. Training after the
explicitly permitted consolidation succeeded, and loading the earlier snapshot
restored the original target log-probability within 1e-5.

Snapshots atomically publish a manifest after syncing immutable payloads, check
checksums/compatibility before restore, and contain all adapted base projections.
Weights-only restore resets optimizer state. Explicit resume restores optimizer
moments, Python/NumPy/torch/CUDA RNG, and KTO running baseline. Adapter dtype is
preserved. Resumable snapshots reproduced the exact next CPU update and the
exact next Qwen3-0.6B GPU update with AdamW8bit. Legacy weights-only snapshots
remain usable against their original base; mismatched revisions fail closed.
The trajectory offset stays provenance metadata and never truncates history.

CPU tests cover interrupted publication, empty-overwrite stale residuals,
preflight incompatibility, partial-copy rollback for consolidation and restore,
preview non-mutation, quantization-resolution rejection, and original/modified
base round trips. The real-model test also bounds additional GPU allocation to
32 MiB. CPU NF4 execution was verified on bitsandbytes 0.50.2 / torch 2.10.0.

Model loading now raises the documented `TrainferMatmulRebindError` if Unsloth's
kernel signature drifts; it previously warned and silently disabled the fast
residual path. An invalid autosave similarly fails startup instead of silently
serving cold weights. These deliberate failure semantics preserve engine
correctness and are documented in OPERATING.md.

Follow-up validation: 331 CPU tests passed (one optional dependency skip), and
12 controller/lifespan tests passed, including refusal to serve after a failed
autosave restore without overwriting that snapshot. Seven CPU bitsandbytes
consolidation tests and a GPU-resident residual preview memory regression pass.
The latter bounds additional GPU allocation below 1 MiB for a 2048×2048 bf16
residual: copy to CPU before converting to float32. Normal residual merges
already store their source tensors on CPU; this also covers resident inputs.

## Microbatch accumulation

`microbatch_size` splits each optimizer update independently of queue
`chunk_size`. SFT, weighted SFT, and combinations of these preserve their
respective denominators and outer weights. One clip/optimizer step follows all
backward passes under the model lock. A callable-specific reduction contract
allows extension to other separable losses. Stateful/batch-dependent objectives
and sidecars without such a contract reject before admission. No automatic OOM
retry changes the batch. Forward/backward failure clears partial gradients;
optimizer-step failure is not promised to roll back already mutated weights.

CPU tests compare two successive updates, parameter values, Adam moments,
clipped gradient norms, uneven slices, unequal/zero weights, fully masked
responses, combined objectives, late-forward failure cleanup, and queue commit
counts. Existing combined-objective tests pass too (20 tests together).

RTX 3090/Qwen3-0.6B, four samples, microbatch size one, rank 8, sequence limit
512, same saved weights/optimizer/RNG for each comparison:

| Input lengths | Full-batch peak allocated | Accumulated peak allocated | Full loss | Accumulated loss | Full/accumulated update cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unequal | 1,484,168,192 B | 892,971,008 B | 0.945974 | 0.952357 | 0.987562 |
| Equal | 1,469,107,200 B | 889,205,760 B | 0.817177 | 0.804951 | 0.985320 |

This is a 39–40% reduction in total peak allocation for these workloads, not a
largest-model capacity or throughput claim. Loss changes of approximately
0.7–1.5% remain with equal-length inputs. Thus full-batch bitwise/numerical
equivalence is not claimed for this low-precision backend.

An independent reference uses the existing combined-loss implementation with
one forward per sample, all graphs retained, and one combined backward. Its
loss matches accumulation within 1e-5 relative tolerance; gradient relative
error was at most 2.3e-11 and update cosine approximately 1.0. This isolates the
accumulation implementation from the change in forward tensor shapes. Long
vector norm/cosine measurements use float64: float32 test metrics initially
reported a spurious reference-update difference, even a gradient cosine above
one. The CPU full-batch comparison independently verifies reduction semantics.

Memory admission, broader model/adapter controls, general update/optimizer
extension points, and mixed-workload comparative benchmarks remain outstanding.

## Memory admission and model controls

Execution-time admission now checks driver free memory plus reusable allocator
reserve while holding the model lock. Training budgets missing gradients, lazy
built-in optimizer state, and an explicit caller-calibrated workspace. Chat
budgets a dense decoder KV estimate and an explicit extra workspace. Dense
merge budgets residual copies plus per-layer temporaries. Rejection never
silently changes data or selects a lossy strategy. The default 256 MiB reserve
is headroom, not a workload-fit guarantee; zero workspace budgets leave
activation/prefill costs unknown. Health makes these limitations observable.
Snapshot restore/direct state operations are not covered by these guards.

Rank/alpha/dropout, target projections, checkpointing mode, and base dtype are
now CLI choices. Unsloth's for_training defaults would re-enable checkpointing,
so engine and CCD/CCPD mode restoration explicitly preserve off. Resume checks
persisted model training settings. Backend-created non-LoRA trainable weights
are rejected because existing adapter snapshot/reset semantics cannot preserve
them. Fewer adapted projections reduce memory but also change the space of
possible updates; this is an explicit user choice, not an automatic compromise.

CPU coverage includes lazy-state estimates, existing gradients, reserve
rejection, existing/new residuals, KV geometry with grouped-query attention,
CPU bypass, cached versus driver memory, pre-generation rejection in both chat
paths, mode-lock release, CLI validation, backend argument propagation, and
resume incompatibility. RTX 3090 Qwen3-0.6B tests pass with q/v-only rank-4 LoRA,
bf16 loading, standard and disabled checkpointing, forced rejection before
forward with unchanged adapters, subsequent training, inference, then training
again. The concurrent ten-chat/ten-update test also passes with these guards
enabled. Real AdamW8bit snapshot resume passes after adding config checks.

This implements resource controls, not yet a calibrated largest-model capacity
study. The broader memory/capacity acceptance item remains open pending that
evidence and the final coverage audit.

Validation for this phase: 362 CPU tests passed (one optional-dependency skip),
95 torchless tests passed (three skips), and 38 focused CPU control/resume/CLI
tests passed. The final GPU-environment bundle passed 29 tests (two real-model
configuration cases plus CPU memory/resume tests); the separate concurrency
bundle passed both configuration cases and the real concurrent workload.
Changed-file Ruff and whitespace checks pass.

## Optimizer and update-procedure extensions

Trusted installed packages register through `trainfer.extensions` before model
load. Optimizer factories are selected by configured name, receive the live
parameters and JSON kwargs, and may provide memory estimates. Factory versions,
resolved classes, options, and exact parameter-group ordering are checked on
resume. Unknown names reject rather than silently selecting AdamW. Historical
8-bit dependency-absence/per-objective fallbacks remain documented for
compatibility; constructor/runtime failures now propagate.

Update procedures are distinct from losses. A procedure can implement a
derivative-free update or compose several `engine.gradient_step` calls. It owns
one queue task/commit and executes under the same model lock after invalidating
inference adapter caches. Pure validation runs before admission; trajectory/SSE
events name the procedure. Result metadata must be finite JSON without tensors
or retained graphs. Errors clear gradients and fail the commit; prior internal
updates are not rolled back. Long procedures delay inference for their whole
duration, so external learning policies choose the update-cycle granularity.

Procedure state and versions participate in resumable snapshots, while
weights-only restore and consolidation clear them. CPU preflight protects live
RNG even if plugin constructors consume randomness. Plugin state must use
portable tensors/primitive containers and validate independently of the model.
Direct custom allocations need a declared memory estimate; arbitrary trusted
Python algorithms cannot be automatically bounded by the engine.

Validation covers CPU optimizer/procedure resume equivalence, version and
parameter-order mismatches before mutation, RNG preservation on failed
preflight, copied-parameter rejection, separate per-objective instances,
memory callbacks, registration failure/idempotence, one-commit worker execution,
lock ownership, gradient cleanup, mode-switch failure propagation, and repeated
built-in steps. Real Qwen3-0.6B passes exact next-update restoration for a
registered momentum-SGD optimizer with a stateful two-step procedure, alongside
the existing AdamW8bit resume case. The full CPU run passed 378 tests (one skip).

## Decoder-path audit correction

Review of the pinned backend found that single-token cached decoding uses
`fast_linear_forward`/GEMV directly, bypassing `matmul_lora`. The previous
prefill/residual test did not cover this path. Installation now also patches
that entry point, validates its signature, and rebinds imported aliases. It
adds the residual only for sequence length one; prefill already goes through
the matmul wrapper and must not double-add it.

Adapter reset and restore also invalidate Unsloth's cached cast `_fast_lora`
tensors under the model lock. Updating float32 adapter Parameters alone leaves
those bf16 decoder caches stale until a training-mode switch.

CPU tests cover decoder/prefill exactly-once addition, signature drift,
idempotence, alias rebinding, and cache invalidation. The real Qwen regression
populates a quantized projection's decoder cache with a deliberately visible
adapter update, then checks merge equivalence and exact snapshot restoration.
It also checks that omitting the residual produces mean absolute output error
greater than 1, preventing a vacuous pass on tiny updates. Both that test and
the existing full-model residual-forward test pass. Full CPU suite: 383 pass,
one optional-dependency skip. Mixed-workload timing must be rerun after this
hot-path change; the earlier records are not final performance evidence.
