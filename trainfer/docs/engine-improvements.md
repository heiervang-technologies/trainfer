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
- [ ] Consolidation has a bounded-memory alternative to dense GPU residuals,
  with quantitative preflight estimates and explicit approximation semantics.
- [ ] Memory admission, microbatch accumulation, and model/adapter configuration
  support training near capacity without changing the intended update silently.
- [ ] Update procedures and optimizers have documented extension points beyond
  the built-in loss registry.
- [ ] Snapshots distinguish weight restore from resumable optimization state,
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
