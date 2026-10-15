# Engine improvement acceptance audit

This audit maps the original engine improvement checklist to implementation and
observable checks. Learning-policy design and evidence of continual learning
remain in `cont`. Resource measurements establish particular configurations and
trade-offs, not a universal fastest or largest-model optimum.

| Requirement | Implementation inspected | Evidence and limits |
| --- | --- | --- |
| Repeated consolidation remains trainable; one shared model | `state.py` resets A randomly/B to zero, invalidates decoder casts, patches matmul and cached single-token GEMV; controller/engine share the reentrant model lock | `test_adapter_restart.py`, `test_matmul_lora_patch.py`, real repeated-merge and cached-decoder regressions in `test_merge_and_e2e.py`; `test_residual_live_path.py`. Explicit compact application is followed by another nonzero-gradient update in `test_consolidation_serving.py`. |
| Dependencies and atomic batch admission | `queue.py` retains bounded outcomes and reserves complete batches; controller/server reject unsuccessful dependencies before streaming headers | `test_queue_cursor.py`, `test_commit_dependencies.py`, `test_accumulation.py`: unknown/failed/expired/timed-out tokens, earlier chunk failures, and no partial admission. Expired history returns 410 after 1024 outcomes. |
| Responsive event loop, ordered events, safe shutdown | Queue dispatch uses `asyncio.to_thread`; completion callback runs after cursor publication; shutdown tracks active threads and defers autosave/logger close | Thread-backed responsiveness/callback-order test, `test_queue_graceful_drain.py`, `test_graceful_shutdown.py`, `test_stream_failure.py`. Active work may exceed shutdown grace; it is not falsely marked completed or autosaved concurrently. |
| Scheduling trade-off and progress | `scheduling.py` selects FIFO requests within each class and bounds inference overtaking | Real thread contention asserts the exact service order and timeout cleanup. Nine real mixed runs verify all dependencies and bit-identical final adapters. Throughput is unchanged within observed variation; burst 1 favors update completion, burst 4 favors chat latency relative to burst 1. |
| Compact consolidation with quantitative preflight | `consolidation.py` processes one projection on CPU; `quantized.py` preserves dense/quantized formats and Parameter identity; staged bases support rollback | Seven bitsandbytes CPU tests cover preview nonmutation, error rejection, workspace rejection, partial-copy rollback, and snapshot base revisions. Real 0.6B application and 14B preview bound additional GPU memory. Default gate rejects tested early updates; explicit permissive runs do not prove retained learning. |
| Capacity controls without silently changing updates | `memory.py` checks free plus reusable reserved memory under lock; config/CLI expose adapter and checkpointing controls; accumulation uses callable-specific reduction contracts | CPU tests compare Adam moments/parameters across uneven weighted/masked microbatches and reject unsupported reductions before admission. Real q/v-only controls cover standard/off checkpointing; real accumulation cuts peak allocation about 40%. Pinned 14B train/infer/train uses 12.41 GB peak allocated memory. Admission rejects oversized dense merge before mutation. Workspace budgets need calibration; full-batch GPU numerical equivalence and an absolute capacity limit are not claimed. |
| Optimization methods beyond built-in losses | Versioned trusted optimizer factories and stateful procedures, installed extension registrars, one queue task/lock per procedure | CPU derivative-free and composed-gradient procedures, factory ownership/version/group validation, state/RNG preflight, error cleanup; real registered momentum-SGD/two-step procedure resumes the exact next update. Arbitrary plugin allocations need their own estimates; earlier internal steps are not rolled back on failure. |
| Reliable weights-only and resumable snapshots | Atomic manifest/payload publication, checksums, base projection payloads, compatibility preflight, disk-backed rollback, optimizer/procedure/RNG state | `test_snapshot_resume.py`, `test_trajectory_snapshot.py`, `test_consolidation.py`, optimizer/procedure extension suites, real AdamW8bit and SGD resume. Weights-only restore resets training state; snapshots increase disk usage; direct restore is outside execution admission guards. |
| Reproducible mixed metrics | Synthetic integration/resource tests with committed raw records and reproduction commands | `measurements/README.md`: nine scheduling runs include memory, token throughput, update throughput/latency, mode-switch cost, and first-token dependency checks. Queued consolidation separately measures its serving pause; 14B records include exact sequence lengths/checkpoint revision and preflight rejection. Shared GPU and three repetitions cannot establish production tail latency. |
| Justified PR and validation | PR #4 and `OPERATING.md` describe APIs, defaults, intentional error contracts, plugin trust, and approximation limits | Final validation results below; no claim that requantization preserves task quality or that continual learning has been demonstrated. |

## Validation record

The final CPU collection at source `5d9f8f8` passed **419 tests**, with one
optional-dependency skip. The torchless run passed **96 tests**, with three
skips, before the final CPU-only long-merge regression; that file is not part of
the torchless collection. Seven CPU bitsandbytes consolidation tests also passed in the GPU environment.
The final real-model regression bundle passed **15 tests** in 168.58 seconds
on the shared RTX 3090. Changed-Python-file Ruff and `git diff --check` pass.
The fetched upstream main remains at baseline `1c6391f`; no remote CI checks
are configured/reported for this PR. All checklist requirements are covered
within the stated engine scope; the limits below remain explicit.

Previously completed targeted GPU runs also cover the mixed scheduler, 14B
capacity/preview, queued consolidation, CCPD engine backward, and the
GPU-resident residual conversion memory bound. CCD's existing reasoning-model
expected failure remains a limitation, not a claimed passing objective.

The 332-second 14B preview exposed the former fixed 300-second merge wait. The
controller now awaits accepted merge completion; a queue-backed CPU regression
verifies result/caller attachment. Client/proxy deadlines must accommodate the
operation, and disconnecting does not cancel accepted mutations.

## Scope boundaries

No second serving model, automatic lossy fallback, automatic batch shrinking,
or continual-learning policy was introduced. The user selects approximation,
adapter capacity, workspace allowance, update granularity, and scheduling
preference. Exact task-quality evaluation and representative activation
sampling remain external policy responsibilities. The engine supplies error
previews, an activation-error helper, extension points, and measured costs.

## Reproduction

Use an existing torch CPU environment for the first command, a torchless
installation for the second, and the pinned HT Unsloth environment described in
`OPERATING.md` for the remaining commands. Keep model caches, temporary files,
and compiler caches on the configured bulk disk. Do not install GPU dependencies
into a shared research environment.

```bash
python -m pytest -m 'not gpu' -q
python -m pytest -m cpu_only -q
python -m pytest trainfer/tests/test_consolidation.py -q
python -m pytest trainfer/tests/test_merge_and_e2e.py \
  trainfer/tests/test_residual_live_path.py \
  trainfer/tests/test_concurrent_load.py -q
```

The first command's 419-pass count comes from a torch environment without
Unsloth or bitsandbytes; collection excludes modules whose imports require
Unsloth. A GPU environment collects additional real-model modules, some of which
predate consistent GPU markers. Use the explicit paths above for comparable
results. Resource measurement commands and their separate configurations are
in [measurements/README.md](measurements/README.md).
