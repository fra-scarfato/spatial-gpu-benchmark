# Design notes, methodology and traps

The project's memory: every non-obvious choice, the alternative it was chosen over, and the
measurements behind it. Written to be usable directly when writing the thesis.

`README.md` is the front page — what the program is, how to build and run it, how to reproduce
the experiments. This file is the long form.

## Contents

1. Layout
2. Build and run
3. Methodology, and why each choice
4. Spatial partitioning with green contexts
5. Architecture decisions and the alternatives rejected
6. Measurements so far
7. Things learned the hard way
8. Environment
9. The measurement protocol: `utility/sweep.sh` and `utility/aggregate.py`
10. Experiment 3: MPS, and why it is not a green context
11. Open items
12. Reproducing a result
13. The models
14. Reproducing the results section

---

## 1. Layout

```
include/
  config.hpp      every parameter (compile-time cfg:: + run-time Options) and
                  the command-line declarations
  gpu.hpp         the GPU layer: Logger, PinnedHostBuffer, SmAllocation/WholeGpu,
                  SmPartition, PartitionPlan, WqConfig, GreenContextAllocation,
                  TrtEngine
  inputs.hpp      listImages, preprocess (the ImageNet input contract), InputSet
  stats.hpp       Stats + the statistics formatting (no CUDA, no images)
  run.hpp         RunSpec, StartGate, Network, Run
src/
  config.cpp      the command line: parseArgs + printUsage
  gpu.cpp         EVERY CUDA and TensorRT call in the program
  inputs.cpp
  stats.cpp
  run.cpp
  main.cpp        wiring only (~30 lines)
```

Dependencies are acyclic and point downward:

```
main.cpp → run.hpp → { stats.hpp, inputs.hpp, gpu.hpp, config.hpp }
                          inputs.hpp → gpu.hpp → config.hpp
                          stats.hpp  → (nothing)
```

**`stats.hpp` is a leaf** (only `<vector>`, `<cmath>`, `<numeric>`), so the
statistics can be read and tested without the GPU stack.

### Object model

```
Run                          one process invocation = one "run" of the protocol
├── Options                  a copy of the configuration
├── images                   the dataset file list
└── Network[0 .. N-1]        one per engine
    ├── SmAllocation             which SMs → the stream
    ├── TrtEngine                engine, context, device buffers
    ├── InputSet                 the tensors
    └── (measurement)            warm-up, timed loop, stats, CSV — behaviour
```

`Network` also **owns its measurement** rather than containing an
`Experiment` object. Rationale in §5.

---

## 2. Build and run

```bash
make                # build ./runner  (-O2 -std=c++17 -Wall)
make run            # development run
make full           # the protocol, fixed-input mode
make full-dataset   # the protocol, end-to-end mode
make clean
```

System include/library paths in the `Makefile` are **absolute on purpose**
(that is where the toolchain lives); project paths are relative:

| What | Where | Note |
|---|---|---|
| TensorRT headers | `/usr/include/aarch64-linux-gnu` | `NvInfer.h` lives here, not in `/usr/include` |
| OpenCV 4 headers | `/usr/include/opencv4` | `pkg-config` claims `/usr/local/include/opencv4`, which does not exist |
| CUDA libraries | `/usr/local/cuda/targets/sbsa-linux/lib` | `sbsa` = ARM server base architecture |

### Command line

```
runner <engine.engine> <image_dir> [options]           # one network
runner --dataset <dir> --net <engine>:<pct>[:<label>]  # one or more networks
        [--net <engine>:<pct>[:<label>] ...]
```

| Option | Meaning |
|---|---|
| `--net <spec>` | `engine:pct[:label]`; repeat to run several concurrently, each on its own green context. Percentages are of the device total and must sum to ≤ 100 |
| `--dataset <dir>` | dataset directory (with the `--net` form) |
| `--fixed <N>` | fixed set of N preprocessed inputs, cycled (default 32), done once at startup |
| `--end-to-end` | decode and preprocess inside the loop instead |
| `--images <N>` | limit the image pool to the first N (0 = all) |
| `--min-inferences <N>` | at least N inferences (default 5000; 0 disables) |
| `--min-duration <S>` | at least S seconds (default 30; 0 disables) |
| `--one-pass` | ignore both bounds: one pass over the input set |
| `--csv <prefix>` | output prefix (default `bench`) |
| `--allocation <name>` | `whole` (no partitioning) or `greenctx` |
| `--sm-percent <P>` | with the positional form: SMs for `greenctx`, % of the device |
| `--wq-scope <s>` | work queue sharing scope for each green context: `device_ctx` (the driver default: all contexts share one set of queues) or `green_ctx_balanced` (the driver tries to give balanced green contexts non-overlapping queues). **Omitted = no work queue resource in the descriptor at all**, which is what every run before Experiment 4 did |
| `--wq-limit <N>` | how many concurrent stream-ordered workloads to declare to the driver, as a **hint** it consults when assigning queues. Not a stream limit. Needs `--allocation greenctx`. Distinct from `CUDA_DEVICE_MAX_CONNECTIONS`, which bounds how many queues exist and is read from the environment (§4.6) |
| `--submit-threads <N>` | how many threads submit work to the GPU. `1` (default) drives every network from a single thread, keeping one inference in flight per network. `2` is one thread per network, the naive application: it is what the two-thread measurements of Experiments 2, 4 and 5 used, and `utility/sweep.sh` states it explicitly wherever that is what it means. Same engines, same partitions, same protocol, so the two differ in the submitter and nothing else |

Outputs: `<prefix>_latencies.csv` (one row per inference:
`phase,index,image,prep_ms,wall_ms,gpu_ms`), `<prefix>_stats.csv` (one row per
metric). With several networks each gets `<prefix>_<label>_…` so files never
collide.

---

## 3. Methodology, and why each choice

### 3.1 Fixed preprocessed inputs, not the dataset

**Decision:** for the performance characterisation, decode and preprocess a
small fixed set of images **once**, then cycle those tensors.

**Why:** image content is not a variable of interest — a conv net executes the
same kernels whatever the input values are — but the *CPU work* around the
inference is not neutral: it perturbs the GPU. Measured on this board, with
clocks pinned at 1.3 GHz:

| | GPU compute (median) |
|---|---|
| fixed inputs (no CPU work in the loop) | **0.537 ms** |
| preprocessing inside the loop (end-to-end) | **0.919 ms** (+71%) |

So the pipeline, not the model, was dominating the measurement. Fixed inputs
also make every configuration see **byte-identical inputs**.

**Sample count vs. image count:** the statistical samples are the
*inferences*, not the distinct images — the loop cycles the set as often as
needed. The set size only controls input *diversity*. 32 evenly spaced images
cost 18 MB of pinned memory and cover the dataset's size distribution. (The
first N images are a bad sample: they are nearly all 500×375.)

### 3.2 Warm-up: 200 inferences, not measured

The GPU is not born fast: clocks ramp, caches fill, the CUDA context settles.
The first inference measured ~37 ms against a ~0.54 ms steady state. Warm-up
rows are written to the CSV (so the curve is visible) but excluded from the
statistics.

Warm-up also runs **concurrently** in the two-network case, so both networks
reach a *contended* steady state before the timed phase begins.

### 3.3 The stop rule: ≥ N inferences **AND** ≥ S seconds

```
stop when  n >= minInferences  AND  elapsed >= minDurationS
```

Both bounds are needed and they answer different things: the count gives the
percentiles enough samples; the duration gives every configuration the same
*observation window* (a fast network reaches 5000 inferences in ~3 s, which is
too short to be comparable with a slow one). A bound of `0` disables it; both
at `0` (or `--one-pass`) means one pass, which is the development mode.

**How many samples are enough?** Measured convergence for MobileNet V2:

| N | median | std | p99 | tail (p99−med) | wall |
|---|---|---|---|---|---|
| 100 | 0.5373 | 0.00346 | 0.5495 | 0.0122 | 0.06 s |
| **1000** | **0.5314** | 0.00257 | 0.5386 | 0.0071 | 0.6 s |
| 5000 | 0.5321 | 0.00250 | 0.5393 | 0.0072 | 3.0 s |
| 20000 | 0.5335 | 0.00226 | 0.5401 | 0.0065 | 11.8 s |

At N=100 the p99 degenerates to the maximum (a 2% error on the tail). By
N≈1000 everything is converged. Beyond that the within-run sampling error is
already an order of magnitude **smaller** than the between-run drift
(~0.5–1%), so more samples stop helping — more *runs* do. Hence the protocol's
"5 independent runs": the count bound exists to get p99 out of the noise, and
the repeated runs are what make the result trustworthy.

`make run` therefore uses 1000 inferences (~0.6 s) rather than one pass.

### 3.4 Three timings per image

| Metric | What it is | How |
|---|---|---|
| `gpu_ms` | **pure GPU compute**: only the inference kernels | CUDA events around the kernel launch |
| `wall_ms` | the whole step **taken on the stream**: H2D + infer + D2H, copies included | CUDA events on that network's own stream |
| `prep_ms` | decoding + preprocessing (≈0 in fixed mode) | `std::chrono` |

**Why CUDA events for the GPU number:** a wall clock measures CPU time
including launch overhead and scheduler jitter; an event is a timestamp the GPU
itself records on the stream, so `gpu_ms` is immune to host-side noise. It
also excludes the copies, which is what makes it comparable with trtexec's
"GPU Compute Time".

In fixed mode `prep_ms ≈ 0.0001 ms` **by construction** — the runner prints an
explicit note in the summary so that is not misread as "preprocessing is
instantaneous".

### 3.5 The statistics

`mean`, `median`, `std` (sample, n−1), `p95`, `p99`, `max`, `min`, and
**`tail = p99 − median`**. Percentiles interpolate linearly between sorted
samples — the same convention numpy uses, so the runner and a Python analysis
of the CSV agree. For a real-time system the tail is the number that decides
deadlines, which is why it is reported rather than buried.

`max` is the **observed maximum** of that run: the largest sample it actually
saw. The runner prints it as `obs-max` so it is never read as a worst-case
bound; the CSV column keeps the short name `max` because the analysis and every
file already on disk use it. Across runs it is the one statistic that is **not**
reduced with the median — `aggregate.py:obs_max` takes the maximum over the runs
instead, which is exactly the maximum over every inference measured, because the
maximum of a union is the maximum of the maxima. Taking the median of five
maxima would discard the rare stall the column exists to catch, and a stall
normally spoils one run in five rather than all of them.

### 3.6 Lock the clocks — and check that they are locked

The `nvhost_podgov` governor drops the GPU to **306 MHz** (from 1.3 GHz) when it
goes idle for a moment. GPU compute went from **0.62 ms** (GPU kept busy) to
**2.1 ms** (GPU idle during CPU preprocessing) — a 3.4× inflation from clock
throttling, not from the model. Temps were ~47 °C, so it is not thermal.

- lock with `sudo jetson_clocks`, then verify `min_freq == max_freq`
  (`/sys/class/devfreq/17000000.gpu/`);
- judge by `min == max`, **never by the governor name** — `schedutil` and
  `nvhost_podgov` still appear when the clocks are pinned;
- the lock does **not** survive a reboot;
- **lock before building the engine too** (next section).

---

## 4. Spatial partitioning with green contexts

### 4.1 The workflow

CUDA 13.1+ exposes green contexts through the **runtime** API as an
"execution context" (`cudaExecutionContext_t`). Five steps:

```
1. device SM resource        cudaDeviceGetDevResource
2. split into N groups       cudaDevSmResourceSplit        ← ONE call for all groups
3. finalize a descriptor     cudaDevResourceGenerateDesc
4. create the green context  cudaGreenCtxCreate
5. create streams on it      cudaExecutionCtxStreamCreate
```

Everything launched on such a stream can only use the provisioned SMs. Because
the context is inferred **from the stream you pass**, libraries work with green
contexts as long as they are handed the stream — the CUDA guide says so
explicitly, and it is why `TrtEngine`, `InputSet` and the measurement code
needed **no changes at all**.

### 4.2 Why the resource comes from the *device*

`cudaDeviceGetDevResource` is the documented starting point. The
context/stream variants (`cudaExecutionCtxGetDevResource`,
`cudaStreamGetDevResource`) report what an *existing* context or stream is
bound to — useful for **verification**, and `cudaStreamGetDevResource` supports
SM-type resources only. Obtaining a partition *from a stream* would be
circular: a stream bound to a green context requires the green context to exist
first.

### 4.3 Why the split must be a single call (measured)

```
one call split into 2x8: OK
  group0=8  group1=8  remainder=0  (sum groups=16)
  ctx0 got 8 SMs, ctx1 got 8 SMs

with an 8-SM green ctx alive, the DEVICE now reports 16 SMs   ← the trap
```

`cudaDeviceGetDevResource` **always reports the full device**, even with green
contexts alive, and SM **oversubscription is explicitly permitted**. So N
independent "split off my share" calls are *not* guaranteed to be disjoint —
and the API exposes no SM *identities* to check afterwards (`cudaDevResource`
carries counts only). The documented pattern is therefore one split into N
groups with one descriptor per group. Splitting off one group and re-splitting
the remainder is actively discouraged (the remainder is unstructured and offers
fewer guarantees).

This is why `PartitionPlan` exists and why `makeSmAllocations(kind, percents)`
takes **all** the percentages at once.

### 4.4 Granularity on this SoC

CC 8.x and Tegra: `smCount` must be a **multiple of 2**, minimum partition 2,
co-scheduled alignment 2. Requested percentages are converted to SM counts,
rounded up to the alignment, clamped, and the driver's **actual grant is
queried back** with `cudaExecutionCtxGetDevResource` and reported — the log
records what was provisioned, not what was asked for. A plan that leaves SMs
unassigned prints a warning, because the protocol always intends full capacity.

### 4.5 Provisioning happens at construction

`SmAllocation` has **two** hooks, not three:

| Hook | Used by |
|---|---|
| *construction* | green contexts (provision SMs) and MPS (set `CUDA_MPS_SM_PARTITION` before CUDA initialises) |
| `createStream()` | green contexts (return the bound stream) |
| `enqueue()` | nvtaskset / libsmctrl (per-launch SM mask) — not used yet |

There is deliberately **no `init()` step**: an earlier version had one and it was
called twice in the same process, which silently created a second green context
and leaked the first. Provisioning in the constructor makes "usable as soon as
it exists" an invariant instead of a convention. Two consequences to keep in
mind:

- construction must be the **first CUDA-touching call** in the process;
- a constructor that throws does **not** run the destructor, so the green
  context is acquired last and nothing after it may throw.

### 4.6 Work queues: the other resource a green context can partition

SMs are one provisionable resource; **work queues are the other**. The guide's
warning is why it matters:

> If independent GPU work tasks (e.g., kernels submitted on different CUDA
> streams) map to the same workqueue, a false dependence between these tasks may
> be introduced, which can lead to their serialized execution.

So disjoint SM partitions are not enough to make the work inside them overlap: if
two contexts' streams share a queue, a kernel enqueued later waits for an earlier
one even with every SM free. Experiment 3's `mps_off` is that failure mode at its
most extreme — the two networks land within 0.03 % of each other on latency,
which is what serialisation looks like.

Three knobs, and conflating them is the easy mistake:

| knob | what it is | where it lives |
|---|---|---|
| `sharingScope` | how queues are shared between contexts | the green context's resource descriptor |
| `wqConcurrencyLimit` | how many concurrent stream-ordered workloads we *expect*, declared as a **hint** used when the driver assigns queues. Not a stream limit: `--wq-limit 1` does not mean "one stream" | same descriptor |
| `CUDA_DEVICE_MAX_CONNECTIONS` | how many queue connections exist **at all** | the environment, read before CUDA initialises |

`--wq-scope` and `--wq-limit` set the first two; the third is set by whoever
launches the process (`utility/sweep.sh`).

**Measured on this board (CUDA 13.2), because the defaults are not what one would
guess:**

| condition | device-reported `wqConcurrencyLimit` | scope |
|---|---|---|
| one process, no MPS | **8** (the documented `CUDA_DEVICE_MAX_CONNECTIONS` default) | `device_ctx` |
| `CUDA_DEVICE_MAX_CONNECTIONS=1` / `=32` | 1 / 32 (32 is the documented maximum) | `device_ctx` |
| **under MPS**, nothing set | **2** | `device_ctx` |
| under MPS, `CUDA_DEVICE_MAX_CONNECTIONS=8` | 8 | `device_ctx` |

The third row is the one that bites: **the two concurrent experiments were run at
different queue budgets without anyone choosing it** — 8 in the one-process
green-context runs, 2 under MPS — so their dispersion differs in three things at
once (the mechanism, the process model, and the budget). Unpicking that is what
the `concurrent-wq` family exists for.

**What the read-back proves, and what it does not.**
`cudaExecutionCtxGetDevResource` returns what a live context was provisioned with,
and every run logs it next to the granted SM count, so the log says which knobs
were applied. It does **not** prove a particular queue assignment was achieved: the
request is a hint, and *"the user has no direct control over the specific work
queues that may be used under the hood"*. The guide is explicit that even with
disjoint SMs **and** balanced queues, concurrency is not guaranteed — the
techniques "remove factors which can prevent concurrent execution" rather than
promising overlap.

### 4.7 What green contexts do *not* do

- They partition **SMs and (optionally) work queues — not memory bandwidth or
  L2**. NVIDIA's own framing of the goal is to *"reduce, or better control,
  interference due to use of common resources"*, not to provide complete
  isolation (that is MIG's job, which needs datacenter GPUs).
- Disjoint SM partitions still do **not guarantee** that kernels run
  concurrently or make forward progress; concurrency can also be blocked by
  work-queue sharing.
- Two documented cases where a workload may use **more** SMs than provisioned
  (never fewer): MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` on Volta+, and CUDA
  Dynamic Parallelism on CC 9.x.
- SM *identities* are not exposed by the API; Nsight Compute's session page
  shows the provisioned-resource bitmask if visual confirmation of disjointness
  is needed.

### 4.8 CUDA graphs do not interfere here

A CUDA graph's *launch stream* does not determine the SMs; each kernel node
carries the execution context it was created with. In standard TensorRT, graph
capture is **manual and opt-in** (the user wraps `enqueueV3` in
`cudaStreamBeginCapture`/`cudaStreamEndCapture`); it is not something TRT does
behind `enqueueV3`, and this runner never captures a graph. (TensorRT-RTX has
automatic capture, but that is not what is installed.) So the green-context
stream genuinely governs the work.

---

## 5. Architecture decisions and the alternatives rejected

| Decision | Why | Alternative rejected |
|---|---|---|
| `Network` owns its measurement (no separate `Experiment` class) | measuring is something a Network *does*; a separate object adds an ownership edge that carries no information | `Network` owning an `Experiment` — felt arbitrary, and it also spent the word "experiment", which belongs to the thesis's campaigns |
| `Run` (not `Session`) for one process invocation | matches the protocol's vocabulary ("5 independent runs per configuration") and `make run` | `Session` (synonym of "experiment", confusing); `Benchmark` (collides with the results directory) |
| `Network` is the subject *and* its measurement | it is the only thing that can be measured; the engine and inputs are already its parts | separate measurement object |
| One run-level object assembles everything (`Run`) | the orchestration (gate, threads, aggregate) is policy that does not belong in `main` | keeping it in `main` (94 lines of policy in the entry point) |
| In-process threads + `StartGate` for concurrency | exact, identical timed windows in one process; maximum experimental control | two processes (simpler, but the window alignment is only approximate); a loop of runs inside the binary (would violate "restart between runs") |
| One joint split in `makeSmAllocations` | disjointness (see §4.3) | per-network independent splits — **not guaranteed disjoint** |
| Device buffers are engine-owned, host **output** buffer too, but the **input** host buffer is the caller's | the input is the caller's data (the InputSet owns 32 tensors and cycles them); the output is the engine's product, one buffer overwritten per inference | owning both (the old code made the engine own the caller's input, exposed via a mutable pointer) |
| `PinnedHostBuffer` for the input set and the output | `cudaMemcpyAsync` only *stays* asynchronous with page-locked host memory. Measured per H2D+D2H iteration: **0.0419 ms pinned vs 0.0963 ms pageable** (+130%) | plain `std::vector` (would have cost ~9% of `wall_ms`; `gpu_ms` unaffected either way) |
| All CUDA/TRT calls in `gpu.cpp` | one place to audit; the other layers contain zero CUDA calls. `PinnedHostBuffer` exists so lower layers can *own* pinned memory without *calling* CUDA | spreading CUDA across modules |
| `readFile` private inside `TrtEngine` | it only ever loads the engine; it is how the engine loads itself, not a public helper | a generic file utility (a grab-bag) |
| `--csv <prefix>` names all outputs; multi-network runs append `_<label>` | the prefix is a path, so run identity lives in the name, not in a new flag. No `--run-id`/`--tag` needed | dedicated run-id flags (redundant) |
| `config.cpp` holds the CLI | the parser, the usage text and the afield defaults all describe the same thing, so they live together | piping in `main`, where the usage text drifts from the fields |
| `stats.hpp` separated from the run | pure arithmetic; makes it a dependency-free leaf and keeps one percentile convention | leaving it beside the GPU code |
| `--sm-percent` only with the positional form; `--net` carries its own percentage | two clean forms, no third. The code now *rejects* mixing them | silently treating a positional engine as network #1 (undocumented and surprising) |
| `--end-to-end` (not `--dataset`) for the mode switch | `--dataset` is needed for the *directory*; having it mean "mode" while `--dataset-dir` meant "path" was ambiguous | the previous `--dataset`/`--dataset-dir` pair |

---

## 6. Measurements so far

Board: Jetson AGX Orin 64 GB, 16 SMs, CC 8.7, clocks locked at 1.3005 GHz.
**Five runs per configuration, 30 s windows, one fresh process each**; every
number is the median of the five run medians and the run-to-run spread is under
1 % throughout. Raw data in `$DATA/bench/`, derived tables in
`$DATA/bench/aggregate/tables.txt`.

### 6.1 Experiment 1 — one network, varying the SM budget (medians, ms)

| SMs | MobileNet V2 | ResNet-50 v2 | linear model (16/N) |
|---|---|---|---|
| 16 (`whole`) | 0.534 | 1.356 | 1.00× |
| 16 (`gc100`) | 0.536 (1.00×) | 1.363 (1.01×) | 1.00× |
| 12 (`gc75`) | 0.584 (1.09×) | 1.611 (1.19×) | 1.33× |
| 8 (`gc50`) | 0.675 (1.26×) | 1.771 (1.31×) | 2.00× |
| 4 (`gc25`) | 1.031 (1.93×) | 2.972 (2.19×) | 4.00× |

`greenctx 100%` matches `whole` within 0.5%, so a full-size green context is a
valid control. **The scaling is strongly sublinear**: halving the SMs costs
26–31% latency, not 100% — neither model is SM-throughput-bound (consistent
with per-layer tactics and a mix of compute- and memory-bound layers). This is
what makes partitioning attractive: two networks at 50/50 each lose ~30%
latency, so aggregate throughput rises.

### 6.2 Experiment 2 — two networks, disjoint partitions (medians, ms)

| split | SMs mob/res | MobileNet | ResNet | aggregate |
|---|---|---|---|---|
| 50 / 50 | 8 / 8 | 0.909 | 2.021 | 1442 /s |
| 75 / 25 | 12 / 4 | 0.713 | 3.309 | 1484 /s |
| 25 / 75 | 4 / 12 | 1.286 | 1.813 | 1269 /s |
| **baseline, no partitioning** | 16 shared | 0.892 | 1.803 | **1565 /s** |

### 6.3 Residual interference: concurrent ÷ isolated, same SM count

| | MobileNet V2 | ResNet-50 v2 |
|---|---|---|
| 8 SMs (50/50) | **+34.7%** | **+14.1%** |
| 12 SMs (75/25) | +22.2% | +11.3% |
| 4 SMs (25/75) | +24.7% | +12.6% |
| unpartitioned baseline | +67.1% | +33.0% |

**The point of the experiment:** partitioning does not eliminate interference —
every disjoint configuration is still 11–36% slower than isolated — but it
halves the cost of the unpartitioned baseline. The residual is the interference
that SM partitioning cannot remove (memory bandwidth, L2, work queues), which
is exactly the gap the "green contexts are not MIG" caveat predicts.

**Jitter is where it hurts most.** At 50/50, MobileNet's `std` goes from
0.0025 ms isolated to **0.1013 ms** under co-execution (40.5×), with
`p99 − median` rising from 0.0066 to **0.2734 ms**, while ResNet stays tight
(0.0043 → 0.0178). At 75/25 MobileNet's tail multiplier reaches **56.9×**. Short
kernels are far more exposed to interference jitter, which is a strong argument
for why mean latency is the wrong headline metric.

**Throughput vs predictability:** the unpartitioned baseline has the highest
aggregate throughput (1565 /s) but the worst per-network latency and jitter.
The 75/25 split beats 50/50 on aggregate (1484 vs 1442) because the faster
network gets more SMs — so the ratio itself is a tuning knob.

### 6.4 Experiment 3 — two processes, MPS (medians, ms)

| configuration | MobileNet | ResNet | aggregate |
|---|---|---|---|
| `mps_single` (control: one process under MPS) | 0.534 | — | 1683 /s |
| `mps_off` (two processes, **no** MPS) | 2.076 | 2.077 | 934 /s |
| `mps_default` (two processes, MPS) | 0.877 | 1.825 | 1585 /s |
| ATP 50/50 · 75/25 · 25/75 | 1.369 · 0.759 · 1.459 | 2.262 · 3.505 · 1.870 | 1129 · 1491 · 1170 |
| static 50/50 · 75/25 · 25/75 | 0.841 · 0.703 · 1.273 | 2.058 · 3.336 · 1.822 | 1561 · **1591** · 1275 |

- **MPS's per-client cost is zero.** `mps_single` is 0.5342 against the isolated
  `whole` run's 0.5342, identical to four decimal places, and the 5-run ranges
  overlap.
- **Without MPS the process model costs ~40 % of the throughput.** `mps_off`
  serialises the two networks almost perfectly — they land 0.03 % apart, against
  a 2.5× ratio in isolation.
- **Static SM partitions beat ATP at every ratio** (38.6 % faster for MobileNet at
  50/50, +38 % aggregate) *and* are tighter on every dispersion metric, so static
  wins both axes rather than trading one for the other.
- **Static and a green context agree on the median to within ±2 %** — five of six
  comparisons sit inside the process-model bound (§9.4). That agreement is what
  the work queue experiment exists to explain rather than merely record.

### 6.5 Dispersion, and the observed maximum

Partitioning shows up in the tail, not the median. MobileNet's `p99 − median`
goes from 0.007 ms isolated to 0.273 ms at 50/50 and 0.404 ms at 75/25 — a 40–57×
multiplier on the same GPU at the same SM budgets — while ResNet's moves by only
3–5×. `mps_off` is the instructive outlier: its `std` is 16× isolated while its
*tail* is 0.9×, because serialisation offsets the distribution instead of
fattening it.

The **observed maximum** (§3.5) is a different view, and the p99 does not bound
it: it exceeds the p99 by up to 4.0× in the isolated MobileNet runs (2.7468 ms at
8 SMs against a 0.6816 ms p99), and `mps_default` holds the worst single event in
the campaign — 4.6940 ms for ResNet, 3.09× isolated. Per-configuration numbers
with their `× iso` multipliers are in `$DATA/bench/aggregate/tables.txt` (the
DISPERSION block), which is also where the thesis tables come from.

---

One caveat on that column, found by checking it rather than by theory. The
observed maximum is a single sample, so it takes *any* system event, not only a
GPU-related one. Run 4 of `concurrent-wq/devctx_8_50_50` has a tail of 2.4 ms on
**both** networks where its four siblings sit at 0.03 and 0.2 ms --- one event
hitting two networks at once, with no profiler running. The medians and the
median-across-runs tails are unaffected by it, which is the aggregation doing its
job; `obs-max` is not, and that is inherent to the statistic rather than a bug.

## 7. Things learned the hard way

1. **Engines are not portable across TensorRT versions.** An engine built with
   a different TRT fails to load (`deserializeCudaEngine` returns nullptr,
   message: *"old deserialization call on a newer plan file"*). Always build on
   the target machine.
2. **Build the engine with the clocks locked.** TensorRT profiles and selects
   tactics *at build time under the current clock*; an engine built while the
   GPU was throttled measured **0.619 ms** against **0.529 ms** for the same
   ONNX built at full clock (~15% slower, permanently). The symptom was also
   `[TRT] Using an engine plan file across different models of devices…`.
3. **Check every pointer.** `deserializeCudaEngine` and
   `createExecutionContext` can return nullptr; without a check you get a
   segfault with no message.
4. **`enqueueV3` + `setTensorAddress` on your own stream.** The old
   `executeV2` style expects a bindings array and does not mix with
   `setTensorAddress` (error: *"bindings != nullptr"*). The default stream adds
   hidden synchronisations.
5. **Pinned memory for async copies** (see §5 for the measured cost of
   getting it wrong).
6. **CPU preprocessing starves the GPU** (clock collapse, §3.6) **and**
   interferes even with clocks pinned** (§3.1). Two distinct channels.
7. **`--fp16` is a hint, not a constraint.** TensorRT picks the fastest tactic
   per layer, which may be FP32. Measured on MobileNet V2's detailed layer
   dump: **135 FP16 vs 2 FP32** mentions — the FP32 ones being the engine's
   **I/O bindings** (the engine reports `input FLOAT [1,3,224,224]`,
   `output FLOAT [1,1000]`; TRT keeps I/O in FP32 unless told otherwise). Forcing
   FP16 everywhere with
   `--precisionConstraints=obey --layerPrecisions=*:fp16` gave
   0.5310 ms vs 0.5391 ms (≈1.5% faster), i.e. the layers were already FP16 in
   practice. Note also that **TF32 is enabled by default** in TensorRT for FP32
   layers (`trtexec` prints a notice; `--noTF32` disables it) — relevant when
   comparing precisions.
   To see per-layer precisions, the engine must be built with
   `--profilingVerbosity=detailed`; otherwise the inspector returns names only
   (*"The profiling verbosity was set to ProfilingKLAYER_NAMES_ONLY…"*).
8. **trtexec flag traps:** `--warmUp=N` is **milliseconds**, not iterations;
   `--duration` takes precedence over `--iterations`; use
   `--duration=0 --iterations=N` for an exact count.
9. **`cudaDeviceGetDevResource` always returns the full device**, whatever
   green contexts exist (§4.3). This is the single most surprising API
   behaviour found, and it is the reason for the joint split.
10. **Granularity is 2 SMs on this SoC**, and the granted count may differ from
    the request — always query it back.
11. **A constructor that throws does not run the destructor**: acquire GPU
    resources last.
12. **The dataset directory held 50,001 entries**: 50,000 JPEGs plus the
    original `ILSVRC2012_img_val.tar` (6.3 GB, later removed to free the disk,
    which was at 100% and blocking compiles). The "50,002 files" count included
    the directory itself.
13. **ImageNet input contract** (getting it wrong silently produces
    confident garbage): resize shortest side to 256, centre-crop 224×224,
    scale to [0,1], normalise per channel with mean
    `[0.485,0.456,0.406]` / std `[0.229,0.224,0.225]` (the PyTorch training
    constants), layout **NCHW**, and remember OpenCV loads **BGR** while the
    network wants **RGB**.
14. **`input` is the engine's I/O name** here; the runner asks the engine
    (`getIOTensorName` / `getNbIOTensors`) rather than hard-coding it.

---

- **The one-process MobileNet tail is host submission time.** Measured directly,
  by timing the host inside `enqueueV3` for a few runs. The probe was temporary
  and has been removed, but the numbers are worth keeping: in the unpinned
  configuration the slowest 1 % of inferences have a submit time of
  1.05--1.10 ms against a typical 0.33--0.34 ms, a factor of **3.1--3.3**, and
  84--90 % of them are the *same* inferences as the slowest submissions. With the
  two threads pinned to one core the same ratio is **1.00** and the overlap is at
  chance, and the tail is seven times smaller. So the tail is not the GPU being
  slow: it is the host taking longer to submit, with the queue already drained
  because every iteration ends in a `cudaStreamSynchronize`. ResNet shows no such
  correlation, which is why its tail stays thin.
- **The disk is a shared resource the guard does not watch.** See the note at the
  end of §9.3. Same class of hazard as another user on the GPU, and the same
  remedy: remove the dependency rather than measure around it.

## 8. Environment

| Item | Value |
|---|---|
| Board | NVIDIA Jetson AGX Orin Developer Kit 64 GB, aarch64, CC 8.7 |
| SMs | 16 |
| GPU clock | 1.3005 GHz max, 306 MHz min (governor `nvhost_podgov`) |
| Memory | 64 GB LPDDR5, ~204.8 GB/s, shared CPU/GPU |
| TensorRT | 10.16.2 |
| CUDA | 13.2 |
| OpenCV | 4.8.0 (CUDA off) |
| JetPack / L4T | R39 |
| Pinned-memory limit | ~8 GB (32 inputs = 18 MB, no problem) |
| Models | MobileNet V2-12, ResNet-50 v2-7 (ONNX model zoo), FP16, fixed 1×3×224×224 |

---

## 9. The measurement protocol: `utility/sweep.sh` + `utility/aggregate.py`

Two scripts, deliberately separate: `utility/sweep.sh` only *measures*,
`aggregate.py` only *analyses*. The analysis is therefore repeatable without
re-running two hours of GPU work.

### 9.1 The run matrix

| Experiment | Configurations | Runs | Allocation |
|---|---|---|---|
| 1 `isolated-scaling` | `whole`, `gc100`, `gc75`, `gc50`, `gc25` × 2 networks | 10 | one network, alone |
| 2 `concurrent-greenctx` | `baseline`, `50_50`, `75_25`, `25_75` | 4 | two networks, **one process**, disjoint green contexts |
| 3 `concurrent-mps` | `mps_single`, `mps_off`, `mps_default`, `mps_atp_*`, `mps_static_*` | 6–9 | two networks, **two processes**, MPS |
| 4 `concurrent-wq` | 5 arms × `50_50`/`75_25`, plus `unpart_8`/`unpart_2` | 12 | Experiment 2's setup with the work queue configuration pinned (§4.6) |

**5 runs each.** Everything selected is 32 configurations / 160 runs, a bit over
2 hours. Experiment 3's `mps_static_*` appear only with `--static-chunks <n>`,
because the chunk count has to come off `lspart` on the real board first.

The work queue family is deliberately **not** in the default selection: it is 60
runs, and a routine re-run should not start it by accident. Ask for it:

    ./utility/sweep.sh --exp wq

### 9.2 Why these choices

- **`gc100` is not redundant with `whole`.** It is the control that proves the
green-context *mechanism* costs nothing (measured 0.536 vs 0.534 ms), so a
slowdown at 75 % cannot be blamed on the mechanism itself.
- **Experiment 2 uses a MIXED pair** (MobileNet + ResNet), not mob-mob/res-res.
Three reasons: (i) a scheduler co-locates *different* workloads, so the mixed
pair is the deployment actually studied; (ii) one mixed sweep yields each
network's co-run latency at 4, 8 and 12 SMs — exactly the points Experiment 1
measures in isolation; (iii) the **asymmetry is itself a result** (at 50/50 the
residual is +34.7 % for MobileNet against +14.1 % for ResNet) — the lighter model
suffers more because it has less work to hide the interference behind. Same-pairs
would answer a different question (how residual interference scales with the
co-tenant's weight) and are an extension, not the core.
- **Honest caveat for the thesis:** in a mixed pair the co-tenant's SM count
moves with the split (at 75/25 MobileNet's co-tenant has only 4 SMs), so the
three splits answer *"which ratio is best?"*, not *"interference at a fixed k
SMs"*. 50/50 is the only fully symmetric point.
- **Run index is the OUTER loop**, configurations the inner one, so thermal and
machine-state drift is spread evenly over configurations instead of favouring
whichever one ran first.
- **The sweep is resumable**: a run whose `_stats.csv` exists is skipped. An
interrupted sweep is restarted, not repeated. (Because of this, smoke results
go to a separate directory — a smoke run left under a real name would be
skipped and silently kept.)

### 9.3 The GPU cannot be locked — so it is measured

The shared board offers **no way to exclude another tenant**:

```
nvidia-smi --help :  -c/--compute-mode  0/DEFAULT … 3/EXCLUSIVE_PROCESS   (advertised)
nvidia-smi -q     :  Compute Mode : N/A                                 (not supported)
                     Persistence Mode : N/A
                     MIG Mode : not supported on Orin
```

The Tegra driver reports no compute-mode support (and it would need root);
there is no MIG; and every user runs as the **same `nvidia` account**, so there
is no permission boundary either. `nvidia-smi --query-compute-apps` returns
`N/A` on Tegra, `/sys/devices/gpu.0/load` does not exist, and `fuser /dev/nvidia*`
reports nothing.

So the board cannot be locked — it can only be **checked**:

| Layer | What | Cost |
|---|---|---|
| Preflight | clock check (`min == max`), then a calibration | ~2 s |
| Device-holder scan, before every run | any process holding `/dev/nvgpu/*`, `/dev/nvmap` or `/dev/dri/renderD*` → abort | ms |
| Sentinel before every run | a fixed 500-inference MobileNet run; median must match the 0.535 ms reference within ±15 %, else abort | ~1.3 s × 70 |
| Post-hoc | `aggregate.py` flags any run whose median is > 25 % above its 4 siblings | free |

The device scan is **not** a name heuristic. A CUDA process on this board opens
`/dev/nvgpu/igpu0/ctrl` and `/dev/nvmap` (and the DRM render node) and keeps them open
for its lifetime, whereas a compiler opens none of them. Two traps it has to avoid:
`/dev/nvidia*` exists here but CUDA never holds it, so checking for it matches
nothing (that was the first attempt); and the MPS control daemon holds these nodes
for its whole life without doing GPU work, so it is excluded.

**An earlier version matched on command-line text, and it failed in real use.** It
flagged a colleague's *compiler* — `nvcc`/`cicc`, invoked on a source file whose
*path* contained "green_ctx" — and aborted a sweep that was 113 runs in. Compiling
does not touch the GPU. Names guess; file descriptors do not.

Observed working: a co-tenant's `green_ctx_benchmark` inflated the median from
0.53 to 2.83 ms, and the sentinel aborted the sweep with **+42 %** before any
time was wasted. Observed limitation: once, the sentinel read a clean 0.5392 ms
*while* a co-tenant was running — 500 inferences can fall between the other
job's iterations. **The sentinel is a tripwire, not a guarantee**; the cross-run
consistency check is the real safety net. The honest fix is social:
coordinate with the other user (and note the co-tenant is running *the same
experiment* on `--sm 4/6/8`, which is worth mentioning in the thesis).

**The disk is shared too, and the guard does not watch it.** A flush that ends up
queued behind another writer can block for hundreds of milliseconds, and because
the flush happens inside the measured window the GPU idles while it waits. One
traced run shows `writev` calls of **385 ms and 316 ms**, one per network, with
both green contexts idle across them; the profiler was the other writer there,
but the lesson is general. The sweep now writes its CSVs to `/dev/shm` and moves
them into the results tree when the run finishes, which removes the dependency
and changes nothing downstream. Measured effect on clean runs: none --- tail
0.3659 ms with the CSV on disk against 0.3644 ms in RAM, same day, same
configuration.

### 9.4 Aggregation: the run is the unit of replication

The single decision that drives `aggregate.py`:

- **Level 1 (within a run)** — the runner has already summarised each run
(`_stats.csv`: median/mean/p99/std over that run's ~10k–60k inferences).
- **Level 2 (across runs)** — the point estimate is the **median of the 5
run-level values**, and the run-to-run spread is reported as **[min, max]**.

Pooling all ~300k raw inferences into one vector and taking its p99 — the
tempting alternative — is the classic **pseudo-replication** error: it treats
correlated observations from only 5 processes as independent and understates
the uncertainty several-fold. The 5 fresh processes exist precisely to provide
a between-run error bar, so the aggregation must expose it.

The results are a tree: one folder per experiment, one per configuration, with a
run's files inside it. The point is that the file name only has to carry the run
number — the directory already says which experiment and which configuration.

```
$DATA/bench/
├── isolated-scaling/     one network, alone, varying the SM budget
│     mobilenet_whole/r1_latencies.csv   r1_stats.csv   r1.log    (10 configs)
├── concurrent-greenctx/  two networks, ONE process, disjoint green contexts
│     50_50/r3_latencies.csv   r3_mobilenet_stats.csv   …       (4 configs)
├── concurrent-mps/       two networks, TWO processes, MPS
│     mps_static_50_50/r3_mobilenet_stats.csv   …               (9 configs)
└── aggregate/            <exp>.csv   summary.csv   tables.txt
```

The folder names say what each experiment *is* rather than which number it has
in the thesis, because the numbering can move as the text is written and a
directory called `exp1` tells a reader nothing. The analysis code keeps short
internal keys and maps them once, in `aggregate.py`'s `EXP_DIR`.

Data and derived numbers never mix: measurements live in the tree, everything
generated from them lives in `aggregate/`.

Outputs, all in `$DATA/bench/aggregate/`:

| File | Content |
|---|---|
| `<exp>.csv` | per-run detail, tidy/long, **one file per experiment**: one row per (config, network, run, metric) — for plotting. The experiment stays a column, so the files concatenate back into the single table this used to be |
| `summary.csv` | across-run point estimate and spread per config/network |
| `tables.txt` | the derived tables (also printed) |
| `<run>.log` | full stdout of each run (holds the granted SM counts) |

The tables: **Exp 1** (latency vs SM share, ratio vs `whole`), **Exp 2**
(per network median/p99/std + aggregate throughput), **residual interference**
(concurrent ÷ isolated at the same SM count), **jitter** (`std`, `p99 − median`,
`obs-max`), **Exp 3** (the two-process MPS runs), **DISPERSION** (`std`, `p95`,
`p99`, `tail` and `obs-max`, each with an `× iso` multiplier against the isolated
`whole` run, one row per configuration and network), and **MPS EFFECTS** (each row
pairing a control with a measurement). The `obs-max` column is the observed
maximum over every inference measured, never the median of the per-run maxima.
Derived numbers are computed by script, never by hand.

---

## 10. Experiment 3: MPS — and why it is not a green context

Green Context and MPS are not two implementations of the same idea:

| | Green Context | MPS |
|---|---|---|
| unit | an execution context **inside one process** | a *server* shared by several CUDA **processes** |
| purpose | lightweight partitioning / resource control, usable within a process | co-schedule multiple processes and reduce the usual time-sharing between contexts |
| disjointness | guaranteed only by a **joint** `cudaDevSmResourceSplit` in one process | server-managed partitions, hence exclusive across processes |
| clients | 1 process, N contexts | N processes |

**The consequence that shapes the experiment:** the two mechanisms cannot be run
at the same process count. Two processes each creating their own green context
are *not* guaranteed disjoint (§4.3), so Experiment 2 must stay in-process;
MPS exists precisely for the multi-process case. A GC-vs-MPS comparison
therefore carries a **process-model** difference — which is why Experiment 3
*measures* that difference (`mps_off` = two plain processes) instead of
assuming it away.

### 10.1 The three phases, and what each buys

| phase | configurations | what it buys |
|---|---|---|
| 1 | `mps_single`, `mps_off`, `mps_default` | the minimal comparison, plus the two controls that make it readable |
| 2 | `mps_atp_50_50`, `mps_atp_75_25`, `mps_atp_25_75` | whether active-thread-percentage is a substitute for spatial partitioning (**it is not**: ATP limits the fraction of SM *thread slots*, not which SMs) |
| 3 | `mps_static_50_50`, `…75_25`, `…25_75` | the true MPS analogue of a green context |

`mps_single` (one process under MPS) is Experiment 3's `gc100`: it measures the
**per-client cost** of routing through the MPS server, so that
`mps_default − mps_off` can be read as co-scheduling rather than as
co-scheduling *plus* overhead. The derived rows live in the
`MPS EFFECTS` table, each pairing a control with a measurement.

### 10.2 Alignment: the two processes share no clock

Experiment 2 aligns its threads with an in-process `StartGate`. Two processes
cannot: there is no shared memory to rendezvous in, so each starts its timed
phase when **its own** warm-up ends. The skew is the difference in pre-timed
work — 200 inferences × (0.53 ms MobileNet, 1.36 ms ResNet) ≈ **0.16 s out of
30 s, i.e. ~0.5 %**. A `--start-at <monotonic deadline>` would make it exact and
is the planned refinement; it is not needed to detect an effect large enough to
matter. The caveat is printed on the Experiment 3 table so it cannot be
forgotten.

### 10.3 Static SM partitioning: what the installed stack supports

**MPS is supported on Jetson AGX Orin from CUDA 12.5 / JetPack 6.1 onwards**
(this board: CUDA 13.2, JetPack R39, and the daemon was confirmed to start and
answer commands). It was **not** supported on Tegra before that, which is why
older forum answers say the opposite. Tegra exceptions worth stating in the
thesis:

- GPU **compute modes are not supported** on Tegra (`nvidia-smi` reports
  `Compute Mode : N/A`) — the same reason `EXCLUSIVE_PROCESS` cannot be used to
  lock the board (§9.3);
- MPS **client termination, CUDA IPC and cooperative launches** are not
  supported with MPS on Tegra;
- **one MPS server can only be active for one user at a time**. On this board
  everyone shares the `nvidia` uid, so a private `CUDA_MPS_PIPE_DIRECTORY` is
  what keeps our server from being joined by someone else's client (§10.4).

The board exposes the **MPS v2 interface**:

```
nvidia-cuda-mps-control -d [-S]              -S enables static partitioning mode
sm_partition add <device UUID> <n_chunks>
sm_partition rm  <device UUID> <partition ID>
lspart
```

**A chunk is 2 SMs on an integrated GPU** (Orin), on any architecture; it is
4 SMs for Ampere dGPUs under MPS v2 and 8 for Hopper+. So this 16-SM device
exposes **8 chunks**, which means **50/50, 75/25 and 25/75 are all exactly
expressible — at the same 2-SM granularity as a green context.** That is a much
fairer comparison than expected; had a chunk been 8 SMs, 75/25 would have been
impossible on 16 SMs.

`lspart`'s layout (the GPU and the partition are **separate columns** — the
`GPU-<uuid>/<partition>` identifier a client needs does *not* appear as such):

```
GPU          Partition  free used  free used clients
                           chunk chunk  SM   SM
GPU-74d43ed3 -           0   10     0   80 -
GPU-74d43ed3 Dx4AAAA...  -    5     -   40 Yes
```

Two traps the parser handles: the header spans **two** lines (the second,
`chunk chunk SM SM`, otherwise looks like a partition row), and — because we
query `lspart` *before* the clients attach — a freshly created partition may
have its size in `free` rather than `used`, so the parser sums both columns.
(The documented example has clients attached, which is why it only ever shows
`used` filled.)

Other established facts:

- a client joins a partition by setting **`CUDA_MPS_SM_PARTITION`** in its
  environment *before creating a CUDA context*, in the form
  `GPU-<full-uuid>/<partition-id>` (the full UUID from `nvidia-smi -L` joined to
  the partition string from `lspart`);
- if static mode is on and that variable is missing, **context creation
  fails** (`CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION`). This is the most
  reassuring property of phase 3: a misconfigured run **dies loudly** instead of
  silently measuring an unpartitioned GPU (contrast the *silent* green-context
  oversubscription trap in §4.3);
- partitions must be created **before** any client connects — the driver rejects
  a static request once the device has active clients;
- the SMs in a partition are exclusive to its clients and cannot be borrowed,
  and over-allocation is refused;
- **when static partitioning is enabled, active-thread-percentage is ignored**
  — ATP and static partitions are mutually exclusive, which is the mechanism
  behind the claim that ATP is *not* a substitute for spatial partitioning;
- static partitioning cannot be combined with locality domains (Blackwell+,
  so irrelevant here);
- requires Ampere or newer (Orin is CC 8.7). The `sm_partition`/`lspart`
  spelling is the v2 interface; MPS v3 uses a `sm-partition` module plus a TOML
  configuration file, and partition-id encodings are **not** portable between
  the two.

### 10.4 Two operational traps

- **`quit` terminates the MPS daemon.** It is a teardown command, never a way
  to end a probe: using it in a probe loop silently kills the daemon, and the
  next command reports *"Cannot find MPS control daemon process"*. The driver
  uses it only to shut down.
- **The pipe directory is the isolation boundary.** `utility/sweep.sh` uses a private
  `CUDA_MPS_PIPE_DIRECTORY` (`/tmp/mps-$USER`), which makes this daemon
  invisible to every other user of the shared board: a client joins a server
  only if it was told about the same directory. That is what makes running MPS
  on a machine other people are using safe. The daemon is torn down by an
  `EXIT` trap, so an aborted sweep cannot leave it running.

---

## 11. Open items

- **The work queue experiment** (§4.6, `concurrent-wq`) is staged and
  smoke-tested but not yet swept: `--exp wq`, 60 runs. It is the current
  question — whether the MobileNet tail difference between Experiments 2 and 3 is
  a work queue configuration artefact, and whether the two experiments differ in
  the queue budget as well as in the mechanism and the process model.
- **The process-model bound has not been recomputed for the dispersion.** §6.2
  bounds the process model at ±2 % on the *median*; the same two unpartitioned
  rows differ by 3.03× on MobileNet's *tail*, which is the number the work queue
  question turns on. `unpart_8` vs `unpart_2` in the new family does it at equal
  queue budget.
- **`--start-at`** is still unimplemented: the two-process runs are aligned to
  ~0.5 % by warm-up symmetry rather than exactly (§10.2).
- **CPU affinity.** The two concurrent networks run in two threads; pinning them
  to distinct cores would remove the one confound in-process concurrency adds.
- **Accuracy.** Not implemented. Needs the ImageNet ground-truth labels
  (`ILSVRC2012_validation_ground_truth.txt`, distinct from the class-name file
  already present), and is the natural companion to the end-to-end mode.
- **The auxiliary-stream census** — how many streams TensorRT actually adopts per
  engine — is the prerequisite for choosing a defensible `wqConcurrencyLimit`
  instead of guessing one, and for the multi-branch networks (Inception-v3, an
  FPN detector) that would exercise it.
- **nvtaskset / libsmctrl** as alternative mechanisms behind the same
  `SmAllocation` seam, for the comparison the thesis promises.
- **Nsight Compute** session page for a visual check of the provisioned-SM
  bitmask (the API exposes counts only).

## Reproducing a result

Everything reported in the thesis comes from the runs in `$DATA/bench/`, and the chain
from those runs to a number on a page is three commands:

    ./utility/sweep.sh [--exp <family>]     # measure: bench/<experiment>/<config>/r<k>_{latencies,stats}.csv
    python3 utility/aggregate.py            # reduce:  bench/aggregate/, one csv per experiment + summary.csv + tables.txt
    python3 utility/tex-tables.py           # render:  the LaTeX tables for Experiment 5, on stdout

Each step reads only what the previous one wrote, so any step can be re-run alone.
`aggregate.py` reads the `_stats.csv` files each run writes; `tex-tables.py` reads
the same files through the same module, so a printed table cannot disagree with the
aggregate beside it.

`--dry-run` prints every command `utility/sweep.sh` would run without running any of it,
and the `--exp` filter selects a family by substring (`isolated`, `greenctx`, `mps`,
`wq`, `aux`, `1t`).

What is deliberately NOT in this repository, because it is data rather than code:
the TensorRT engines, the ImageNet subset and the results are on the NVMe, in `$DATA`
— `models/`, `dataset/`, `labels/` and `bench/`. Not on the root filesystem, which is
small and shared, and not in this repository, which stays code. They do not need to
follow the method, but the numbers will only match with the same board, the same
engines, the same images and the same protocol.

**Not yet scripted:** the LaTeX for Experiments 1 to 4. Those tables were
transcribed from `$DATA/bench/aggregate/tables.txt` and verified against it, but only
Experiment 5 is rendered by `tex-tables.py`. The numbers are reproducible either
way, because `tables.txt` is generated from the runs.

## The models

The engines themselves live in `$DATA/models`, because they are data.
What belongs here is how they were made, which until now was recorded nowhere.

```
$DATA/models/<model>/          one folder per network: the source graph, an .onnx plus any
                            data file it needs, and its labels if it ships them
$DATA/models/<model>/engine/   every engine build of that network, with trtexec's own log
                            beside each one
```

`trtexec` is not on `$PATH`; the full path is `/usr/src/tensorrt/bin/trtexec`.

What is there now: `mobilenetv2-12/engine/` and `resnet50-v2-7/engine/` each hold
the default engine and the four auxiliary-stream builds; `inception_v3-onnx-float/`
holds a downloaded graph that has not been built yet, with a 299x299 input and its
weights in a separate `.data` file beside the `.onnx`, which trtexec picks up on its
own.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/mobilenetv2-12/mobilenetv2-12.onnx --fp16 \
    --saveEngine=models/mobilenetv2-12/engine/mobilenetv2-12.engine

/usr/src/tensorrt/bin/trtexec --onnx=models/resnet50-v2-7/resnet50-v2-7.onnx --fp16 \
    --saveEngine=models/resnet50-v2-7/engine/resnet50-v2-7.engine
```

The census builds are the same line plus `--maxAuxStreams=N` for N in 0, 1, 2 and 4,
with a different `--saveEngine`. Their build logs are kept because they *are* the
census result: what the builder decided, and why. MobileNet reports, under every cap,

```
[TRT] [MS] Multi stream is disabled as cannot find an opportunity to leverage it
```

and ResNet, with a cap of at least one,

```
[TRT] [MS] Number of aux streams is 1
```

Two things about `--fp16`. It is permission, not instruction: TensorRT uses half
precision where it judges it fastest and leaves the two I/O bindings in FP32, which
is why a built engine reports 135 FP16 and 2 FP32 layer mentions. And asking for
strict FP16 instead (`--precisionConstraints=obey --layerPrecisions=*:fp16`) measured
0.5310 ms against 0.5391 ms, so the engines used here are the permissive build. A
difference that size is inside run-to-run noise; the point is that it is not a
different engine in any way that matters.

**Engines are not portable across TensorRT versions.** One built by a newer
TensorRT fails to load on an older one — `deserializeCudaEngine` returns null and
the process dies. Always build on the machine that will run it.

### Adding a network

1. Build it the same way, with `--fp16`, and keep `trtexec`'s log beside the engine.
2. Run `utility/auxprobe <engine>` for the auxiliary-stream count, and build the
   0/1/2/4 caps if it reports more than zero. That is the census, and it decides
   whether the declared-concurrency experiments have anything to test on it: a
   network with one stream cannot exercise them.
3. Nothing else is needed. The runner reads the input geometry from the engine, so
   a network whose input is not 224x224 needs no source change, and every log says
   what it found (`... 8 inputs, 3x224x224 in`).
4. The preprocessing assumes ImageNet-style inputs: three channels, square, resize
   the short side to 8/7 of the side, centre-crop. An engine that is not square or
   not three-channel is refused when it is loaded, with a message saying so, rather
   than measured and reported as if it had been.

### Rebuilding them

`engines.mk`, beside this README, rebuilds every engine from its ONNX:

```bash
cd runner
make -f engines.mk            # every engine
make -f engines.mk census     # what each built engine reports
make -f engines.mk clean      # remove the engines, keep the build logs
```

It is a separate file from the main Makefile because it needs the model tree, which
is not in this repository. Two things it documents by being runnable: every build
passes `--fp16`, and a dynamic-shape ONNX must be given `--shapes`, or its symbolic
channel dimension resolves to one and the first convolution fails on a weight-count
mismatch that reads like a corrupt file.

## Reproducing the results section

The engines are rebuilt from the ONNX files first: they are not bit-reproducible across builds,
so results are comparable within a build and never across two.

    make -f engines.mk                 # rebuild every engine (MODELS=$DATA/models)

Then the measurements, the reduction and the tables:

    ./utility/sweep.sh --exp section    # every family the results section renders (83 runs)
    ./utility/sweep.sh --exp mem        # only the memory work (13 runs)
    ./utility/sweep.sh --exp appendix   # the rest, which the history and the tools read (52)
    ./utility/sweep.sh --exp all        # everything
    python3 utility/aggregate.py --out $DATA/bench
    python3 utility/results-tables.py > tables.tex
    python3 utility/split-tables.py tables.tex --out $DATA/bench/aggregate/tables
    python3 utility/overview.py         # $DATA/bench/OVERVIEW.md: every family and its runs

The sweep checks two preconditions before every run and aborts rather than measure: the clocks
locked (`sudo jetson_clocks`) and the GPU quiet. The MPS static arms additionally need
`--static-chunks 8`; they start and stop their own daemon, and the 4 SM co-runner arms use a
colon ratio (50:25) which places three partitions and leaves the last quarter unassigned.

Two stages need root or a profiler and are deliberately outside the sweep, because profiling
perturbs what it measures:

    sudo ./utility/soc-traces.sh itf     # DRAM counters for the two interferer arms
    ./utility/take-traces.sh             # three kernel traces, then exported to sqlite
    python3 utility/itf-trace.py $DATA/scratch/traces/trc   # the per-inference decomposition

`utility/overview.py` writes the same split the tables make: a family feeds the results section
if its aggregate key appears in `utility/results-tables.py`, and everything else is the appendix.
