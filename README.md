# Inference benchmark runner — design, methodology, findings

TensorRT inference benchmarking on a Jetson AGX Orin, built to characterise how
**spatial partitioning of the GPU's SMs** affects latency, jitter and
throughput — first for a single network, then for two networks running
concurrently on disjoint partitions.

This file is the project's memory: every non-obvious choice, the alternative it
was chosen over, and the measurements behind it. It is written to be usable
directly when writing the thesis.

---

## 1. Layout

```
include/
  config.hpp      every parameter (compile-time cfg:: + run-time Options) and
                  the command-line declarations
  gpu.hpp         the GPU layer: Logger, PinnedHostBuffer, SmAllocation/WholeGpu,
                  SmPartition, PartitionPlan, GreenContextAllocation, TrtEngine
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
| `wall_ms` | the whole step: H2D + infer + D2H + sync | `std::chrono` |
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

### 4.6 What green contexts do *not* do

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

### 4.7 CUDA graphs do not interfere here

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
Windows of 3 s, one run each — indicative; the protocol wants 5 runs per
configuration. Run-to-run drift is ~0.5–1%.

### 6.1 Experiment 1 — one network, varying the SM budget (medians, ms)

| SMs | MobileNet V2 | ResNet-50 v2 | linear model (16/N) |
|---|---|---|---|
| 16 (`whole`) | 0.5390 | 1.3607 | 1.00× |
| 16 (gc 100%) | 0.5351 (1.01×) | 1.3564 (1.00×) | 1.00× |
| 12 (gc 75%) | 0.5833 (1.10×) | 1.6027 (1.18×) | 1.33× |
| 8 (gc 50%) | 0.6700 (1.26×) | 1.7723 (1.31×) | 2.00× |
| 4 (gc 25%) | 1.0309 (1.93×) | 2.9727 (2.19×) | 4.00× |

`greenctx 100%` matches `whole` within 0.5%, so a full-size green context is a
valid control. **The scaling is strongly sublinear**: halving the SMs costs
26–31% latency, not 100% — neither model is SM-throughput-bound (consistent
with per-layer tactics and a mix of compute- and memory-bound layers). This is
what makes partitioning attractive: two networks at 50/50 each lose ~30%
latency, so aggregate throughput rises.

### 6.2 Experiment 2 — two networks, disjoint partitions (medians, ms)

| split | SMs mob/res | MobileNet | ResNet | aggregate |
|---|---|---|---|---|
| 50 / 50 | 8 / 8 | 0.9095 | 2.0172 | 1439 /s |
| 75 / 25 | 12 / 4 | 0.7133 | 3.3094 | 1480 /s |
| 25 / 75 | 4 / 12 | 1.2831 | 1.8222 | 1269 /s |
| **baseline, no partitioning** | 16 shared | 0.8664 | 1.8170 | **1587 /s** |

### 6.3 Residual interference: concurrent ÷ isolated, same SM count

| | MobileNet V2 | ResNet-50 v2 |
|---|---|---|
| 8 SMs | **+35.7%** | **+13.8%** |
| 12 SMs | +22.3% | +13.7% |
| 4 SMs | +24.5% | +11.3% |
| unpartitioned baseline | +60.7% | +33.5% |

**The point of the experiment:** partitioning does not eliminate interference —
every disjoint configuration is still 11–36% slower than isolated — but it
halves the cost of the unpartitioned baseline. The residual is the interference
that SM partitioning cannot remove (memory bandwidth, L2, work queues), which
is exactly the gap the "green contexts are not MIG" caveat predicts.

**Jitter is where it hurts most.** At 50/50, MobileNet's `std` went from
0.0028 ms isolated to **0.0944 ms** under co-execution (34×), with
`p99 − median` rising from 0.0071 to **0.2388 ms**, while ResNet stayed tight
(0.0026 → 0.0147). Short kernels are far more exposed to interference jitter,
which is a strong argument for why mean latency is the wrong headline metric.

**Throughput vs predictability:** the unpartitioned baseline has the highest
aggregate throughput (1587 /s) but the worst per-network latency and jitter.
The 75/25 split beats 50/50 on aggregate (1480 vs 1439) because the faster
network gets more SMs — so the ratio itself is a tuning knob.

---

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

## 9. The measurement protocol: `sweep.sh` + `aggregate.py`

Two scripts, deliberately separate: `sweep.sh` only *measures*,
`aggregate.py` only *analyses*. The analysis is therefore repeatable without
re-running 37 minutes of GPU work.

### 9.1 The run matrix

| # | Experiment | Configuration | Allocation |
|---|---|---|---|
| 1–5 | 1 | `whole` (baseline) | whole GPU |
| 6–10 | 1 | `gc100` (control) | green context, 100 % |
| 11–15 | 1 | `gc75` | green context, 75 % |
| 16–20 | 1 | `gc50` | green context, 50 % |
| 21–25 | 1 | `gc25` | green context, 25 % |
| 26–50 | 1 | *the same five* | for ResNet-50 |
| 51–55 | 2 | `baseline` | two networks, **no** partitioning |
| 56–60 | 2 | `50_50` | mobilenet 50 % + resnet 50 % |
| 61–65 | 2 | `75_25` | mobilenet 75 % + resnet 25 % |
| 66–70 | 2 | `25_75` | mobilenet 25 % + resnet 75 % |

**14 configurations × 5 runs = 70 fresh processes**, ≈ 40 minutes.

### 9.2 Why these choices

- **`gc100` is not redundant with `whole`.** It is the control that proves the
green-context *mechanism* costs nothing (measured 0.5351 vs 0.5390 ms), so a
slowdown at 75 % cannot be blamed on the mechanism itself.
- **Experiment 2 uses a MIXED pair** (MobileNet + ResNet), not mob-mob/res-res.
Three reasons: (i) a scheduler co-locates *different* workloads, so the mixed
pair is the deployment actually studied; (ii) one mixed sweep yields each
network's co-run latency at 4, 8 and 12 SMs — exactly the points Experiment 1
measures in isolation; (iii) the **asymmetry is itself a result** (MobileNet
+35.7 %, ResNet +13.8 %) — the lighter model suffers more because it has less
work to hide the interference behind. Same-pairs would answer a different
question (how residual interference scales with the co-tenant's weight) and are
an extension, not the core.
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
bench/
├── isolated-scaling/     one network, alone, varying the SM budget
│     mobilenet_whole/r1_latencies.csv   r1_stats.csv   r1.log    (10 configs)
├── concurrent-greenctx/  two networks, ONE process, disjoint green contexts
│     50_50/r3_latencies.csv   r3_mobilenet_stats.csv   …       (4 configs)
├── concurrent-mps/       two networks, TWO processes, MPS
│     mps_static_50_50/r3_mobilenet_stats.csv   …               (9 configs)
└── aggregate/            raw.csv   summary.csv   tables.txt
```

The folder names say what each experiment *is* rather than which number it has
in the thesis, because the numbering can move as the text is written and a
directory called `exp1` tells a reader nothing. The analysis code keeps short
internal keys and maps them once, in `aggregate.py`'s `EXP_DIR`.

Data and derived numbers never mix: measurements live in the tree, everything
generated from them lives in `aggregate/`.

Outputs, all in `bench/aggregate/`:

| File | Content |
|---|---|
| `raw.csv` | tidy/long: one row per (experiment, config, network, run, metric) — for plotting |
| `summary.csv` | across-run point estimate and spread per config/network |
| `tables.txt` | the derived tables (also printed) |
| `<run>.log` | full stdout of each run (holds the granted SM counts) |

The tables: **Exp 1** (latency vs SM share, ratio vs `whole`), **Exp 2**
(per network median/p99/std + aggregate throughput), **residual interference**
(concurrent ÷ isolated at the same SM count), **jitter** (`std`, `p99 − median`,
`max`), plus **Exp 3** (the two-process MPS runs) and **MPS EFFECTS** (each row
pairing a control with a measurement). Derived numbers are computed by script,
never by hand.

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
- **The pipe directory is the isolation boundary.** `sweep.sh` uses a private
  `CUDA_MPS_PIPE_DIRECTORY` (`/tmp/mps-$USER`), which makes this daemon
  invisible to every other user of the shared board: a client joins a server
  only if it was told about the same directory. That is what makes running MPS
  on a machine other people are using safe. The daemon is torn down by an
  `EXIT` trap, so an aborted sweep cannot leave it running.

---

## 11. Open items

- **Experiment 3 (MPS)** is written but **never executed**: `sweep.sh` supports
  it (`--exp 3`, `--static-chunks <total>` for phase 3), validated by dry run and
  by offline tests of the `lspart` parser against the documented layout. Phase 1
  waits for the board; phase 3 additionally needs the real `lspart` output to
  confirm the total chunk count (expected: 8).
- **`--start-at`** is still unimplemented: the two-process runs are aligned to
  ~0.5 % by warm-up symmetry rather than exactly (§10.2). Worth adding for the
  final thesis-quality runs.
- **CPU affinity.** The two concurrent networks run in two threads; pinning them
  to distinct cores would remove the one confound that in-process concurrency
  introduces.
- **Accuracy.** Not implemented. Needs the ImageNet ground-truth labels
  (`ILSVRC2012_validation_ground_truth.txt`, distinct from the class-name file
  already present), and is the natural companion to the end-to-end mode.
- **MPS static partitioning** and **nvtaskset / libsmctrl** as alternative
  mechanisms behind the same `SmAllocation` seam, for the comparison the thesis
  promises.
- **`--start-at <epoch>`** if the concurrent experiments are ever moved to two
  processes.
- **Nsight Compute** session page for a visual check of the provisioned-SM
  bitmask (the API exposes counts only).
