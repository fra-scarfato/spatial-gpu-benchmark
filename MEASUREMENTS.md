# How the numbers of the memory subsection are collected

Every number in the memory subsection comes from one of two sources: the **campaign** (latencies)
or a **profiled run** (DRAM traffic and kernel timelines). This file defines each quantity, says
where it is stored, which code computes it, and what limits it carries.

## 1. Where the data lives

All data is under `$DATA = /mnt/nvme/f.scarfato`, outside the repository.

```
bench/                          THE CAMPAIGN: latencies, five runs per configuration
  <family>/<config>/r<k>_stats.csv        one row per statistic of one run (gpu_ms is the one used)
  <family>/<config>/r<k>.log              the runner's report, including a co-runner's own figures
  aggregate/                              aggregate.py's reduction, stress.py's files, the tables
  stress-reporter/<config>/                the stress reporter beside SAXPY or a network (reporter.sh)
  stress-reporter-target/<config>/         the same with LAYOUT=target
profiles/                       PROFILED RUNS: one run per arm, read for traffic and timelines
  traces/<arm>.*                CUDA kernel traces (no root)
  soc-budget/<arm>.*            DRAM counters, each network alone on 4 budgets, and 3 pairs (root)
  soc-itf/<arm>.*               DRAM counters, the target alone and beside compute_gemm (root)
  footprint/<arm>.*             DRAM counters, compute_gemm with its operand side varied (root)
  calibration/                  DRAM counters during devprobe's device-to-device copies (root)
  ncu/                          Nsight Compute output, per-kernel only (root); nets-<net>.csv, nets/
  derived/*.csv                 everything profiles.py computes from the above
  archive/                      earlier traces and the outputs of scripts that were replaced
scratch/                        temporary files; nothing reads from it
```

Every arm in `profiles/<set>/` has the same files:

| file | content |
|---|---|
| `<arm>.nsys-rep` | the Nsight Systems report |
| `<arm>.log` | the runner's own report: inferences, throughput, and the co-runner's units, bytes and FLOPs |
| `<arm>_stats.csv` (or `<arm>_<net>_stats.csv`) | the runner's latency statistics for that run |
| `<arm>.kernels.csv.gz` | every kernel: start, end (ns), stream, correlation id, thread blocks, name |
| `<arm>.launches.csv.gz` | every kernel-launch call on the host: correlation id, start, end, thread |
| `<arm>.dram.csv.gz` | counter runs only: the SoC DRAM and the GPU's own read and write series, one row per sample |

The three `.csv.gz` files are extracted from the report once, by `profiles.py`, and the sqlite
export (hundreds of MB, minutes to produce) is then deleted. Everything downstream reads the
`.csv.gz` files.

## 2. The code

| step | command | writes |
|---|---|---|
| campaign | `./utility/sweep.sh --exp <family>` | `bench/<family>/...` |
| reduce | `python3 utility/aggregate.py` | `bench/aggregate/*.csv` |
| profiled runs | `[sudo] ./utility/profile.sh <set>` | `profiles/<set>/...` |
| read the profiles | `python3 utility/profiles.py` | `profiles/derived/*.csv` |
| tables | `python3 utility/results-tables.py > tables.tex` | every table of the section |

`./pipeline.py` runs the same steps as stages. `profile.sh <set> --list` prints the arms of a set
and the exact runner command of each, without running anything.

Every run, campaign or profiled, is preceded by the same guard (`utility/lib/guards.sh`): the GPU
and CPU clocks must be locked, no other process may hold the GPU, and a short sentinel run of
MobileNet must land within 15 % of 0.535 ms. A profiled arm is also watched while it runs: if
another process opens the GPU during the arm, the arm is moved to `rejected/` and taken again.

## 3. Latency: the campaign

**What is timed.** `gpu_ms` is the time between two CUDA events recorded on the network's stream,
one before the inference's first kernel and one after its last. It therefore includes every
kernel and every gap between kernels, and excludes the host-to-device and device-to-host copies.

**Protocol.** A run is a fresh process that performs 200 untimed warm-up inferences, then at least
5000 timed inferences **and** at least 30 s. Five runs per configuration.

**Median.** The value reported for a configuration is the median of the five run medians
(`aggregate.py`; `profiles.py` recomputes it the same way from `r<k>_stats.csv`).

**Tail.** `p99 - median` of one run, then the median over the five runs.

**Multiplier (`×iso`).** The configuration's median divided by the target alone **in the same
family**:
- the interferer and budget families use `isolated-scaling/mobilenet_gc50`, 0.6752 ms;
- the additivity family uses its own `mobilenet-alone`, 0.6718 ms, measured in the same session as
  its pairs.

The 0.5 % between the two references is the session-to-session spread of one configuration.

**Penalty.** `(median / reference - 1) × 100`.

## 4. DRAM traffic: the SoC counters

### What the counter is

`nsys profile --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000` samples the
Orin's memory-controller counters 10,000 times a second, each as an integer percentage of a peak.
Two pairs matter:
- **GPU Read / GPU Write Throughput**: the GPU's own traffic to and from DRAM. **Every quantity in
  GB is built from this pair.** It reads 0 when the GPU is idle.
- **DRAM Read / DRAM Write Throughput**: every client of the memory controller, the CPU and the
  display among them. It never reads below 1 %, and on a quiet GPU it reads 1–2 % read and 1.5–3 %
  write. An earlier revision built the GB from this pair: it overstated a network's bytes by up to
  7 % (MobileNet, the lightest) and by 0–3 % for the others, and left the pair ratios unchanged.

Both pairs:
- cover the **whole GPU**. A reading belongs to a network only when that network is the only
  thing running on it, which is why the budget curve is taken one network at a time;
- They need **root**.
- They have no usable time alignment with the CUDA timeline: the SoC clock sits about 14 s from the
  CUDA clock in the export. No quantity below needs a time window. Each is either a statistic of
  the whole series or an integral over it.

**Extraction.** The export's `SOC_METRICS` table has no column names, so its rows are read by
position: `(start, end, domain, index, value)`. The index of each counter is looked up by name in
`TARGET_INFO_SOC_METRICS` (`soc_series()`); on this board GPU read and write are indices 2 and
3, DRAM read and write 6 and 7.

### The statistics of one series

| name | definition | used for |
|---|---|---|
| **peak** | the largest sample | not used for any conclusion |
| **busy mean** | the mean of the samples above 1 %, which drops the idle samples between inferences | `derived/dram.csv` only; no longer in the thesis |
| **integral** | the sum of all samples; the GPU pair reads 0 when idle, so no floor is subtracted | everything in GB |

The busy mean depends on how much idle time falls inside a 100 µs sample, so it understates the
rate during the kernels. (The 17.4 % per kernel once quoted from Nsight Compute against it is not a
fair comparison: ncu flushed the L2 before every kernel, which inflates each kernel's DRAM
traffic.) The integral has no such problem, which is why every quantity in GB is built from it.

### Percent to GB: the calibration

A percentage becomes GB/s through one measurement (`profile.sh calibration`; `calibrate()` in
`profiles.py`, written to `derived/calibration.csv`):

1. `utility/probes/devprobe` performs device-to-device copies under the counters. The trace records each
   copy's bytes and duration: 635 copies, 25.54 GB in 0.279 s, so the copies ran at
   **91.5 GB/s**. A copy reads every byte once and writes it once, so that is 91.5 GB/s of read
   and 91.5 GB/s of write.
2. While the copies run, both GPU counters sit on a plateau. The plateau is the samples where GPU
   read exceeds 20 %: 2799 samples, 280 ms, which matches the 279 ms of copy time. Their medians
   are **45 % read and 44 % write** (the DRAM pair, CPU included, reads 47 and 45).
3. So 1 % of read is 91.5/45 = 2.03 GB/s and 1 % of write is 91.5/44 = 2.08 GB/s. The implied
   peaks are 203 GB/s read and 208 GB/s write, against the Orin's 204.8 GB/s LPDDR5 peak; the
   counter is an integer, so each plateau median is only known to ±0.5 %, i.e. ±1 % in GB.

The GB a run moved is then

    GB = ( Σ GPU read% / 45  +  Σ GPU write% / 44 ) × 91.5 / 10000

because each sample covers 1/10000 s.

**A correction.** Before this revision the calibration point was 48 % read and **83 %** write,
which are the maxima of the series. The 83 % belongs to 85 samples in which read is about 1 %:
write-only phases, which are devprobe's memsets initialising its buffers, not the copy. Converting
with 83 % understated every write by 83/45 = 1.8×. All absolute GB and GB/s in the earlier drafts
(the 21–29 GB/s of the real co-runners, for instance) were affected. Ratios between runs were
barely affected.

### Quantities built on the integral

**Bytes per inference** (`derived/dram.csv`, Table `tab:mem-bytes`):

    MB per inference = GB of the run × 1000 / (timed inferences + 200 warm-up)

The inference counts come from the run's own log. This uses the whole run, so idle time costs
nothing and needs no window.

**Run-average GB/s** of a network alone:

    GB/s = MB per inference × the run's own throughput (inf/s) / 1000

This is the rate the network applies to the memory system averaged over time, idle gaps included.
It is the quantity used as a co-runner's demand.

**Pair traffic** (`derived/pair-traffic.csv`, Table `tab:mem-pair-traffic`):

    predicted GB = Σ over the two networks of (their inferences in the pair run + 200)
                                          × their MB per inference alone at the same SM share
    ratio        = GB measured in the pair run / predicted GB

A ratio of 1 means the two networks together move exactly the bytes they move apart. Above 1
means extra DRAM traffic when they share the device, such as extra L2 misses.

**The co-runners' demand** (`derived/demand.csv`, Table `tab:mem-demand`):

| co-runner | its demand is |
|---|---|
| a network at k SMs | its run-average GB/s alone at k SMs |
| `mem_saxpy` | its own count: each `cublasSaxpy` over 256 MB vectors reads 2×256 MB and writes 256 MB, so 3×256 MB per call, divided by the elapsed time measured with CUDA events. Median over the five campaign runs. |
| `compute_gemm` | by subtraction, from `soc-itf`: (GB of the run beside the target − the target's inferences × its MB per inference alone) / the co-runner's elapsed time. This assumes the target moves the same bytes per inference with and without it, so an increase in the target's traffic would be counted as the co-runner's. |
| `spin` | zero by construction: its kernel touches no memory |

**The fit** (`derived/demand-fit.csv`): an ordinary least-squares line of penalty (%) against
demand (GB/s), over all co-runners with a measured demand and over the synthetic and network
subsets. It is a description of nine points (`spin` twice, at 4 and 8 SMs), not a model of the
hardware.

**Two biases, both in the same direction.** A network's demand is its rate *alone*, while
`mem_saxpy`'s is its own count *beside the target*; beside the target the networks run 6–15 % slower,
so their real demand there is lower than the table's. `compute_gemm`'s subtraction counts as its
own any extra traffic the target makes beside it, and real pairs move 9–15 % more than apart. Both
overstate the demand of the non-SAXPY co-runners, so the excess of their penalty over the SAXPY
slope is if anything understated.

## 5. Kernel timelines: the CUDA traces

### What the trace records

`nsys profile --trace=cuda` records, for every kernel:
- its start and end on the GPU (ns);
- its stream;
- its grid (thread blocks);
- its correlation id.

For every launch call it records:
- the host thread;
- the time the call was entered and the time it returned.

The correlation id links a kernel to the call that launched it. No root is needed.

### Slicing the target into inferences

The target is MobileNet on 8 SMs. Its stream is the one on which `copyPackedKernel`, the first
kernel of an inference, recurs every 55 kernels (`target_stream()`). The first 200 × 55 kernels
(the warm-up) are dropped. An inference begins at each `copyPackedKernel`, and only inferences with
exactly 55 kernels are kept, so a kernel missing from the trace can never shift the positions of
those after it. Position 0–54 within an inference is then the same layer in every inference. The
`whole_inferences` column of `trace-split.csv` is the share of inferences kept, and should be 1.

### Per inference (`derived/trace-split.csv`, Table `tab:mem-trace-split`)

| quantity | definition |
|---|---|
| **span** | last kernel's end − first kernel's start |
| **kernels** | Σ of the 55 kernel durations (end − start) |
| **gap** | start of kernel i − end of kernel i−1, for i = 1..54 (the gap before an inference's first kernel is not part of it) |
| **host-late** part of a gap | the part of the gap that elapsed before the host returned from the call launching kernel i: min(gap, max(0, launch-call end − end of kernel i−1)). During it, the device had nothing to run because the host had not finished submitting. |
| **queued** part of a gap | gap − host-late. The kernel had been submitted and the device had not yet started it. |
| reported | the median of each over all inferences of the run |

So span = kernels + queued + host-late, inference by inference.

### With and without a co-runner (`derived/trace-overlap.csv`; no longer a thesis table, quoted in the text)

All kernels on other streams belong to the co-runner: the interferer's, or the other network's.

| quantity | definition |
|---|---|
| **covers** | the fraction of the target's total kernel time during which some co-runner kernel was also executing |
| **coverage of one kernel** | the fraction of its own duration overlapped by co-runner kernels |
| **free** duration of position p | the median duration of the kernel at position p over the inferences where its coverage was below 1 %. Taken from the same run when every position has at least 30 such samples, otherwise from the trace of the target alone (the `free_basis` column says which). |
| **overlapped** duration of position p | the median over the inferences where its coverage was above 99 % |
| **fixed** and **proportional** | the least-squares line `overlapped − free = fixed + proportional × free` through the 55 positions. *Fixed* is µs added to every kernel regardless of its length; *proportional* is the share of its length added. |
| **gap, co-runner idle / running** | the median gap between consecutive target kernels whose interval was covered by co-runner kernels below 1 % / above 99 % |
| **gap, co-runner running, nothing queued** | the same, restricted to gaps that opened while the co-runner had no launch that had returned to the host and not yet started on the device, so that its queue was empty |

The side-64 footprint arm (`footprint/A1-l1`) carries a CUDA trace as well. It is read by the same
code under the name `gemm64`, with its own idle stretches as the free reference.

**Launch calls** (`derived/launch-calls.csv`): the mean duration of the launch calls of each host
thread. The target's thread is the one that launched the target's kernels.

## 6. The footprint arms

`profile.sh footprint` runs `compute_gemm` beside the target at seven operand sides (64 to 1024).
The side sets the footprint (3 × side² × 4 bytes), but also the size of every GEMM kernel and the
number of kernels per unit (4.3 × 10⁹ / side³). So the arms vary footprint, kernel size and launch
rate together, and cannot attribute an effect to the footprint alone. One profiled run per arm,
read for the target's median against the idle arm (`derived/footprint.csv`). `dram_mb_per_target_inference` is the
device's DRAM bytes over the run (the GPU counters, as in section 4) divided by the target's
inferences including warm-up: it holds the GEMM's own misses as well as the target's, so it shows
where the combined traffic steps, not how much of it is the target's. The thesis reads the step
between 0.9 and 1.8 MB (`tab:mem-footprint`) as consistent with the two working sets no longer
fitting together in the 4 MB L2, and states the arms' confounds beside it.

## 7. The stress model (MPS static only)

The question: how much does a network **stress** the memory system, how **sensitive** is a target
to that stress, and does the pair of numbers **predict** the target's slowdown beside a co-runner
that was never profiled with it. Code: `utility/probes/itfproc.cu` (the stressor),
`utility/probes/gpu-latency/` (the reporter), `utility/reporter.sh` (reporter runs), the
`stress-sensitivity` and `stress-validation` arms in `utility/lib/matrix.sh`, `utility/stress.py`
(the model).

### The stressor: throttled SAXPY

`itfproc saxpy --level L` performs the same operation as `mem_saxpy` (y = a*x + y over two 256 MB
vectors, so 3 x 256 MB moved per unit), as its own kernel so that it can be throttled from inside.
Each thread moves 16 elements, then waits a fixed time on the GPU's own clock (`%globaltimer`).
Thousands of threads, out of phase with one another, lower the pressure uniformly at microsecond
scale, and the kernel stays resident at every level.

- **The level L is the achieved bandwidth divided by the same kernel's full-speed bandwidth on the
  same partition.** It is not a time-based duty cycle. The full-speed time comes from 9 warm-up
  units. The wait is then set from the model unit time = full-speed time + bursts per thread x
  wait, refined by three short corrections, and **frozen** for the run.
- Once a second `itfproc` prints `level: setpoint S achieved A`, where A is the median of that
  second's units (full-speed unit time / unit time). The analysis uses the median of A over the
  run, and reports its spread.
- L = 0 runs the same kernel with waits only: resident, no memory traffic.
- The stressor holds 8 SMs (4 chunks) beside the target, and 4 SMs in the 4-SM variant below.

### The reporter: gpu-latency

RRZE-HPC's pointer chase (`utility/probes/gpu-latency/UPSTREAM` gives the commit and every change).
One thread follows a random chain through 128-byte cache lines of a **64 MB** buffer, so every access
goes to DRAM and waits for the one before. It runs in its own 2-SM partition and adds almost no
traffic.

**Windows.** One long launch of 2000 windows of 512 dereferences (about 0.3 ms each), each closed by a
`clock64` mark. The mark is read only after the window's last load has completed, because the
thread first branches on the loaded pointer (verified in the SASS). Cycles are converted to ns with
the pinned 1300.5 MHz, and the first window of each launch is dropped. A 30 s run gives about
85,000-95,000 windows.

**Why windows.** A network's DRAM traffic comes in bursts. Upstream's samples, each a separate launch
averaging 0.3 s of accesses, hid them: all four networks read 603–612 ns, within the band between
SAXPY 0 % and 10 %. Short separate launches carry about 0.3 ms of launch and event overhead under MPS,
as long as the chase itself. Windows timed inside the kernel resolve the bursts with no overhead.

**Validation.** The window means reproduce upstream's values on the 64 MB chain: 617 against 617 ns
idle, 603 against 606 beside SAXPY 0 %, 648 against 649 beside SAXPY 20 %.

**The coordinate:** the **p99 of a run's windows**, with the median recorded too. Configuration value
= the median over five runs.

Not used:
- a 2 MB (L2-sized) chain, which any co-runner, even SAXPY at 10 %, evicts to DRAM latency at once:
  it carries no level;
- the idle reading, which is about 14 ns slower than SAXPY 0 %, a resident kernel that moves no data.
  This is consistent with the shared LPDDR5 dropping into a low-power state when nothing touches it.
  The zero of the scale is SAXPY 0 %.

### Steps and quantities

| quantity | from | definition |
|---|---|---|
| `R(L)` | `reporter.sh calibration` | the reporter's p99 (and median) window latency beside SAXPY at achieved level L, at levels 0, 2, 4, 6, 8, 10, 15, 20, 30 … 100 %; SAXPY 8 SMs + reporter 2 SMs |
| `L_eq(net)` | `reporter.sh networks` | the level whose `R` equals the reporter's reading beside the network (network 4 SMs, at least 5000 inferences and 30 s, + reporter 2 SMs for as long as the network runs), by inverse linear interpolation; a reading outside the calibrated range is clipped and flagged |
| occupancy, inf/s | the network co-runner's own `_stats.csv` | mean GPU time per inference x inferences / run duration, and inferences per second |
| `S_p99(L)`, `S_med(L)` | `stress-sensitivity` | the target's p99 (median) beside SAXPY at level L divided by its p99 (median) alone, target 8 SMs + SAXPY 8 SMs, medians over five runs |
| predicted slowdown | `stress.py predict` | `S(L_eq(co-runner))` by linear interpolation, for p99 and median, from each reporter statistic (p99 is the model's coordinate) |
| measured slowdown | `stress-validation` | the target's p99 (median) beside the co-runner (target 8 + co-runner 4 SMs) over its alone p99 (median) |
| error | `stress.py compare` | predicted minus measured slowdown, in points (x100) |

**The bottom of the p99 curve.** At level 0 the stressor moves no data, and the reporter's p99 there is
power-state noise: 621-635 ns across five runs, against +-0.6 ns at 2 %, and above the 2 % point. The
inversion therefore uses each curve from its lowest point upward. A reading below that point is
mapped to its level and flagged as outside the range: MobileNet and Inception read below the 2 %
point, so they show no tail stress beyond being present. The median curve is monotonic from 0 %.

**Held-out rule:** `predict` reads only the reporter runs and `stress-sensitivity`, and writes
`stress-predictions.csv` before any pair is read. `compare` is the only reader of
`stress-validation`, and it refuses to run without the predictions.

**Why the reporter is the bridge:** the stressor holds 8 SMs and a real co-runner 4. Equal GB/s
from different numbers of SMs is not equal stress, because a different number of requests is in
flight. The reporter reads either with the same instrument, in the same partition.

**Domain:** at L = 1 the 8-SM SAXPY moves 137 GB/s (the 4-SM one 91 GB/s), against peaks of about
203 GB/s read and 208 GB/s write. Predictions hold for co-runners whose reading falls inside the calibrated range.

Output: `$DATA/bench/stress-reporter/<config>/r<k>_latency.txt`, `r<k>_c<i>.log` and
`r<k>_c<i>_stats.csv`. `stress.py` writes `$DATA/bench/aggregate/stress-calibration.csv`,
`stress-corunners.csv`, `stress-sensitivity.csv`, `stress-predictions.csv` and
`stress-comparison.csv`.

### The 4-SM stressor (`stress.py ... 4sm`)

The same model with SAXPY on the 4 SMs a real co-runner holds, so that stressor and co-runner share
one geometry: `reporter.sh calibration-4sm` (configurations `saxpy4-L*`, L relative to full speed on
4 SMs, 91.1 GB/s) and the `stress-sensitivity-4sm` family (exp15: target 8 SMs {0,1,4,5}, SAXPY 4 SMs
{2,6}, the co-runner's chunks in `stress-validation`). `stress.py predict 4sm` and `compare 4sm` write
the same files with the suffix `-4sm`; the 8-SM files are not touched. The 4-SM calibration is flat
and not monotonic below L 0.2, so `predict 4sm` refuses to invert it.

### Where the reporter sits (`LAYOUT`)

`reporter.sh` puts the reporter on one 2-SM chunk created after the co-runners (`LAYOUT=chunk`, the
default and the layout of every reading above): co-runner {0,4} or {0,1,4,5}, reporter {1} or {2}.
`LAYOUT=target` creates the reporter's partition first with 8 SMs, so that it sits where the target
sits in the sweeps, {0,1,4,5}, and every co-runner on the chunks it holds beside a target ({2,6} for
4 SMs, {2,3,6,7} for 8). The reporter is one thread, so the size of its partition changes only its
place. Output: `$DATA/bench/stress-reporter-target/`.

### L2 traffic of each network (`ncu.sh nets`, root)

Each network alone on a 4-SM green context, all kernels of four inferences after the warm-up, with
the L2 kept warm between kernels (`--cache-control none`): `lts__t_bytes` (L2 traffic), the L2 and
L1 hit rates and the kernel durations. L2 MB per inference x the network's throughput gives its L2
traffic in GB/s; L2 bytes x (1 - hit rate) should match the DRAM bytes per inference of the SoC
counters, which checks the two instruments against each other. Output:
`$DATA/profiles/ncu/nets-<net>.csv`. The first run (2026-10-07, in `profiles/ncu/nets/`) profiled
kernels inside the warm-up.

### Names: the geometries of `stress.py`, and code against thesis

A geometry names the stressor, how the reporter reads, and where it sits. Steady is the
bandwidth-based SAXPY (`--level`), Burst the time-based one (`--duty`); Windowed is the reporter
with `WINDOW=512`, Plain with `WINDOW=0`; Separate is `LAYOUT=chunk`, Symmetric `LAYOUT=target`.

| geometry | stressor | reporter | layout | families |
|---|---|---|---|---|
| `8sm` | Steady on 8 SMs | Windowed | Separate | exp13 |
| `4sm` | Steady on 4 SMs | Windowed | Separate | exp15 |
| `8smt` | Steady on 8 SMs | Windowed | Symmetric | exp13 |
| `tb` | Burst | Plain | Symmetric | exp17 |
| `tbw` | Burst | Windowed | Symmetric | exp17 |
| `tbsw` | Burst, scrambled in 2 MB windows | Windowed | Symmetric | exp18 |
| `onoff`, `uniform` | ON/OFF over 400 ms; uniform 32 MB units over ~3 ms | Windowed | Symmetric | exp23 |
| `onoff88`, `uniform88` | the same | Windowed, networks on 8 SMs | Symmetric | exp23, pairs exp24 |
| `smooth88` | Smooth: 100 us periods, pulses of ~3 us (the thesis's stress model) | Windowed, networks on 8 SMs | Symmetric | exp23, pairs exp24 |
| `smoothx88` | Smooth, topped by the plain kernel back to back | the same | Symmetric | exp23, pairs exp24 |
| `smoothnew` | `smooth88` on the validation networks | the same | Symmetric | exp26, pairs exp27 |

One term per concept; the code's term where it differs from the thesis's:

| code | thesis | meaning |
|---|---|---|
| reporter (`gpu-latency`) | probe | the pointer chase that reads the memory latency |
| victim (`geometry.py` columns) | target | the network whose slowdown is measured or predicted |
| co-runner | co-runner | every other network or stressor on the GPU |
| level (`--level`) | level | achieved bandwidth over the same kernel's full-speed bandwidth on the same partition |
| duty (`--duty`) | duty | the fraction of each period the time-based SAXPY runs |
| `L_eq` | equivalent stress | the stressor's level (or GB/s) whose probe reading equals the reading next to a network |
| `S(L)` | sensitivity | the target's slowdown next to the stressor at level L, over the target alone |
| presence, `PRESENCE_US`, `p_t` | dispatch term | kernels per inference x 1.05 us over the target's median latency alone |
| `L`, `L_t`, `L_c` (`counter-model.py`) | L2 bandwidth | ncu L2 MB per inference x inferences per second |
| `g` | growth of the dispatch term | two spin co-runners over one (1.60 on MobileNet) |
| `rho`, `f` | composition factor | the two-SAXPY ratio of `compose.py`; `f`, a constant on the summed load |

"Pressure" and "bubble" (Bubble-Up's word for a stressor) are not used: the terms are stress and
stressor.

## 8. Limits, all of which the text states where they apply

- A profiled run is perturbed by the profiler. Its latencies (MobileNet alone: 0.69 ms profiled,
  0.675 ms in the campaign) are never reported as results. Only shapes, ratios, traffic and
  per-kernel structure are taken from it.
- One profiled run per arm, against five in the campaign.
- The absolute GB/s rest on one calibration: one copy pattern, one session.
- The counters cover the whole GPU, and attribute traffic only when one client runs. Inside a pair
  only the total is known.
- The counters are integer percentages, about 2 GB/s a step. Over hundreds of thousands of samples
  rounding averages out; truncation would not, and cannot be ruled out from the data.
- The host-late split uses the time a launch call *returned*. A kernel can start before its call
  returns, so host-late is an upper bound on host lateness and queued a lower bound on device
  lateness.
- The counters cover DRAM only. No counter on this board reports L2 traffic while two clients run:
  Nsight Compute serialises the other clients around each kernel it measures.
