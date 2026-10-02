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
  aggregate/                              aggregate.py's reduction and the generated tables
profiles/                       PROFILED RUNS: one run per arm, read for traffic and timelines
  traces/<arm>.*                CUDA kernel traces (no root)
  soc-budget/<arm>.*            DRAM counters, each network alone on 4 budgets, and 3 pairs (root)
  soc-itf/<arm>.*               DRAM counters, the target alone and beside compute_gemm (root)
  footprint/<arm>.*             DRAM counters, compute_gemm with its operand side varied (root)
  calibration/                  DRAM counters during devprobe's device-to-device copies (root)
  ncu/                          Nsight Compute output, per-kernel only (root)
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
| `<arm>.dram.csv.gz` | counter runs only: the DRAM read and write series, one row per sample |

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
Orin's memory-controller counters 10,000 times a second. Two of them matter: **DRAM Read
Throughput** and **DRAM Write Throughput**, each given as a percentage of a peak.

- They are **device-wide**. A reading belongs to a network only when that network is the only
  thing running, which is why the budget curve is taken one network at a time.
- They need **root**.
- They have no usable time alignment with the CUDA timeline: the SoC clock sits about 14 s from the
  CUDA clock in the export. No quantity below needs a time window. Each is either a statistic of
  the whole series or an integral over it.

**Extraction.** The export's `SOC_METRICS` table has no column names, so its rows are read by
position: `(start, end, domain, index, value)`. The index of each counter is looked up by name in
`TARGET_INFO_SOC_METRICS`; on this board read is index 6 and write is index 7.

### The statistics of one series

| name | definition | used for |
|---|---|---|
| **peak** | the largest sample | not used for any conclusion |
| **busy mean** | the mean of the samples above 1 %, which drops the idle samples between inferences | Table `tab:mem-pressure` only |
| **integral** | the sum of all samples, after subtracting the idle floor (the median of the samples below 1 %, zero on this board) | everything in GB |

The busy mean depends on how much idle time falls inside a 100 µs sample, so it understates the
rate during the kernels: MobileNet on 16 SMs averages 7.1 % by this rule, against 17.4 % averaged
over its kernels by Nsight Compute. The integral has no such problem, which is why every quantity
in GB is built from it.

### Percent to GB: the calibration

A percentage becomes GB/s through one measurement (`profile.sh calibration`; `calibrate()` in
`profiles.py`, written to `derived/calibration.csv`):

1. `utility/probes/devprobe` performs device-to-device copies under the counters. The trace records each
   copy's bytes and duration: 635 copies, 25.54 GB in 0.279 s, so the copies ran at
   **91.5 GB/s**. A copy reads every byte once and writes it once, so that is 91.5 GB/s of read
   and 91.5 GB/s of write.
2. While the copies run, both counters sit on a plateau. The plateau is the samples where read
   exceeds 20 %: 2799 samples, 280 ms, which matches the 279 ms of copy time. Their medians are
   **47 % read and 45 % write**.
3. So 1 % of read is 91.5/47 = 1.95 GB/s and 1 % of write is 91.5/45 = 2.03 GB/s. The implied
   peaks are 195 GB/s read and 203 GB/s write, consistent with the Orin's 204.8 GB/s LPDDR5 peak.

The GB a run moved is then

    GB = ( Σ read% / 47  +  Σ write% / 45 ) × 91.5 / 10000

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
subsets. It is a description of seven points, not a model of the hardware.

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
kernel of an inference, recurs every 55 kernels (`target_stream()`). The first 1000 kernels (the
warm-up) are dropped. An inference begins at each `copyPackedKernel`, and only inferences with
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

### With and without a co-runner (`derived/trace-overlap.csv`, Table `tab:mem-trace-overlap`)

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
read for the target's median against the idle arm (`derived/footprint.csv`).

## 7. Limits, all of which the text states where they apply

- A profiled run is perturbed by the profiler. Its latencies (MobileNet alone: 0.69 ms profiled,
  0.675 ms in the campaign) are never reported as results. Only shapes, ratios, traffic and
  per-kernel structure are taken from it.
- One profiled run per arm, against five in the campaign.
- The absolute GB/s rest on one calibration: one copy pattern, one session.
- Device-wide counters attribute traffic only when one client runs. Inside a pair only the total
  is known.
- The host-late split uses the time a launch call *returned*. A kernel can start before its call
  returns, so host-late is an upper bound on host lateness and queued a lower bound on device
  lateness.
- The counters cover DRAM only. No counter on this board reports L2 traffic while two clients run:
  Nsight Compute serialises the other clients around each kernel it measures.
