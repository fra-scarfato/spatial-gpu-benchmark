# runner

A TensorRT inference benchmark for the NVIDIA Jetson AGX Orin, built to measure how much
co-located clients interfere when each is given a **disjoint partition of the GPU's SMs**, and
where that interference comes from. It runs one or more networks on CUDA **green contexts** or
under **MPS static partitions**, optionally beside a synthetic co-runner, and records every
inference.

Part of the KubeDeadline master's thesis on spatial GPU reservation for real-time containers.

## Features

- One or more TensorRT networks per process, each on its own green context, or one per process
  under MPS. SM budgets are quantised to two SMs on this SoC, and the grant is reported.
- Synthetic co-runners in their own partition: `mem_saxpy` (DRAM traffic only), `compute_gemm`
  (L2-resident arithmetic), `spin` (a resident kernel that touches no memory) and `idle`.
- One or two submitting threads, CPU pinning and real-time priority, as experimental variables.
- Per-inference GPU time from CUDA events, written as `r<k>_latencies.csv` and summarised in
  `r<k>_stats.csv`.
- Refuses to measure when the clocks are unlocked or another process holds the GPU.
- A campaign driver, a reducer, LaTeX table generators and a profiling driver that turn the runs
  into the thesis tables with no number typed by hand.

## Requirements

- Jetson AGX Orin, JetPack 7.2: CUDA 13.2 (with `nvcc`), TensorRT 10.16, cuBLAS, OpenCV 4
- GPU and CPU clocks locked with `sudo jetson_clocks`, which needs repeating after every boot
- Python 3 with numpy and pandas for `utility/profiles.py`; the rest uses the standard library
- Nsight Systems (`nsys`) for the profiled runs, and root for the DRAM counter sets
- ONNX models and ImageNet validation images under `$DATA`, default `/mnt/nvme/f.scarfato`
  (see [Data](#data))

## Quick start

```sh
make                          # builds ./runner
make -f engines.mk            # builds every TensorRT engine from its ONNX file
sudo jetson_clocks

./runner --dataset $DATA/dataset/imagenet-1k \
         --net $DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine:50:mobilenet \
         --allocation greenctx --submit-threads 1 --fixed 32 \
         --min-inferences 5000 --min-duration 30 --csv /dev/shm/out
```

The run prints the configuration it was granted and a summary per network:

```
[info] network 'mobilenet': greenctx 8/16 SMs (50%), 32 inputs, 3x224x224 in
=== LATENCY SUMMARY (timed phase) ===
inferences: 40839   duration: 30.00 s
metric                        n        mean      median         std  ...     p99-med
gpu compute (ms)          40839      0.6748      0.6747      0.0023  ...      0.0060
Throughput (whole loop): 1361.3 inferences/s
```

`./runner --help` lists every option. A plain `./runner` call has **no** busy-board guard; the
scripts below run every measurement behind one.

## Reproducing the results

`pipeline.py` is the single entry point. It runs the stages in order (engines, measure, reduce,
tables, overview) and stops at the first failure:

```sh
./pipeline.py                               # interactive: what is measured, then what to run
./pipeline.py section                       # every family the results section uses
./pipeline.py mem --runs 3                  # one group, fewer replicates
./pipeline.py all --stage reduce,tables     # regenerate the tables, measure nothing
./pipeline.py section --dry-run             # print every command instead of running it
./pipeline.py --state                       # what is measured so far
```

The same stages by hand:

```sh
./utility/sweep.sh --exp section                       # measure -> $DATA/bench/<family>/<config>/
python3 utility/aggregate.py                           # reduce  -> $DATA/bench/aggregate/
python3 utility/results-tables.py > tables.tex         # the results section's tables
python3 utility/split-tables.py tables.tex --out tables/   # one file per table
```

Profiled runs (Nsight Systems kernel traces and DRAM counters) perturb what they measure, so they
are kept out of the campaign and serve as evidence about mechanisms, not as latencies:

```sh
./utility/profile.sh traces                 # CUDA kernel traces (no root)
sudo ./utility/profile.sh soc-budget        # DRAM counters: each network alone on 4 budgets
sudo ./utility/profile.sh soc-itf           # DRAM counters: target alone and beside compute_gemm
sudo ./utility/profile.sh footprint         # DRAM counters: compute_gemm's operand size
sudo ./utility/profile.sh calibration       # DRAM counters during devprobe's copies
python3 utility/profiles.py                 # all of the above -> $DATA/profiles/derived/*.csv
```

These are also pipeline stages: `./pipeline.py mem --stage traces,profiles`.

## Measurement protocol

- Five independent runs per configuration, each in a fresh process.
- Each run does 200 untimed warm-up inferences, then at least 5000 timed inferences **and** at
  least 30 s.
- The same 32 preprocessed inputs in every configuration, and one submitting thread unless the
  experiment varies it.
- A reported value is the median of the five run medians, never a pooled percentile. The tail is
  `p99 - median`, reduced the same way.

## Repository layout

```
src/, include/          the runner: options, inputs, statistics, SM allocation (green contexts,
                        MPS), the TensorRT engine, networks, the synthetic co-runner, the run
src/spin.cu             the spin co-runner's kernel, built by nvcc
pipeline.py             the entry point
engines.mk              every TensorRT engine, rebuilt from its ONNX file
utility/sweep.sh        the campaign; its arms are listed in utility/lib/matrix.sh
utility/aggregate.py    runs -> per-experiment CSVs and summary.csv, and flags suspect runs
utility/results-tables.py, tex-tables.py   the LaTeX tables of the section and the appendix
utility/profile.sh      every profiled run;  utility/profiles.py  their reading
utility/lib/            the guards, the MPS helpers, the arm list, the family map, the table writer
utility/*.cpp           standalone probes (auxprobe, wqprobe, devprobe, itfproc); see utility/README.md
```

## Data

All data lives outside the repository under `$DATA` (default `/mnt/nvme/f.scarfato`):

```
bench/<family>/<config>/r<k>_*   the campaign: one stats, latencies and log file per run
bench/aggregate/                 the reduction: <exp>.csv per experiment, summary.csv
bench/OVERVIEW.md                every experiment and configuration, from utility/overview.py
profiles/<set>/                  the profiled runs: report, log, stats, extracted .csv.gz
profiles/derived/                everything profiles.py computes
models/<model>/                  ONNX files, and engine/ with the built engines and their logs
dataset/imagenet-1k/             ImageNet validation images (32 are used)
scratch/                         temporary files; nothing reads from it
```

## Documentation

- [`DOC.md`](DOC.md): design decisions, methodology, the data model and the traps found on the way
- [`MEASUREMENTS.md`](MEASUREMENTS.md): how every number of the memory analysis is collected and
  computed (GB/s, bytes per inference, kernel gaps, the calibration)
- [`utility/README.md`](utility/README.md): the probe tools

## Caveats

- The board is shared. Every scripted run checks the clocks, the GPU's other users and a sentinel
  latency first, and a profiled run is also watched while it runs.
- TensorRT engines are not bit-reproducible across builds. Compare results within one build.
- Profiled runs are single runs and are perturbed by the profiler. Only shapes, ratios and
  traffic are taken from them, never latencies.
