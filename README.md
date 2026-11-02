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
- Python 3 with the packages of `requirements.txt` (numpy, pandas), at the versions the thesis used
- Nsight Systems (`nsys`) for the profiled runs, and root for the DRAM counter sets
- ONNX models and ImageNet validation images under `$DATA`, default `/mnt/nvme/f.scarfato`
  (see [Data](#data))

## Quick start

```sh
make                          # builds ./runner
make -C probes        # builds the probes (device-probe, corunner, ...)
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
./measure/sweep.sh --exp section                       # measure -> $DATA/bench/<family>/<config>/
python3 analysis/aggregate.py                           # reduce  -> $DATA/bench/aggregate/
python3 analysis/tables/results-tables.py > tables.tex         # the results section's tables
python3 analysis/tables/split-tables.py tables.tex --out tables/   # one file per table
```

Profiled runs (Nsight Systems kernel traces and DRAM counters) perturb what they measure, so they
are kept out of the campaign and serve as evidence about mechanisms, not as latencies:

```sh
./measure/profile.sh traces                 # CUDA kernel traces (no root)
sudo ./measure/profile.sh soc-budget        # DRAM counters: each network alone on 4 budgets
sudo ./measure/profile.sh soc-itf           # DRAM counters: target alone and beside compute_gemm
sudo ./measure/profile.sh footprint         # DRAM counters: compute_gemm's operand size
sudo ./measure/profile.sh calibration       # DRAM counters during device-probe's copies
python3 analysis/profiles.py                 # all of the above -> $DATA/profiles/derived/*.csv
```

These are also pipeline stages: `./pipeline.py mem --stage traces,profiles`.

The stress chapter (the stress, counter and composition models, other partition sizes, two
co-runners, energy) is measured by the queues in `experiments/`, one folder per part of the thesis,
whose README maps folders to sections and keeps the order of what was run. `make tables` regenerates its numbers from the raw runs, re-running
only what is out of date, and writes its tables as LaTeX rows (`stress-tables.py`,
`ensemble-tables.py`, `geometry-tables.py` in `analysis/tables/`).

## Measurement protocol

- Five independent runs per configuration, each in a fresh process.
- Each run does 200 untimed warm-up inferences, then at least 5000 timed inferences **and** at
  least 30 s.
- The same 32 preprocessed inputs in every configuration, and one submitting thread unless the
  experiment varies it.
- A reported value is the median of the five run medians, never a pooled percentile. The tail is
  `p99 - median`, reduced the same way.

## Repository layout

The folders follow the path of a number: built, measured, recorded, reduced and modelled, then
printed as a table. Raw data never enters the repository (see [Data](#data)).

```
README.md, Makefile         how to build and reproduce; `make` (the runner), `make check` (the
                            self-tests), `make tables` (the stress chapter's numbers and tables)
pipeline.py                 the results section's entry point: engines, measure, reduce, tables
engines.mk                  every TensorRT engine, rebuilt from its ONNX file
requirements.txt            the Python packages, pinned
src/, include/              the runner: options, inputs, statistics, SM allocation, the TensorRT
                            engine, networks, the synthetic co-runner, the run (src/spin.cu: nvcc)
probes/                     standalone CUDA programs with their own Makefile, README and the pin
                            file of the binaries: aux-streams-probe, work-queue-probe, device-probe (the GB/s
                            calibration), corunner (the co-runner and stressor as a separate
                            process), gpu-latency (RRZE-HPC's pointer chase, the probe, GPL-3)
measure/                    everything that runs on the GPU, guarded
  sweep.sh                    the campaign: runs every arm of a family, resumable
  reporter.sh                 the probe beside stressor levels or networks, under MPS static
  profile.sh, ncu.sh          the profiled runs (kernel traces, DRAM counters, Nsight Compute)
  mps-additivity-block.sh     the additivity family under MPS (three clients, three processes)
  int8-calib.py               the INT8 calibration engines.mk uses
  lib/guards.sh               the checks before every run: clocks, a quiet GPU, pins, manifest
  lib/matrix.sh               every arm the campaign can run, one line each
  lib/mps.sh                  the MPS daemon and its static partitions
experiments/                one folder per part of the thesis: its README (what, how, outputs) and
                            the queues that measured it, as run; experiments/README.md is the map
analysis/                   everything that reads the data, CPU only
  aggregate.py                raw runs -> bench/aggregate/<exp>.csv and summary.csv
  profiles.py                 profiled runs -> profiles/derived/*.csv
  stress.py                   the stress models (predict|compare <geometry>)
  inference-average-model.py, temporal-model.py   the temporal model
  counter-model.py            the counter model (DRAM, L2, both)
  ensemble.py                 correction, maximum and margin of the three p99 models
  compose.py                  two stressors together (the composition ratio)
  geometry.py                 other partition sizes, two co-runners, power
  overview.py                 bench/OVERVIEW.md: every family, its configurations and runs
  lib/families.py             family folder -> aggregate key, target-first families, groups
  lib/tables.py               the LaTeX writer the results-section generators share
  tables/                     every thesis table, as LaTeX rows (the map below)
  explore/                    one-off investigations kept for the record, outside the pipeline:
                              l2-residual, onoff, probe-window-bursts, peak-test, models, streams
docs/                       DESIGN.md (design decisions and traps), MEASUREMENTS.md (what every
                            number means, the names of the geometries and the glossary)
```

### Where each thesis table comes from

| Thesis | Generator | Run by |
|---|---|---|
| results section | `analysis/tables/results-tables.py` (+ `split-tables.py`) | `./pipeline.py --tables` |
| appendix | `analysis/tables/appendix-tables.py` | `./pipeline.py --tables` |
| stress chapter: `tab:stress-scale` ... `tab:stress-compose` (11) | `analysis/tables/stress-tables.py` | `make tables` |
| stress chapter: `tab:models-*`, `tab:bound-*`, `tab:stress-margin`, `tab:stress-pessimism` (7) | `analysis/tables/ensemble-tables.py` | `make tables` |
| stress chapter: `tab:stress-sizes*`, `tab:stress-triple*`, `tab:stress-energy*` (6) | `analysis/tables/geometry-tables.py` | `make tables` |

Each generator heads every block with its `\label`, so a table is found by its label.

To add a configuration, add an arm to `measure/lib/matrix.sh` (and, for a new family, its folder
to `analysis/lib/families.py`, and its key to `TARGET_FIRST` there when its configurations name the
target first); a single-network arm whose network the configuration name does not state otherwise
also needs a rule in `aggregate.py`'s `load()`. To add a table, add it to
`analysis/tables/results-tables.py`.

## Data

All data lives outside the repository under `$DATA` (default `/mnt/nvme/f.scarfato`):

```
bench/<family>/<config>/r<k>_*   the campaign: one stats, latencies and log file per run
bench/aggregate/                 everything derived, regenerated by aggregate.py and make tables
bench/frozen/                    predictions frozen before their pairs were read, with their
                                 calibrations; never regenerated, hashed in analysis/frozen.sha256
bench/tables/                    every generated table: results.tex and results/ (pipeline.py),
                                 stress.tex, ensemble.tex, geometry.tex (make tables)
bench/OVERVIEW.md                every experiment and configuration, from analysis/overview.py
profiles/<set>/                  the profiled runs: report, log, stats, extracted .csv.gz
profiles/derived/                everything profiles.py computes
models/<model>/                  ONNX files, and engine/ with the built engines and their logs
dataset/imagenet-1k/             ImageNet validation images (32 are used)
scratch/                         temporary files; nothing reads from it
```

## Documentation

- [`docs/DESIGN.md`](docs/DESIGN.md): design decisions, methodology, the data model and the traps found on the way
- [`docs/MEASUREMENTS.md`](docs/MEASUREMENTS.md): how every number of the memory analysis is collected and
  computed (GB/s, bytes per inference, kernel gaps, the calibration)
- [`probes/README.md`](probes/README.md): the probe programs

## Caveats

- The board is shared. Every scripted run checks the clocks, the GPU's other users and a sentinel
  latency first, and a profiled run is also watched while it runs.
- TensorRT engines are not bit-reproducible across builds. Compare results within one build.
- Profiled runs are single runs and are perturbed by the profiler. Only shapes, ratios and
  traffic are taken from them, never latencies.
