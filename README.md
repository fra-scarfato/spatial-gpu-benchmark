# runner

Measures how much two clients interfere on a Jetson AGX Orin when they are given **disjoint SM
partitions**, and where the interference comes from.

TensorRT engines over a fixed set of inputs, one stats file per network per run, green contexts
and MPS static partitions at matched SM budgets. Built for the KubeDeadline thesis, on spatial
GPU reservation.

## What it does

- runs one or two networks in one process on **green contexts**, or two processes under **MPS**
- gives every client an SM budget, quantised to two SMs on this SoC, and reports what is left idle
- can run a synthetic **participant** that does either memory traffic or arithmetic and never
  both, so the two can be told apart
- times every inference and writes `r<k>_stats.csv` and `r<k>_latencies.csv` per network per run
- refuses to measure when the clocks are not locked, or when somebody else is using the GPU

## Build

    make                    # the runner, into ./runner
    make -C utility         # the probe tools

Needs CUDA, TensorRT and cuBLAS on aarch64. Nothing else is linked.

    make -f engines.mk      # rebuild every TensorRT engine from its ONNX file

Engines are not bit-reproducible across builds: results compare within one build, never across
two. That is why every engine carries the log of the build that produced it.

## Run

    sudo jetson_clocks                 # with a free governor the sweep refuses to measure

    ./runner --dataset $DATA/dataset/imagenet-1k \
             --net $DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine:50:mobilenet \
             --allocation greenctx --submit-threads 1 --fixed 32 \
             --min-inferences 5000 --min-duration 30 --csv /dev/shm/out

    [info] network 'mobilenet': greenctx 8/16 SMs (50%), 32 inputs, 3x224x224 in
    --- network 'mobilenet' ---
      median      0.6752 ms
      p99         0.6816 ms
      tail        0.0066 ms
      1361 inf/s over 30.0 s

`./runner --help` lists every option: `--allocation whole|greenctx|mps-static`, `--net` (repeatable),
`--interferer mem_saxpy|compute_gemm`, `--submit-threads`, `--cpu`, `--priority`, `--wq-scope`,
`--wq-limit`, and the protocol bounds.

## Reproduce

    ./pipeline.py                       # menu: what is measured, then what to run
    ./pipeline.py section               # the families the results section renders
    ./pipeline.py appendix              # the rest, for the history and the tools
    ./pipeline.py mem --runs 3          # one group, fewer replicates
    ./pipeline.py --tables              # regenerate and split the tables, measure nothing

It runs the stages in order — engines, measure, reduce, tables, overview — and stops at the
first failure. `--dry-run` prints every command instead of running it, `--stage measure` stops
after the measurement, `--state` prints what is measured and stops, and a selection can be a
group, a family, or a list of either.

The stages are four commands, and can be run by hand:

    ./utility/sweep.sh --exp section    # measure -> $DATA/bench/<family>/<config>/r<k>_*
    python3 utility/aggregate.py        # reduce  -> $DATA/bench/aggregate/
    python3 utility/results-tables.py > tables.tex      # the section's tables
    python3 utility/split-tables.py tables.tex          # one file per table, plus an index
    python3 utility/overview.py                         # $DATA/bench/OVERVIEW.md

Groups: `section`, `appendix`, `mem`, `all`, or any single family. The MPS static arms need
`--static-chunks 8`, which the pipeline passes by default. Two stages stay outside the sweep
because profiling perturbs what it measures, and the second needs root:

    ./utility/take-traces.sh                            # kernel traces, then sqlite exports
    python3 utility/itf-trace.py $DATA/scratch/traces/trc
    sudo ./utility/soc-traces.sh itf                    # DRAM counters

Both are reachable as `--traces` and `--soc`.
### Every stage goes through the entry point

**Measurements** produce the numbers the thesis reports, and measuring a family produces all of its
measurements — for `interference-additivity` that means the sweep and the MPS static block, whose
job spec cannot express three clients:

    ./pipeline.py section --stage measure           # the sweep, at the campaign protocol
    ./pipeline.py interference-additivity --stage measure   # ... and its MPS static half

**Diagnostics** profile the thing they measure, so their numbers describe mechanisms rather than
latencies; four of the five need root:

    ./pipeline.py section --stage probes            # build and run the probe tools
    ./pipeline.py section --stage soc-budget        # DRAM pressure curve: 4 networks x 4 budgets (root)
    ./pipeline.py section --stage soc               # DRAM counters on the interferer pairs (root)
    ./pipeline.py section --stage ncu               # per-kernel DRAM bytes: diagnostic, then the matrix (root)
    ./pipeline.py section --traces                  # kernel traces, then the per-inference decomposition

The two groups are the same everywhere: in `pipeline.py`, in the menu it prints, in `--help`, and
here. `--stage all` means the quick measurement stages, so a default run is unchanged. `--dry-run`
prints a stage's commands instead of running them, which is how each was checked.

## Protocol

Five independent runs per configuration, each in a fresh process. At least 5 000 inferences and
30 seconds per run. The same 32 preprocessed inputs in every configuration. One submitting
thread. The run is the unit of replication: a reported value is the median of five run medians,
never a pooled percentile.

## Layout

    src/ include/     the runner: config, inputs, stats, the SM allocation hierarchy,
                      the TensorRT engine, the synthetic participant, the run
    utility/          the harness: sweep, aggregate, tables, tools, probes
    utility/README.md the probe tools
    DOC.md            design notes, methodology, traps, data model, table conventions

Measurements live outside the repository, under `$DATA` (default `/mnt/nvme/f.scarfato`), one
directory per family and one per configuration inside it:

    bench/            results, the aggregate, the tables, OVERVIEW.md
    models/           ONNX files and the engines built from them
    dataset/          ImageNet, 50 000 JPEGs of which 32 are used
    labels/           the label file
    scratch/          traces and captures; disposable

`$DATA/bench/OVERVIEW.md` lists every experiment, what it feeds and every configuration with its
replicates, generated from the tree by `utility/overview.py`.
