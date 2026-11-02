# Experiments

One folder per part of the thesis, each with its README (what it shows, the queues, how to run them,
what they write, what to change on another board) and the queues that measured it, as run on board 1.
The code they share is outside: `measure/` runs on the GPU, `analysis/` reads the data, `probes/` holds
the standalone CUDA programs.

| Folder | Thesis | Tables |
|---|---|---|
| [`partitioning-mechanisms/`](partitioning-mechanisms/) | `results-section.tex`: green contexts against MPS, memory contention and its causes | `tab:iso-rates` ... `tab:mem-presence` |
| [`first-stressors/`](first-stressors/) | none: the Steady and Burst stressors that were replaced | none |
| [`stressor/`](stressor/) | `sec:stress-stressor`: ON/OFF, uniform, smooth, the peak test | `tab:stress-scale` ... `tab:stress-peak` |
| [`models/`](models/) | `sec:stress-models`, `sec:stress-bound`: three models, baselines, correction, maximum, margin | `tab:stress-temporal-stat`, `tab:counter-variants`, `tab:models-*`, `tab:bound-*`, `tab:stress-margin`, `tab:stress-pessimism` |
| [`composition/`](composition/) | `sec:stress-compose`: two stressors together | `tab:stress-compose` |
| [`partition-sizes/`](partition-sizes/) | `sec:stress-sizes`, `sec:stress-triple`, `sec:stress-energy` | `tab:stress-sizes*`, `tab:stress-triple*`, `tab:stress-energy*` |
| [`diagnostics/`](diagnostics/) | none: checks of the setup (MPS hang, governor, probe floor) | none |

Folders are named by topic, not by section number, so a reordered thesis changes this table only.

## Running a queue

A queue is a bash script following the board-campaign template: every GPU step guarded and retried,
any other failure logged as FAILED. Copy it to `$DATA/scratch/` and start it with
`nohup bash <name>.sh > <name>.log 2>&1 &`. Before it:

- the clocks pinned (`sudo jetson_clocks` after every boot, the user's step): `source
  measure/lib/guards.sh; check_clocks` must print a GPU at 1300.5 MHz and the memory at 3199 MHz;
- the binaries matching `probes/PINNED-<host>.sha256` (`sweep.sh` and `reporter.sh` refuse otherwise;
  after an intended rebuild, `pin_binaries`);
- a fresh output directory (`--out`, `OUTROOT`): `sweep.sh` and `reporter.sh` skip runs already on
  disk, so a queue run again over the same data measures nothing.

A sweep run takes about 38 s and a reporter run about 40 s. Queues marked root are the user's to run.

## On another board

- SM shares are MPS percentages or green-context SM counts of the Orin's 16 SMs, in
  `measure/lib/matrix.sh` and in the queues' `reporter.sh` calls.
- Board 2 (P3701-0000, 32 GB) has the same GPU as board 1 (P3701-0005, 64 GB) but other DRAM: a
  calibration, its curves and the networks' readings must come from one session on one board.
- `measure/lib/guards.sh` holds the expected clocks and governor; `probes/PINNED-<host>.sha256` is per
  host.

## The order

The queues in the order they ran. Each was copied from `$DATA/scratch/`, where its log
still is under its original name (`<original name>.log`); a queue that waits for another names the
other's original log. Their paths follow the current layout (`utility/` was split into `measure/`,
`analysis/` and `probes/` on 2026-11-02); the as-run originals are in commit `bd0f7b6` and in
`$DATA/scratch/`. A few diagnostic queues use binaries that exist only in scratch (`itfproc.b2`,
`gpu-latency.b2`) and stay records.

| Date | Script | Original name |
|---|---|---|
| 10-05 | `first-stressors/first-stress-model.sh` | `fs-campaign` |
| 10-06 | `diagnostics/mps-hang-repro.sh` | `fs-hang` |
| 10-06 | `diagnostics/mps-hang-repro-b.sh` | `fs-hang2` |
| 10-08 | `first-stressors/stressor-on-4sm.sh` | `fs-campaign-4sm` |
| 10-08 | `first-stressors/stressor-on-4sm-sweep-rerun.sh` | `fs-campaign-4sm-b` |
| 10-08 | `models/ncu-l2-traffic-of-networks.sh` | `fs-ncu-nets` |
| 10-08 | `first-stressors/probe-on-target-partition.sh` | `fs-reporter-target` |
| 10-08 | `first-stressors/rerun-disturbed-calibration-run.sh` | `fs-rerun-l004` |
| 10-08 | `first-stressors/rerun-disturbed-calibration-run-b.sh` | `fs-queue` |
| 10-09 | `first-stressors/burst-stressor-windowed-probe.sh` | `fs-tbw1` |
| 10-09 | (removed, see `first-stressors/`) | `fs-b1`, `fs-b1b` |
| 10-09 | `diagnostics/gpu-governor-ab.sh` | `fs-gov` |
| 10-09 | `diagnostics/gpu-governor-ab-gate.sh` | `fs-gate` |
| 10-10 | `diagnostics/probe-floor-after-boot.sh` | `fs-boot` |
| 10-10 | `diagnostics/probe-floor-two-states.sh` | `fs-state` |
| 10-10 | `diagnostics/memory-controller-snapshot.sh` | `fs-rootdump` |
| 10-10 | `partitioning-mechanisms/l2-presence-gemm-read.sh` | `fs-l2a` |
| 10-10 | `partitioning-mechanisms/saxpy-4sm-matched-levels.sh` | `fs-levels` |
| 10-10 | `partitioning-mechanisms/l2-against-dram.sh` | `fs-l2d` |
| 10-10 | `stressor/onoff-stressor-smoke-test.sh` | `fs-onoff-smoke` |
| 10-10 | `stressor/onoff-uniform-bandwidth-match.sh` | `fs-match` |
| 10-10 | `stressor/onoff-uniform-bandwidth-match-b.sh` | `fs-onoff-match` |
| 10-10 | `stressor/onoff-uniform-stressors.sh` | `fs-onoff` |
| 10-11 | `stressor/onoff-uniform-models.sh` | `fs-models` |
| 10-11 | `stressor/pairs-8-plus-8.sh` | `fs-m88` |
| 10-11 | `stressor/spin-zero-rows.sh` | `fs-zero` |
| 10-11 | `stressor/smooth-stressor-tuning.sh` | `fs-smooth-tune` |
| 10-11 | `stressor/smooth-stressor.sh` | `fs-smooth` |
| 10-12 | `stressor/peak-test-tuning.sh` | `fs-peak-tune` |
| 10-12 | `stressor/peak-test.sh` | `fs-peak` |
| 10-13 | `partition-sizes/dispatch-two-spins.sh` | `fs-dispatch` |
| 10-13 | `models/validation-networks-curves.sh` | `valnets-a` |
| 10-13 | `models/validation-pairs.sh` | `valnets-b` |
| 10-16 | `models/validation-networks-profiles-root.sh` | `valnets-root` |
| 10-22 | `composition/two-stressors-composition.sh` | `compose` |
| 10-28 | `partition-sizes/partition-sizes-and-two-corunners.sh` | `geometry` |

## Regenerating the numbers

```sh
make tables        # every derived CSV that is out of date, then the tables
make -n tables     # what is out of date, without running it
```

The rules are at the end of the repository's `Makefile`, one per analysis step, each naming what it
reads. The tables land in `$DATA/bench/tables/{stress,ensemble,geometry}.tex`, one block
per `stress-model.tex` label, ready to paste; the prose numbers follow each block as comments.

The validation predictions are frozen by `stress.py predict smoothnew`, `temporal-model.py new-predict`
and `counter-model.py new-predict`, before the pairs were read. They live in `$DATA/bench/frozen/`,
hashed in `analysis/frozen.sha256` (`make check`), and are inputs of `make`, never targets: re-running them after a change of model overwrites the frozen files, which is a decision to record, not a routine step.
