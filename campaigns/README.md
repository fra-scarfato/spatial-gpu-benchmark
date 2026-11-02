# Campaigns of the stress chapter

Each script here is the queue that was run on board 1, copied unchanged from
`$DATA/scratch/` (2026-11-02), where its log still is (`<name>.log`, same name). They are the record
of what was run, in what order and with which guards; most follow the queue template of the
board-campaign skill (every step guarded and retried, any other failure logs FAILED). Their paths were
updated to the current layout on 2026-11-02 (`utility/` split into `measure/`, `analysis/`,
`probes/`); the as-run originals are in commit `111269e` and in `$DATA/scratch/`. Since `sweep.sh` and `reporter.sh` skip runs already on disk, measuring
again needs a fresh output directory (`--out`, `OUTROOT`). To run one again, copy it to `$DATA/scratch/` and start it with
`nohup bash <name>.sh > <name>.log 2>&1 &`.

Families are `$DATA/bench/<family>/`, reporter sets `$DATA/bench/stress-reporter*/`, profile sets
`$DATA/profiles/<set>/`. The aggregate keys (exp13 ...) are in `analysis/lib/families.py`.

## The ordered log

Each queue is named by its date and what it measured. Its log is in `$DATA/scratch/` under its
original name (`<original name>.log`), as are the queues as run. A queue that waits for another
names the other's original log.

| Date | Script | Original name | What it ran |
|---|---|---|---|
| 10-05 | `2026-10-05-first-stress-model.sh` | `fs-campaign` | First stress model: probe calibration and networks; families stress-sensitivity (exp13) and stress-validation (exp14) |
| 10-06 | `2026-10-06-mps-hang-repro.sh` | `fs-hang` | Reproduction of an MPS hang (Inception + RetinaNet), with gdb backtraces; no data |
| 10-06 | `2026-10-06-mps-hang-repro-b.sh` | `fs-hang2` | The same, second variant; no data |
| 10-08 | `2026-10-08-stressor-on-4sm.sh` | `fs-campaign-4sm` | Stressor on 4 SMs: probe calibration-4sm and probes-4sm; stress-sensitivity-4sm (exp15) |
| 10-08 | `2026-10-08-stressor-on-4sm-sweep-rerun.sh` | `fs-campaign-4sm-b` | The stress-sensitivity-4sm sweep again, after a guard abort |
| 10-08 | `2026-10-08-ncu-l2-traffic-of-networks.sh` | `fs-ncu-nets` | `ncu.sh nets` (root): L2 traffic of the four networks |
| 10-08 | `2026-10-08-probe-on-target-partition.sh` | `fs-reporter-target` | The probe on the target's partition (LAYOUT=target): stress-reporter-target |
| 10-08 | `2026-10-08-rerun-disturbed-calibration-run.sh` | `fs-rerun-l004` | Re-run of one calibration run (saxpy4-L004 run 2) disturbed by a co-tenant |
| 10-08 | `2026-10-08-rerun-disturbed-calibration-run-b.sh` | `fs-queue` | The same re-run, with the binary the rest of that calibration used |
| 10-09 | `2026-10-09-burst-stressor-windowed-probe.sh` | `fs-tbw1` | Burst stressor (time-based SAXPY) read by the windowed probe: calibration-tb, networks; stress-sensitivity-tb (exp17); `stress.py tbw`. Not in the thesis |
| 10-09 | (removed) | `fs-b1`, `fs-b1b` | Burst-scrambled stressor: calibration-tbs; stress-sensitivity-tbs (exp18); `stress.py tbsw`; the cross-board diagnostic. Not in the thesis; removed on 2026-11-02, the queues stay in `$DATA/scratch/` and commit `111269e` |
| 10-09 | `2026-10-09-gpu-governor-ab.sh` | `fs-gov` | GPU governor A/B (nvhost_podgov against performance) on a probe set |
| 10-09 | `2026-10-09-gpu-governor-ab-gate.sh` | `fs-gate` | Stopped the Burst-scrambled queue after its diagnostic, to run the governor A/B |
| 10-10 | `2026-10-10-probe-floor-after-boot.sh` | `fs-boot` | The probe's floor right after a boot |
| 10-10 | `2026-10-10-probe-floor-two-states.sh` | `fs-state` | What flips the probe's floor between its two states |
| 10-10 | `2026-10-10-memory-controller-snapshot.sh` | `fs-rootdump` | A read-only debugfs snapshot of the memory controller, for the two floor states |
| 10-10 | `2026-10-10-l2-presence-gemm-read.sh` | `fs-l2a` | `profile.sh kernels`; l2-presence, l2-gemm-sets, l2-read (exp19-21) |
| 10-10 | `2026-10-10-saxpy-4sm-matched-levels.sh` | `fs-levels` | Steady SAXPY on 4 SMs: the levels matched to 22.3 and 8.0 GB/s |
| 10-10 | `2026-10-10-l2-against-dram.sh` | `fs-l2d` | Probe l2-vs-dram; l2-vs-dram (exp22) |
| 10-10 | `2026-10-10-onoff-stressor-smoke-test.sh` | `fs-onoff-smoke` | ON/OFF stressor smoke tests |
| 10-10 | `2026-10-10-onoff-uniform-bandwidth-match.sh` | `fs-match` | ON/OFF, uniform and Steady settings matched in mean bandwidth |
| 10-10 | `2026-10-10-onoff-uniform-bandwidth-match-b.sh` | `fs-onoff-match` | The same matching, for given argument lists |
| 10-10 | `2026-10-10-onoff-uniform-stressors.sh` | `fs-onoff` | stress-onoff (exp23): the ON/OFF and uniform stressors next to the four targets |
| 10-11 | `2026-10-11-onoff-uniform-models.sh` | `fs-models` | Probe calibration-onoff and networks; `stress.py onoff`, `uniform` |
| 10-11 | `2026-10-11-pairs-8-plus-8.sh` | `fs-m88` | The 8+8 configuration: probe networks-8sm; stress-validation-8 (exp24, the twelve pairs); `stress.py onoff88`, `uniform88` |
| 10-11 | `2026-10-11-spin-zero-rows.sh` | `fs-zero` | The spin rows (the zero of each scale) of stress-onoff |
| 10-11 | `2026-10-11-smooth-stressor-tuning.sh` | `fs-smooth-tune` | The smooth stressor's settings |
| 10-11 | `2026-10-11-smooth-stressor.sh` | `fs-smooth` | The smooth stressor's curves (stress-onoff smooth-D); `stress.py smooth88` |
| 10-12 | `2026-10-12-peak-test-tuning.sh` | `fs-peak-tune` | The peak test's settings |
| 10-12 | `2026-10-12-peak-test.sh` | `fs-peak` | The peak test (stress-onoff peak-T) |
| 10-13 | `2026-10-13-dispatch-two-spins.sh` | `fs-dispatch` | dispatch-count (exp25): MobileNet next to two spins on 4+4 |
| 10-13 | `2026-10-13-validation-networks-curves.sh` | `valnets-a` | Validation networks: `profile.sh kernels-new`; probe calibration-smooth, networks-new-8sm (stress-reporter-new); stress-new (exp26) |
| 10-13 | `2026-10-13-validation-pairs.sh` | `valnets-b` | stress-validation-new (exp27, the thirty pairs) |
| 10-16 | `2026-10-16-validation-networks-profiles-root.sh` | `valnets-root` | (root, run by the user) `ncu.sh nets-new`, `profile.sh soc-mps` |
| 10-22 | `2026-10-22-two-stressors-composition.sh` | `compose` | Two smooth SAXPY on 4 SMs each: probe calibration-smooth, compose-4sm |
| 10-28 | `2026-10-28-partition-sizes-and-two-corunners.sh` | `geometry` | Partition sizes and two co-runners: stress-split-412/-610/-124, stress-sizes, stress-triple, power-check (exp28-34); `geometry.py` |

## Regenerating the numbers

```sh
make tables        # every derived CSV that is out of date, then the tables
make -n tables     # what is out of date, without running it
```

The rules are at the end of the repository's `Makefile`, one per analysis step, each naming what it
reads. The tables land in `$DATA/bench/tables/{stress,ensemble,geometry}.tex`, one block
per `stress-model.tex` label, ready to paste; the prose numbers follow each block as comments.

The validation predictions are frozen by `stress.py predict smoothnew`, `infwin-model.py new-predict`
and `counter-model.py new-predict`, before the pairs were read. They live in `$DATA/bench/frozen/`,
hashed in `analysis/frozen.sha256` (`make check`), and are inputs of `make`, never targets: re-running them after a change of model overwrites the frozen files, which is a decision to record, not a routine step.
