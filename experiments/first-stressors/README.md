# The first stressors

The stressors tried before the chapter's: Steady (SAXPY at a fixed bandwidth, `--level`) on 8 and on
4 SMs, read by the windowed probe on its own partition and on the target's, and Burst (SAXPY ON for a
duty of each 1 ms period, `--duty`). Their models under-predicted every pair (Steady) or were
replaced by the ON/OFF, uniform and smooth stressors of [`../stressor/`](../stressor/). Thesis: none
of these numbers is in the stress chapter; they are the record of what was replaced.

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-05 | `first-stress-model.sh` | First stress model: probe calibration and networks; families stress-sensitivity (exp13) and stress-validation (exp14) |
| 10-08 | `stressor-on-4sm.sh` | Stressor on 4 SMs: probe calibration-4sm and probes-4sm; stress-sensitivity-4sm (exp15) |
| 10-08 | `stressor-on-4sm-sweep-rerun.sh` | The stress-sensitivity-4sm sweep again, after a guard abort |
| 10-08 | `probe-on-target-partition.sh` | The probe on the target's partition (LAYOUT=target): stress-reporter-target |
| 10-08 | `rerun-disturbed-calibration-run.sh` | Re-run of one calibration run (saxpy4-L004 run 2) disturbed by a co-tenant |
| 10-08 | `rerun-disturbed-calibration-run-b.sh` | The same re-run, with the binary the rest of that calibration used |
| 10-09 | `burst-stressor-windowed-probe.sh` | Burst stressor (time-based SAXPY) read by the windowed probe: calibration-tb, networks; stress-sensitivity-tb (exp17); `stress.py tbw`. Not in the thesis |
| 10-09 | (removed) | Burst-scrambled stressor: calibration-tbs; stress-sensitivity-tbs (exp18); `stress.py tbsw`; the cross-board diagnostic. Not in the thesis; removed on 2026-11-02, the queues stay in `$DATA/scratch/` and commit `bd0f7b6` |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

`burst-stressor-windowed-probe.sh` first copied a scratch build over the co-runner binary; that step
is gone (the time-based mode is in the pinned binary), the original is in commit `bd0f7b6`.

## Outputs

- Reporter sets `$DATA/bench/stress-reporter*/` (calibration, calibration-4sm, probes-4sm,
  stress-reporter-target, stress-reporter-tbw); families stress-sensitivity (exp13),
  stress-validation (exp14), stress-sensitivity-4sm (exp15), stress-sensitivity-tb (exp17).
- Models: `python3 analysis/stress.py predict|compare <geometry>` with 8sm, 4sm, 8smt and tbw
  (`docs/MEASUREMENTS.md` gives each geometry's descriptive name). No thesis table reads them.
