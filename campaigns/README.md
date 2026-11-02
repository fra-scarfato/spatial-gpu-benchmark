# Campaigns of the stress chapter

Each script here is the queue that was run on board 1, copied unchanged from
`$DATA/scratch/` (2026-11-02), where its log still is (`<name>.log`, same name). They are the record
of what was run, in what order and with which guards; most follow the queue template of the
board-campaign skill (every step guarded and retried, any other failure logs FAILED). Paths inside
are those of the day. To run one again, copy it to `$DATA/scratch/` and start it with
`nohup bash <name>.sh > <name>.log 2>&1 &`.

Families are `$DATA/bench/<family>/`, reporter sets `$DATA/bench/stress-reporter*/`, profile sets
`$DATA/profiles/<set>/`. The aggregate keys (exp13 ...) are in `analysis/lib/families.py`.

## The ordered log

| Date | Script | What it ran |
|---|---|---|
| 10-05 | `fs-campaign.sh` | First stress model: reporter calibration and networks; families stress-sensitivity (exp13) and stress-validation (exp14) |
| 10-06 | `fs-hang.sh`, `fs-hang2.sh` | Reproduction of an MPS hang (Inception + RetinaNet), with gdb backtraces; no data |
| 10-08 | `fs-campaign-4sm.sh`, `fs-campaign-4sm-b.sh` | Stressor on 4 SMs: reporter calibration-4sm, probes-4sm; stress-sensitivity-4sm (exp15) |
| 10-08 | `fs-ncu-nets.sh` | `ncu.sh nets` (root): L2 traffic of the four networks |
| 10-08 | `fs-reporter-target.sh` | The reporter on the target's partition (LAYOUT=target): stress-reporter-target |
| 10-08 | `fs-rerun-l004.sh`, `fs-queue.sh` | Re-run of saxpy4-L004 run 2, disturbed by a co-tenant |
| 10-09 | `fs-tbw1.sh` | Burst stressor, windowed reporter: calibration-tb, networks; stress-sensitivity-tb (exp17); `stress.py predict/compare tbw` |
| 10-09 | `fs-b1.sh`, `fs-b1b.sh` | Burst-scrambled: calibration-tbs; stress-sensitivity-tbs (exp18); `stress.py tbsw` (b1b adds the resident no-traffic diagnostic) |
| 10-09 | `fs-gov.sh`, `fs-gate.sh` | GPU governor A/B (nvhost_podgov against performance) on a reporter set |
| 10-10 | `fs-boot.sh`, `fs-state.sh`, `fs-rootdump.sh` | The reporter floor's two states after a boot; a read-only debugfs snapshot |
| 10-10 | `fs-l2a.sh` | `profile.sh kernels`; l2-presence, l2-gemm-sets, l2-read (exp19-21) |
| 10-10 | `fs-levels.sh` | Steady SAXPY on 4 SMs: the levels matched to 22.3 and 8.0 GB/s |
| 10-10 | `fs-l2d.sh` | Reporter l2-vs-dram; l2-vs-dram (exp22) |
| 10-10 | `fs-onoff-smoke.sh`, `fs-match.sh`, `fs-onoff-match.sh` | ON/OFF stressor smoke tests and bandwidth matching |
| 10-10 | `fs-onoff.sh` | stress-onoff (exp23): ON/OFF and uniform stressors |
| 10-11 | `fs-models.sh` | Reporter calibration-onoff and networks; `stress.py onoff`, `uniform` |
| 10-11 | `fs-m88.sh` | The 8+8 configuration: reporter networks-8sm; stress-validation-8 (exp24, the twelve pairs); `stress.py onoff88`, `uniform88` |
| 10-11 | `fs-zero.sh` | The spin rows of stress-onoff |
| 10-11 | `fs-smooth-tune.sh`, `fs-smooth.sh` | The smooth stressor: tuning, then its curves (stress-onoff smooth-D); `stress.py smooth88` |
| 10-12 | `fs-peak-tune.sh`, `fs-peak.sh` | The peak test (stress-onoff peak-T) |
| 10-13 | `fs-dispatch.sh` | dispatch-count (exp25): MobileNet next to two spins on 4+4 |
| 10-13 | `valnets-a.sh` | Validation networks: `profile.sh kernels-new`; reporter calibration-smooth, networks-new-8sm (stress-reporter-new); stress-new (exp26) |
| 10-13 | `valnets-b.sh` | stress-validation-new (exp27, the thirty pairs) |
| 10-16 | `valnets-root.sh` | (root, run by the user) `ncu.sh nets-new`, `profile.sh soc-mps` |
| 10-22 | `compose.sh` | Two SAXPY on 4 SMs each: reporter calibration-smooth, compose-4sm |
| 10-28 | `geometry.sh` | Partition sizes and two co-runners: stress-split-412/-610/-124, stress-sizes, stress-triple, power-check (exp28-34); `geometry.py` |

## Regenerating the numbers

```sh
make tables        # every derived CSV that is out of date, then the tables
make -n tables     # what is out of date, without running it
```

The rules are at the end of the repository's `Makefile`, one per analysis step, each naming what it
reads. The tables land in `$DATA/bench/aggregate/tables-{stress,ensemble,geometry}.tex`, one block
per `stress-model.tex` label, ready to paste; the prose numbers follow each block as comments.

The validation predictions are frozen by `stress.py predict smoothnew`, `infwin-model.py new-predict`
and `counter-model.py new-predict`, before the pairs were read. They are inputs of
`make`, never targets: re-running them after a change of model overwrites the frozen files, which is a decision to record, not a routine step.
