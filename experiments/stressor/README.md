# The choice of the stressor

Which synthetic stressor gives a probe scale and sensitivity curves that predict the twelve pairs:
ON/OFF SAXPY (400 ms periods), uniform SAXPY (short pulses, the same mean bandwidth), and the smooth
stressor (100 us periods), each next to the probe and next to the four targets, then the twelve 8+8
pairs as ground truth; and the peak test (duty 10 %, peaks of 0.01-3 ms), which shows that the
target's p99 responds to the stress over one inference. Thesis: `sec:stress-stressor`
(`tab:stress-scale` to `tab:stress-peak`).

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-10 | `onoff-stressor-smoke-test.sh` | ON/OFF stressor smoke tests |
| 10-10 | `onoff-uniform-bandwidth-match.sh` | ON/OFF, uniform and Steady settings matched in mean bandwidth |
| 10-10 | `onoff-uniform-bandwidth-match-b.sh` | The same matching, for given argument lists |
| 10-10 | `onoff-uniform-stressors.sh` | stress-onoff (exp23): the ON/OFF and uniform stressors next to the four targets |
| 10-11 | `onoff-uniform-models.sh` | Probe calibration-onoff and networks; `stress.py onoff`, `uniform` |
| 10-11 | `pairs-8-plus-8.sh` | The 8+8 configuration: probe networks-8sm; stress-validation-8 (exp24, the twelve pairs); `stress.py onoff88`, `uniform88` |
| 10-11 | `spin-zero-rows.sh` | The spin rows (the zero of each scale) of stress-onoff |
| 10-11 | `smooth-stressor-tuning.sh` | The smooth stressor's settings |
| 10-11 | `smooth-stressor.sh` | The smooth stressor's curves (stress-onoff smooth-D); `stress.py smooth88` |
| 10-12 | `peak-test-tuning.sh` | The peak test's settings |
| 10-12 | `peak-test.sh` | The peak test (stress-onoff peak-T) |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

The tuning and matching queues found the stressors' settings on this board (period, duty, pulse
size); the settings they found are the arms of stress-onoff in `measure/lib/matrix.sh`.

## Outputs

- Reporter sets `$DATA/bench/stress-reporter-onoff/`; families stress-onoff (exp23: the stressors,
  the spin zero, smooth-D, peak-T) and stress-validation-8 (exp24: the twelve pairs).
- Models: `python3 analysis/stress.py predict <geometry>` (onoff88, uniform88, smooth88, smoothx88)
  writes the frozen predictions in `$DATA/bench/frozen/`, before the pairs are read; `compare` scores
  them. `make tables` re-runs the comparisons and `analysis/tables/stress-tables.py`.
- The twelve pairs are also the development set of [`../models/`](../models/).

## On another board

The scale and the curves must be measured in one session, and the probe's floor drifts between
sessions, so never reuse a calibration across boots or boards.
