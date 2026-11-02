# The choice of the stressor

Which synthetic stressor gives a probe scale and sensitivity curves that predict the slowdown of a
network next to another: ON/OFF SAXPY (400 ms periods), uniform SAXPY (short pulses at the same mean
bandwidth) and the smooth stressor (100 us periods), each read by the probe and run next to the four
targets; the twelve pairs of the four networks on 8+8 SMs as ground truth; and the peak test (duty
10 %, peaks of 0.01-3 ms), in which the target's p99 responds to the stress over one inference.
Thesis: `sec:stress-stressor` (`tab:stress-scale` to `tab:stress-peak`).

## Commands

```sh
# 1. The probe next to every stressor and next to each network, in one boot (about 1.5 h)
export LAYOUT=target OUTROOT=$DATA/bench/stress-reporter-onoff
./measure/reporter.sh calibration-onoff
./measure/reporter.sh networks-8sm
unset LAYOUT OUTROOT

# 2. Each target next to every stressor, including spin (the zero) and the peak test (about 4 h)
./measure/sweep.sh --exp stress-onoff --static-chunks 8

# 3. Freeze the predictions of each stressor's model, before the pairs are measured
for g in onoff88 uniform88 smooth88 smoothx88; do python3 analysis/stress.py predict $g; done

# 4. The twelve pairs, each network on 8 SMs (about 40 min)
./measure/sweep.sh --exp stress-validation-8 --static-chunks 8

# 5. The tables
make tables
```

`LAYOUT=target` puts the probe on the target's partition, the stressor or the network on the other 8
SMs. The stressors' settings are the arms of stress-onoff in `measure/lib/matrix.sh` and the
configurations of `calibration-onoff` in `measure/reporter.sh`.

## What it writes

- `$DATA/bench/stress-reporter-onoff/`: the probe's readings. `$DATA/bench/stress-onoff/` and
  `$DATA/bench/stress-validation-8/`: the targets.
- `$DATA/bench/frozen/stress-{calibration,corunners,sensitivity,predictions}-<model>.csv`: each
  model, frozen by `stress.py predict`.
- `$DATA/bench/tables/stress.tex`: this part's tables; the twelve pairs are also the development set
  of [`../models/`](../models/).
