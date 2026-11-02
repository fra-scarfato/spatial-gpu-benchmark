# Composition of two co-runners

Two smooth stressors on 4 SMs each, next to the probe on the other 8 SMs, at four duties each:
whether the equivalent stress of two co-runners is the sum of each one's. Two light stressors read
less than the sum and two heavy ones more; the ratios with both duties at 75 % or more give the
factor f of [`../partition-sizes/`](../partition-sizes/). Thesis: `sec:stress-compose`
(`tab:stress-compose`).

## Commands

```sh
# 1. The probe's scale and the two stressors, alone and together, in one boot (about 2 h)
export LAYOUT=target OUTROOT=$DATA/bench/stress-reporter-compose
./measure/reporter.sh calibration-smooth
./measure/reporter.sh compose-4sm
unset LAYOUT OUTROOT

# 2. The tables
make tables
```

## What it writes

`$DATA/bench/stress-reporter-compose/`; `analysis/compose.py` (run by `make tables`) writes
`$DATA/bench/aggregate/compose-4sm.csv`, and the table is in `$DATA/bench/tables/stress.tex`.
