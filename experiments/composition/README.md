# Composition of two co-runners

Two smooth stressors on 4 SMs each next to the probe on the other 8 SMs, by duty: whether the
equivalent stress of two co-runners is the sum of each one's. Two light stressors read less than the
sum (0.80 at 25 % each) and two heavy ones more (1.38 at 100 % each); the ratios with both duties at
75 % or more give the factor f of [`../partition-sizes/`](../partition-sizes/). Thesis:
`sec:stress-compose` (`tab:stress-compose`).

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-22 | `two-stressors-composition.sh` | Two smooth SAXPY on 4 SMs each: probe calibration-smooth, compose-4sm |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

## Outputs

Reporter sets `calibration-smooth` and `compose-4sm` in `$DATA/bench/`; `analysis/compose.py` writes
`$DATA/bench/aggregate/compose-4sm.csv`, and `make tables` prints `tab:stress-compose`
(`stress-tables.py`).

## On another board

The 4+4+8 split is MPS static at 25, 25 and 50 % of the Orin's 16 SMs.
