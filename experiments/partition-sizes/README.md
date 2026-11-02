# Other partition sizes, two co-runners, power

The counter model with the L2 input, frozen on the 8+8 pairs, tested unchanged on other splits
(4+12, 6+10, 12+4, and 8+4) and with two co-runners (8+4+4, 24 triples), with the presence cost
growing with the number of co-runners; and the power and energy per inference of each network,
alone by size and together. Thesis: `sec:stress-sizes` (`tab:stress-sizes`, `tab:stress-sizes-sens`),
`sec:stress-triple` (`tab:stress-triple`, `tab:stress-triple-where`), `sec:stress-energy`
(`tab:stress-energy`, `tab:stress-energy-together`).

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-13 | `dispatch-two-spins.sh` | dispatch-count (exp25): MobileNet next to two spins on 4+4 |
| 10-28 | `partition-sizes-and-two-corunners.sh` | Partition sizes and two co-runners: stress-split-412/-610/-124, stress-sizes, stress-triple, power-check (exp28-34); `geometry.py` |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

`dispatch-two-spins.sh` measures MobileNet next to two spins on 4+4; against one spin (l2-presence,
exp19, in [`../partitioning-mechanisms/`](../partitioning-mechanisms/)) it gives the growth of the
presence cost with a second co-runner (g in `analysis/geometry.py`), which the triple model uses.
The root profiles of the new sizes (`profile.sh soc-mps-sizes`) are the user's step.

## Outputs

Families stress-split-412, -610, -124, stress-sizes, stress-triple and power-check (exp28-34) and
dispatch-count (exp25). `analysis/geometry.py` (splits, triple-current, power) writes
`$DATA/bench/aggregate/geometry-*.csv`; `make tables` prints the six tables (`geometry-tables.py`).
The triple predictions are frozen (`$DATA/bench/frozen/geometry-triple-predictions.csv`).

## On another board

The splits are MPS static shares of 16 SMs, which round to 2-SM chunks: 25, 37 and 75 % give 4, 6
and 12 SMs (`measure/lib/matrix.sh`, the stress-split arms).
