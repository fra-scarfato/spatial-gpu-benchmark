# Three models of the p99 slowdown

The smooth model, its temporal version (the probe read over the span of an inference) and the
counter model (each network's DRAM and L2 traffic measured alone, no probe), scored on the twelve
pairs of [`../stressor/`](../stressor/) and then on thirty pairs of six networks no model had seen
(the four in INT8, EfficientNet-B0, ViT-B/16), with two baselines; then each model corrected, their
maximum and a margin. Thesis: `sec:stress-models` (`tab:stress-temporal-stat`, `tab:counter-variants`,
`tab:models-dev`, `tab:models-val`, `tab:models-baselines`) and `sec:stress-bound` (`tab:bound-dev`,
`tab:bound-val`, `tab:stress-margin`, `tab:stress-pessimism`).

## The queues, in the order they ran

| Date | Script | What it ran |
|---|---|---|
| 10-08 | `ncu-l2-traffic-of-networks.sh` | `ncu.sh nets` (root): L2 traffic of the four networks |
| 10-13 | `validation-networks-curves.sh` | Validation networks: `profile.sh kernels-new`; probe calibration-smooth, networks-new-8sm (stress-reporter-new); stress-new (exp26) |
| 10-13 | `validation-pairs.sh` | stress-validation-new (exp27, the thirty pairs) |
| 10-16 | `validation-networks-profiles-root.sh` | (root, run by the user) `ncu.sh nets-new`, `profile.sh soc-mps` |

## Running it

Copy a queue to `$DATA/scratch/` and start it with `nohup bash <name>.sh > <name>.log 2>&1 &`;
the checks before it, and what to change on another board, are in [`../README.md`](../README.md).

Two queues need root (Nsight Compute, the SoC DRAM counters) and are the user's to run. The
validation predictions were frozen before `validation-pairs.sh` ran (`stress.py predict smoothnew`,
`temporal-model.py new-predict`, `counter-model.py new-predict`): measuring the pairs again does not
need them re-frozen, and re-freezing them is a decision, not a step.

## Outputs

- Reporter set `$DATA/bench/stress-reporter-new/`; families stress-new (exp26: the new targets'
  curves and alone rows) and stress-validation-new (exp27: the thirty pairs). Profiles
  `$DATA/profiles/ncu/nets*/` (L2 bytes) and the soc-mps sets (DRAM GB/s), reduced to
  `$DATA/profiles/derived/dram.csv`.
- Models: `analysis/temporal-model.py`, `analysis/counter-model.py` (`new-predict`, `new-compare`),
  `analysis/ensemble.py`. Tables: `make tables` (`stress-tables.py`, `ensemble-tables.py`).
