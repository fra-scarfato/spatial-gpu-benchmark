# Three models of the p99 slowdown

The smooth model, its temporal version (the probe read over the span of one inference of the target)
and the counter model (each network's DRAM and L2 traffic measured alone, no probe), scored on the
twelve pairs of [`../stressor/`](../stressor/), then frozen and scored on thirty pairs of six networks
that no model had seen (the four in INT8, EfficientNet-B0, ViT-B/16), next to two baselines; then
each model corrected, their maximum, and a margin. Thesis: `sec:stress-models` and
`sec:stress-bound`.

## Commands

```sh
# 1. Each network's traffic alone (root, then no root): L2 bytes, DRAM GB/s, kernels per inference
sudo ./measure/ncu.sh nets
sudo ./measure/ncu.sh nets-new
sudo ./measure/profile.sh soc-mps
sudo ./measure/profile.sh soc-new
./measure/profile.sh kernels-new
python3 analysis/profiles.py

# 2. The probe and the new networks' curves, in one boot (about 3.5 h)
export LAYOUT=target OUTROOT=$DATA/bench/stress-reporter-new
./measure/reporter.sh calibration-smooth
./measure/reporter.sh networks-new-8sm
unset LAYOUT OUTROOT
./measure/sweep.sh --exp stress-new --static-chunks 8

# 3. Fit the counter model on the twelve pairs, then freeze every prediction for the thirty pairs
python3 analysis/counter-model.py
python3 analysis/stress.py predict smoothnew
python3 analysis/temporal-model.py new-predict
python3 analysis/counter-model.py new-predict

# 4. The thirty pairs (about 1.7 h)
./measure/sweep.sh --exp stress-validation-new --static-chunks 8

# 5. The tables
make tables
```

The new networks' engines (INT8, EfficientNet, ViT) come from `make -f engines.mk`; the INT8 ones are
calibrated on other images than the measured ones (`measure/int8-calib.py`).

## What it writes

- `$DATA/profiles/ncu/nets*/`, `$DATA/profiles/soc-mps/`, `soc-new/`, `kernels-new/`, reduced to
  `$DATA/profiles/derived/dram.csv`.
- `$DATA/bench/stress-reporter-new/`, `$DATA/bench/stress-new/` and
  `$DATA/bench/stress-validation-new/`.
- `$DATA/bench/aggregate/counter-model-coeffs.csv` (the fit on the twelve pairs) and the frozen
  predictions in `$DATA/bench/frozen/`.
- `$DATA/bench/tables/stress.tex` (`tab:stress-temporal-stat`, `tab:counter-variants`) and
  `ensemble.tex` (`tab:models-*`, `tab:bound-*`, `tab:stress-margin`, `tab:stress-pessimism`).
