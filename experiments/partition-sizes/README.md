# Other partition sizes, two co-runners, power

The counter model with the L2 input, frozen on the 8+8 pairs, tested unchanged on other splits (4+12,
6+10, 12+4 and 8+4 SMs) and with two co-runners (8+4+4, 24 triples), with the dispatch cost growing
with the number of co-runners; and the power and the energy per inference of each network, alone by
size and together. Thesis: `sec:stress-sizes`, `sec:stress-triple`, `sec:stress-energy`.

## Commands

```sh
# 1. Each network's DRAM traffic alone at 4, 6, 10 and 12 SMs (root)
sudo ./measure/profile.sh soc-mps-sizes
python3 analysis/profiles.py

# 2. MobileNet next to two spins on 4+4 SMs: the growth of the dispatch cost (a few minutes)
./measure/sweep.sh --exp dispatch-count --static-chunks 8

# 3. Each network alone at every size, and the pairs on every split, with power (about 7.5 h)
for f in stress-sizes stress-split-412 stress-split-610 stress-split-124; do
    POWER=1 ./measure/sweep.sh --exp $f --static-chunks 8
done

# 4. Freeze the 8+4 and 8+4+4 predictions; this refuses once the triples exist
python3 analysis/geometry.py triple-predict

# 5. The triples, with power (about 4 h)
POWER=1 ./measure/sweep.sh --exp stress-triple --static-chunks 8

# 6. The tables
make tables
```

`POWER=1` samples the board's three INA3221 power rails every 200 ms during each run. The splits are
MPS static shares of the 16 SMs, which round to 2-SM chunks: 25, 37 and 75 % give 4, 6 and 12 SMs.
The dispatch growth also uses MobileNet next to one spin, from l2-presence in
[`../partitioning-mechanisms/`](../partitioning-mechanisms/).

## What it writes

- `$DATA/profiles/soc-mps-sizes/`, reduced to `$DATA/profiles/derived/dram.csv`.
- `$DATA/bench/` families dispatch-count, stress-sizes, stress-split-412, -610, -124, stress-triple,
  each run with its `_power.log`.
- `$DATA/bench/frozen/geometry-triple-predictions.csv`; `analysis/geometry.py` (run by `make
  tables`) writes `$DATA/bench/aggregate/geometry-*.csv`, and the tables are in
  `$DATA/bench/tables/geometry.tex`.
