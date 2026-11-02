# The partitioning mechanisms

Green contexts against MPS (static, ATP, default) and no partitioning: the rate and dispersion of
each network alone and in pairs, the submitting thread, the work queues, and the memory contention
that remains between disjoint partitions, with its causes: the cost of dispatching next to a
resident kernel, the DRAM and the L2. Thesis: `results-section.tex`, every table from
`tab:iso-rates` to `tab:mem-presence`.

## Commands

```sh
# 1. The networks alone and together, under every mechanism (about 6 h)
./measure/sweep.sh --exp section --static-chunks 8
M=$DATA/models
./measure/mps-additivity-block.sh --block 5 $M/mobilenetv2-12/engine/mobilenetv2-12.engine:mobilenet \
    $M/retinanet-9/engine/retinanet-9.engine:retinanet $M/inception_v3-onnx-float/engine/inception_v3.engine:inception

# 2. The causes: each target next to spin at several launch rates, and MobileNet next to co-runners
#    that use the L2 differently (about 2.5 h)
for f in l2-presence l2-gemm-sets l2-read l2-vs-dram; do ./measure/sweep.sh --exp $f --static-chunks 8; done

# 3. CUDA traces: where an inference's time goes, and the kernels per inference (no root)
./measure/profile.sh traces
./measure/profile.sh kernels

# 4. The DRAM counters (root): the GB/s calibration, then the networks, the co-runners and the L2 set
sudo ./measure/profile.sh calibration
sudo ./measure/profile.sh soc-budget
sudo ./measure/profile.sh soc-itf
sudo ./measure/profile.sh footprint
sudo ./measure/profile.sh soc-l2

# 5. The tables
make tables
```

`--static-chunks 8` is the Orin's 16 SMs in 2-SM chunks; without it the MPS static runs are skipped.
The additivity family needs three partitions, which one sweep job cannot hold, so its MPS half has
its own script.
`--exp section` also lists 40 configurations of interference-additivity, workqueue-declaration and
workqueue-aux-streams that `measure/lib/matrix.sh` defines but no table reads and that were never
measured (200 runs, about 2 h); `--dry-run` lists them.

## What it writes

- `$DATA/bench/<family>/<config>/r<k>_*`: the families of `SECTION_FAMILIES` in `measure/sweep.sh`,
  and l2-presence, l2-gemm-sets, l2-read, l2-vs-dram. Every arm is a line of `measure/lib/matrix.sh`.
- `$DATA/profiles/<set>/`, reduced by `analysis/profiles.py` (run by `make tables`) to
  `$DATA/profiles/derived/*.csv`.
- `$DATA/bench/tables/results.tex`, and one file per table under `results/`.
