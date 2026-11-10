# Periodic loads

The networks released like cameras instead of back to back, on partitions of the GPU (MPS static) and
on the shared GPU (MPS default, and no MPS): GPU time, response time and its queueing, the highest load
every network sustains within its latency budget, and the energy at equal work; at the maximum and at
two lower GPU clocks. The schedule is `Periodic` in `include/config.hpp`; the families are described in
`measure/lib/matrix.sh`, the analysis in `analysis/periodic.py`.

```sh
# 1. Each network alone on the whole GPU (about 3 h)
POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-alone16 --static-chunks 8
python3 analysis/periodic.py periodic-alone16

# 2. Two networks at three GPU clocks, on 8+8 and on the shared GPU (about 3 h per clock). The clock is
#    locked by root before each block, and GPU_FREQ makes the guard refuse any other.
sudo ./measure/gpu-freq.sh 1300500000
GPU_FREQ=1300500000 POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-freq --only "-f1300$" --static-chunks 8
sudo ./measure/gpu-freq.sh 816000000
GPU_FREQ=816000000 POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-freq --only "-f816$" --static-chunks 8
sudo ./measure/gpu-freq.sh 612000000
GPU_FREQ=612000000 POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-freq --only "-f612$" --static-chunks 8
sudo jetson_clocks
python3 analysis/periodic.py periodic-freq

# 3. The pairs on 8+8 and the shared GPU (about 25 h), then the triples on 8+4+4 (about 27 h)
POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-88 --static-chunks 8
python3 analysis/periodic.py periodic-88
POWER=1 RUN_TIMEOUT=900 ./measure/sweep.sh --exp periodic-844 --static-chunks 8
python3 analysis/periodic.py periodic-844
```

The budgets come from the networks alone on the whole GPU in `../partition-sizes/` (energy-tenants).
