#!/usr/bin/env bash
# fs-peak-tune.sh - 10 s runs of the time-based SAXPY alone on 8 SMs at duty 0.1, burst 2: GB/s against the period (peak length).
# partition of the 8+8 geometry) with the scratch itfproc that has --burst: GB/s against duty and burst.
R=/home/nvidia/f_scarfato/runner
ITF=$R/utility/probes/itfproc
MPS_PIPE=/tmp/mps-smoke MPS_LOG=/tmp/mps-log-smoke STATIC_CHUNKS=8 MPS_MODE=off
source $R/utility/lib/mps.sh
source $R/utility/lib/guards.sh
check_clocks || exit 1
other_gpu_users >/dev/null 2>&1 || { echo "GPU busy"; exit 1; }
mps_stop; mps_start static >/dev/null && mps_create_partitions 50:50 >/dev/null || exit 1
for p in 100 1000 3000 10000 30000; do
    for d in 0.1; do
        CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B \
            timeout -s TERM 10 $ITF saxpy --period-us $p --duty $d --burst 2 > /dev/shm/fs-smooth.log 2>&1
        printf "period %6s us duty %-4s " $p $d
        grep "bytes moved" /dev/shm/fs-smooth.log | tail -1
    done
done
mps_stop
echo FS-PEAK-TUNE-END
