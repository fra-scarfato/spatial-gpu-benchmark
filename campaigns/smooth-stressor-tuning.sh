#!/usr/bin/env bash
# 2026-10-11-smooth-stressor-tuning.sh - 10 s runs of the time-based SAXPY alone on 8 SMs (MPS static 50:50, the stressor's
# partition of the 8+8 geometry) with the scratch itfproc that has --burst: GB/s against duty and burst.
R=/home/nvidia/f_scarfato/runner
ITF=/mnt/nvme/f.scarfato/scratch/smooth/itfproc
MPS_PIPE=/tmp/mps-smoke MPS_LOG=/tmp/mps-log-smoke STATIC_CHUNKS=8 MPS_MODE=off
source $R/measure/lib/mps.sh
source $R/measure/lib/guards.sh
check_clocks || exit 1
other_gpu_users >/dev/null 2>&1 || { echo "GPU busy"; exit 1; }
mps_stop; mps_start static >/dev/null && mps_create_partitions 50:50 >/dev/null || exit 1
for b in 16 4 2 1; do
    for d in 1.0 0.3 0.1 0.02; do
        CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B \
            timeout -s TERM 10 $ITF saxpy --period-us 100 --duty $d --burst $b > /dev/shm/fs-smooth.log 2>&1
        printf "burst %2s duty %-4s " $b $d
        grep "bytes moved" /dev/shm/fs-smooth.log | tail -1
    done
done
mps_stop
echo FS-SMOOTH-TUNE-END
