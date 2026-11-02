#!/usr/bin/env bash
# fs-onoff-smoke.sh - 10 s smoke tests of itfproc's ON/OFF mode on 8 SMs (MPS static 50:50, the
# stressor's partition of the 8+8 geometry), the Steady unit, and the runner's new timestamps.
R=/home/nvidia/f_scarfato/runner
MPS_PIPE=/tmp/mps-smoke MPS_LOG=/tmp/mps-log-smoke STATIC_CHUNKS=8 MPS_MODE=off
source $R/utility/lib/mps.sh
source $R/utility/lib/guards.sh
check_clocks || exit 1
mps_stop; mps_start static >/dev/null && mps_create_partitions 50:50 >/dev/null || exit 1
itf() {   # $1 = tag, rest = itfproc arguments
    local tag=$1; shift
    CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B \
        timeout -s TERM 10 $R/utility/probes/itfproc saxpy "$@" > /dev/shm/fs-smoke-$tag.log 2>&1
    echo "== $tag: $*"
    grep -h "throttle:\|on/off:\|bytes moved" /dev/shm/fs-smoke-$tag.log | tail -4
    grep -h "phase:" /dev/shm/fs-smoke-$tag.log | awk '{d=($4-$3)/1e6; if (NR>1) p=($3-prev)/1e6; prev=$3; s+=d; n++; if (NR>1) {ps+=p; np++}} END {printf "   %d ON spans, mean ON %.2f ms, mean period %.2f ms\n", n, s/n, (np?ps/np:0)}'
}
itf steady --level 1
itf onoff400-d25 --onoff-ms 400 --duty 0.25
itf onoff400-d50 --onoff-ms 400 --duty 0.50
itf onoff400-d100 --onoff-ms 400 --duty 1.00
itf uni4-d50 --mb 32 --onoff-ms 4 --duty 0.50
itf bad --onoff-ms 400 --duty 0.5 --level 0.5
# The runner: MobileNet alone, a short run, then check the new columns.
CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_A $R/runner --submit-threads 1 \
    --dataset /mnt/nvme/f.scarfato/dataset/imagenet-1k --allocation whole \
    --net /mnt/nvme/f.scarfato/models/mobilenetv2-12/engine/mobilenetv2-12.engine:50:mobilenet \
    --fixed 32 --min-inferences 2000 --min-duration 5 --csv /dev/shm/fs-smoke-run > /dev/shm/fs-smoke-run.log 2>&1
echo "== runner exit $?"
head -2 /dev/shm/fs-smoke-run_latencies.csv
awk -F, '$1=="bench"{n++; d=($8-$7)/1e6-$5; if (d<0) d=-d; s+=d; if (d>m) m=d; if (prev && $7<prev) bad++; prev=$8} END {printf "   %d bench rows; |(t_end - t_start) - wall_ms| mean %.4f ms, max %.4f ms; overlapping rows %d\n", n, s/n, m, bad+0}' /dev/shm/fs-smoke-run_latencies.csv
awk -F, '$1=="gpu_ms"{print "   median gpu_ms", $4}' /dev/shm/fs-smoke-run_stats.csv
mps_stop
