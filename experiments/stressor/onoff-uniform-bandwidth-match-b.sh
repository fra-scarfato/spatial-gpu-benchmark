#!/usr/bin/env bash
# 10 s runs of the ON/OFF, uniform and Steady stressors on 8 SMs (MPS static 50:50, the stressor's
# partition of the 8+8 geometry), to find the settings that match their mean bandwidth.
R=/home/nvidia/f_scarfato/runner
MPS_PIPE=/tmp/mps-smoke MPS_LOG=/tmp/mps-log-smoke STATIC_CHUNKS=8 MPS_MODE=off
source $R/measure/lib/mps.sh
source $R/measure/lib/guards.sh
check_clocks || exit 1
mps_stop; mps_start static >/dev/null && mps_create_partitions 50:50 >/dev/null || exit 1
itf() {   # $1 = tag, rest = corunner arguments
    local tag=$1; shift
    CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B \
        timeout -s TERM 10 $R/probes/corunner saxpy "$@" > /dev/shm/fs-smoke-$tag.log 2>&1
    echo "== $tag: $*"
    grep -h "throttle:\|on/off:\|bytes moved" /dev/shm/fs-smoke-$tag.log | tail -4
    grep -h "phase:" /dev/shm/fs-smoke-$tag.log | awk '{d=($4-$3)/1e6; if (NR>1) p=($3-prev)/1e6; prev=$3; s+=d; n++; if (NR>1) {ps+=p; np++}} END {printf "   %d ON spans, mean ON %.2f ms, mean period %.2f ms\n", n, s/n, (np?ps/np:0)}'
}
for a in "$@"; do itf "m-${a// /_}" $a; done
mps_stop
