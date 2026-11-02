#!/usr/bin/env bash
# fs-levels.sh - Steady SAXPY on 4 SMs (MPS static, 50:25 as in the sweeps) at a few levels, 10 s
# each, for the GB/s that itfproc reports: picks the levels matched to 22.3 and 8.0 GB/s for 1d.
R=/home/nvidia/f_scarfato/runner
MPS_PIPE=/tmp/mps-levels MPS_LOG=/tmp/mps-log-levels STATIC_CHUNKS=8 MPS_MODE=off
source $R/utility/lib/mps.sh
source $R/utility/lib/guards.sh
check_clocks || exit 1
mps_stop; mps_start static >/dev/null && mps_create_partitions 50:25 >/dev/null || exit 1
for l in "$@"; do
    CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B \
        timeout -s TERM 10 $R/utility/probes/itfproc saxpy --level $l > /dev/shm/fs-lvl.log 2>&1
    echo "level $l: $(grep -h 'bytes moved' /dev/shm/fs-lvl.log | tail -1) $(grep -h 'steady\|time-based' /dev/shm/fs-lvl.log | head -1)"
done
mps_stop
