#!/usr/bin/env bash
# Reproduce the MPS hang: job 1 = Inception target + RetinaNet co-runner (killed with SIGTERM, as the
# sweep does), job 2 = RetinaNet target + MobileNet co-runner (both under gdb, to get backtraces).
#   fs-hang.sh A|B|C [iterations]
#   A: one daemon throughout (as sweep.sh)   B: fresh daemon before job 2   C: RetinaNet exits by itself
set -u
R=/home/nvidia/f_scarfato/runner; D=/mnt/nvme/f.scarfato
EXP=$1; N=${2:-10}
E=$D/scratch/fs-hang/${EXP}${TAG:-}; mkdir -p $E
export OUT=$E RUNNER=$R/runner DS=$D/dataset/imagenet-1k MOB=$D/models/mobilenetv2-12/engine/mobilenetv2-12.engine
INC=$D/models/inception_v3-onnx-float/engine/inception_v3.engine
RET=$D/models/retinanet-9/engine/retinanet-9.engine
STATIC_CHUNKS=8; MPS_PIPE=/tmp/mps-fshang; MPS_LOG=$E/mpslog-live; MPS_MODE=off
source $R/utility/lib/guards.sh; source $R/utility/lib/mps.sh
save_logs() { mkdir -p "$E/$1"; cp -f $MPS_LOG/*.log "$E/$1/" 2>/dev/null; }
stop_daemon() { save_logs "daemon-$(date +%H%M%S)"; mps_stop; }
trap 'stop_daemon' EXIT
check_clocks >/dev/null || exit 1; require_quiet_gpu || exit 1
P="--submit-threads 1 --dataset $DS --fixed 32 --allocation whole"
co() { env CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B $RUNNER $P --net $1:100:$2 --min-inferences 100 --min-duration $3 --csv /dev/shm/fshang-co > $4 2>&1 & CPID=$!; }

mps_ensure static || exit 1
for i in $(seq 1 $N); do
    # job 1: Inception target, RetinaNet co-runner
    mps_create_partitions "50:25" >/dev/null || exit 1
    if [ "$EXP" = C ]; then cdur=16; else cdur=600; fi
    co $RET retinanet $cdur $E/i$i-job1-co.log; cpid=$CPID; sleep 3
    env CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_A $RUNNER $P --net $INC:50:inception \
        --min-inferences 300 --min-duration 8 --csv /dev/shm/fshang-t > $E/i$i-job1-t.log 2>&1
    if [ "$EXP" = C ]; then wait $cpid; else t0=$(date +%s.%N); kill -TERM $cpid; wait $cpid 2>/dev/null; echo "$EXP iteration $i: co-runner exited $(echo "$(date +%s.%N) - $t0" | python3 -c "print(round(eval(input()),2))") s after SIGTERM"; fi
    mps_remove_partitions
    if [ "$EXP" = B ]; then stop_daemon; mps_ensure static || exit 1; fi
    # job 2: RetinaNet target, MobileNet co-runner, both under gdb
    mps_create_partitions "50:25" >/dev/null || exit 1
    env CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_B gdb -q -batch -ex run -ex "thread apply all bt 20" \
        --args $RUNNER $P --net $MOB:100:mobilenet --min-inferences 100 --min-duration 600 --csv /dev/shm/fshang-co2 \
        > $E/i$i-job2-co.gdb 2>&1 &
    gco=$!; sleep 3
    env CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_SM_PARTITION=$PART_A gdb -q -batch -ex run -ex "thread apply all bt 20" \
        --args $RUNNER $P --net $RET:50:retinanet --min-inferences 100 --min-duration 8 --csv /dev/shm/fshang-t2 \
        > $E/i$i-job2-t.gdb 2>&1 &
    gt=$!
    hung=0
    for s in $(seq 1 120); do kill -0 $gt 2>/dev/null || break; sleep 1; done
    if kill -0 $gt 2>/dev/null; then
        hung=1; save_logs "i$i-hang"
        # interrupt the two runners (children of our gdbs): gdb then prints every thread's backtrace
        for g in $gt $gco; do for c in $(pgrep -P $g); do kill -INT $c; done; done
        sleep 10; wait $gt 2>/dev/null
    fi
    for c in $(pgrep -P $gco); do kill -INT $c; done; sleep 3; kill -TERM $gco 2>/dev/null; wait $gco 2>/dev/null
    mps_remove_partitions
    echo "$EXP iteration $i: $([ $hung = 1 ] && echo HANG || echo ok)"
    if [ $hung = 1 ]; then stop_daemon; mps_ensure static || exit 1; fi
done
rm -f /dev/shm/fshang-*
