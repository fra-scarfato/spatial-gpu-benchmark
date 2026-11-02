#!/usr/bin/env bash
# Board 1, symmetric (LAYOUT=target, stressor on 8 SMs beside an 8-SM target): the tbw model on all four
# targets. Time-based SAXPY calibration and the networks, read by the windowed reporter in one session,
# then the four sensitivity curves, then predict/compare on the 12 stress-validation pairs.
R=/home/nvidia/f_scarfato/runner
S=/mnt/nvme/f.scarfato/scratch
B=/mnt/nvme/f.scarfato/bench
source $R/measure/lib/guards.sh          # only for other_gpu_users; OUT is NOT exported
log() { echo "[$(date "+%F %T")] $*"; }
step() {
    until LAYOUT=target "$@"; do
        log "step aborted (busy GPU or clocks?): $*; retrying in 10 min"; sleep 600
        until other_gpu_users >/dev/null 2>&1; do sleep 300; done
    done
    log "step done: $*"
}
log "waiting for a free GPU"
until other_gpu_users >/dev/null 2>&1; do sleep 300; done
cp $S/itfproc.tb $R/probes/itfproc.tmp && mv $R/probes/itfproc.tmp $R/probes/itfproc
log "time-based itfproc installed (the previous binary is scratch/itfproc.before-tb)"
for d in 0.30 1.00; do
    timeout -s TERM 10 $R/probes/itfproc saxpy --period-us 1000 --duty $d > $S/fs-tbw1-smoke.log 2>&1
    log "smoke duty $d: $(grep -o "([0-9.]* GB/s)" $S/fs-tbw1-smoke.log | tail -1) $(grep -o "achieved [0-9.]*" $S/fs-tbw1-smoke.log | tail -1)"
    grep -q "achieved 0\.[1-9]\|achieved 1\." $S/fs-tbw1-smoke.log || { log "SMOKE FAILED: stopping"; exit 1; }
done
step env RUNS=5 OUTROOT=$B/stress-reporter-tbw $R/measure/reporter.sh calibration-tb
step env RUNS=5 OUTROOT=$B/stress-reporter-tbw $R/measure/reporter.sh networks
step $R/measure/sweep.sh --exp stress-sensitivity-tb --only "^stress-sensitivity-tb_" --static-chunks 8 --runs 5
python3 $R/analysis/aggregate.py --out $B > /dev/null || { log "AGGREGATE FAILED"; exit 1; }
python3 $R/analysis/stress.py predict tbw && python3 $R/analysis/stress.py compare tbw && python3 $S/p99all.py
log "TBW1-END"
