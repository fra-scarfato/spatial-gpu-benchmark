#!/usr/bin/env bash
# The Burst stressor (time-based SAXPY, --period-us 1000 --duty D) read by the windowed probe on
# the target's partition (LAYOUT=target, geometry tbw): the stressor's calibration and the four
# networks read in one session, the four targets' sensitivity curves, then predict and compare on
# the twelve stress-validation pairs. Not in the thesis.
# As run on 2026-10-09 it first copied a scratch build (itfproc.tb) over probes/corunner; the
# time-based mode has been in the pinned probes/corunner since, so that step is gone, and the pin
# check would refuse a swapped binary anyway. The original is in commit bd0f7b6.
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
for d in 0.30 1.00; do
    timeout -s TERM 10 $R/probes/corunner saxpy --period-us 1000 --duty $d > $S/burst-stressor-windowed-probe-smoke.log 2>&1
    log "smoke duty $d: $(grep -o "([0-9.]* GB/s)" $S/burst-stressor-windowed-probe-smoke.log | tail -1) $(grep -o "achieved [0-9.]*" $S/burst-stressor-windowed-probe-smoke.log | tail -1)"
    grep -q "achieved 0\.[1-9]\|achieved 1\." $S/burst-stressor-windowed-probe-smoke.log || { log "SMOKE FAILED: stopping"; exit 1; }
done
step env RUNS=5 OUTROOT=$B/stress-reporter-tbw $R/measure/reporter.sh calibration-tb
step env RUNS=5 OUTROOT=$B/stress-reporter-tbw $R/measure/reporter.sh networks
step $R/measure/sweep.sh --exp stress-sensitivity-tb --only "^stress-sensitivity-tb_" --static-chunks 8 --runs 5
python3 $R/analysis/aggregate.py --out $B > /dev/null || { log "AGGREGATE FAILED"; exit 1; }
python3 $R/analysis/stress.py predict tbw && python3 $R/analysis/stress.py compare tbw
log "BURST-WINDOWED-END"
