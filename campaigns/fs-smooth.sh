#!/usr/bin/env bash
# Board-side campaign queue. Copy to <scratch>/<name>.sh, fill the STEPS section, run with
#   nohup bash <name>.sh > <name>.log 2>&1 &
# Contract: every step is guarded and retried while the GPU is busy; any other failure logs FAILED and
# stops (never let a crash be read as a result); the last line is <NAME>-END.
R=/home/nvidia/f_scarfato/runner            # board 2: ~/f_scarfato/runner
B=/mnt/nvme/f.scarfato/bench
S=$(dirname "$(readlink -f "$0")")
NAME=$(basename "$0" .sh)
source $R/utility/lib/guards.sh              # for other_gpu_users
log()  { echo "[$(date '+%F %T')] $*"; }
fail() { log "FAILED: $*"; exit 1; }
step() {   # guarded experiment step: retry on busy GPU / unlocked clocks, at most 6 times
    local k=0
    until "$@"; do
        k=$((k+1)); [ $k -ge 6 ] && fail "step gave up: $*"
        log "step aborted (busy GPU or clocks?): $*; retrying in 10 min"; sleep 600
        until other_gpu_users >/dev/null 2>&1; do sleep 300; done
    done
    log "step done: $*"
}
run() { "$@" || fail "$*"; }                 # analysis step: must succeed

log "waiting for a free GPU"
until other_gpu_users >/dev/null 2>&1; do sleep 300; done

# ---- Block 2, the smooth stressor (2026-10-11): the time-based SAXPY at 100 us periods, burst 2, read by
# the reporter (calibration-onoff, the measured configurations are skipped) and beside the 4 targets
# (stress-onoff), then the smooth88 model, with spin as level 0.
O=/mnt/nvme/f.scarfato/bench/stress-reporter-onoff
step env LAYOUT=target WINDOW=512 FLOOR_REF=847 OUTROOT=$O ONLY='^smooth-D' $R/utility/reporter.sh calibration-onoff
step $R/utility/sweep.sh --exp stress-onoff --only "^stress-onoff_.*-smooth-D" --static-chunks 8 --runs 5
run python3 $R/utility/aggregate.py --out $B
if python3 $R/utility/stress.py predict smooth88 && python3 $R/utility/stress.py compare smooth88; then
    log "model smooth88: done"
else
    log "model smooth88: REFUSED or failed (see above)"
fi
log "${NAME^^}-END"
