#!/usr/bin/env bash
# Board-side campaign queue. Copy to <scratch>/<name>.sh, fill the STEPS section, run with
#   nohup bash <name>.sh > <name>.log 2>&1 &
# Contract: every step is guarded and retried while the GPU is busy; any other failure logs FAILED and
# stops (never let a crash be read as a result); the last line is <NAME>-END.
R=/home/nvidia/f_scarfato/runner            # board 2: ~/f_scarfato/runner
B=/mnt/nvme/f.scarfato/bench
S=$(dirname "$(readlink -f "$0")")
NAME=$(basename "$0" .sh)
source $R/measure/lib/guards.sh              # for other_gpu_users
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

# ---- Block 2, step C (2026-10-11): the ON/OFF and uniform models on the 12 pairs. The reporter (p99 of
# its 0.3 ms windows) next to idle and block 2s stressors, then next to the 4 networks, one session,
# one directory; the sensitivity is stress-onoff (exp23, same day).
O=/mnt/nvme/f.scarfato/bench/stress-reporter-onoff
step env LAYOUT=target WINDOW=512 FLOOR_REF=847 OUTROOT=$O $R/measure/reporter.sh calibration-onoff
step env LAYOUT=target WINDOW=512 FLOOR_REF=847 OUTROOT=$O $R/measure/reporter.sh networks
for g in onoff uniform; do
    # a flat calibration is refused by stress.py: that is a result, not a crash of the queue
    if python3 $R/analysis/stress.py predict $g && python3 $R/analysis/stress.py compare $g; then
        log "model $g: done"
    else
        log "model $g: REFUSED or failed (see above)"
    fi
done
log "${NAME^^}-END"
