#!/usr/bin/env bash
# Board-side campaign queue. Copy to <scratch>/<name>.sh, fill the STEPS section, run with
#   nohup bash <name>.sh > <name>.log 2>&1 &
# Contract: every step is guarded and retried while the GPU is busy; any other failure logs FAILED and
# stops (never let a crash be read as a result); the last line is <NAME>-END.
R=/home/nvidia/f_scarfato/runner            # board 2: ~/f_scarfato/runner
B=/mnt/nvme/f.scarfato/bench                 # board 2: ~/f_scarfato/data/bench (export DATA=~/f_scarfato/data)
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

# ---- STEPS: the composition test (Daniel, 2026-10-22): the 8-SM ruler, then two 4-SM SAXPY ----

O=$B/stress-reporter-compose
step env LAYOUT=target RUNS=5 OUTROOT=$O $R/measure/reporter.sh calibration-smooth
step env LAYOUT=target RUNS=5 OUTROOT=$O $R/measure/reporter.sh compose-4sm
run  python3 $R/analysis/compose.py

log "${NAME^^}-END"
