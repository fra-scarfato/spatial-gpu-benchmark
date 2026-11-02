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

# ---- Block 2 (Daniel, 2026-10-10): the temporal distribution of the stress at the same mean bandwidth.
#   stress-onoff  4 targets x {alone, ON/OFF 400 ms at D = 25/50/75/100 %, uniform 32 MB ~3 ms at the
#                 same GB/s for 25/50/75 %}, SAXPY on 8 SMs next to the target on 8 (50:50), 5 runs
step $R/utility/sweep.sh --exp stress-onoff --only "^stress-onoff_" --static-chunks 8 --runs 5
run python3 $R/utility/aggregate.py --out $B
run python3 $S/onoff.py $B/stress-onoff
log "${NAME^^}-END"
