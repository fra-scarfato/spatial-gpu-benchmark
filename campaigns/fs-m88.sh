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

# ---- Block 2, the two models in Daniel's 8 + 8 configuration (2026-10-11): the networks on the co-runner's
# 8 SMs next to the reporter, into the same directory as the calibration-onoff run, then the 12 pairs at
# 8 + 8 as ground truth, then the onoff88 and uniform88 models.
O=/mnt/nvme/f.scarfato/bench/stress-reporter-onoff
step env LAYOUT=target WINDOW=512 FLOOR_REF=847 OUTROOT=$O $R/utility/reporter.sh networks-8sm
step $R/utility/sweep.sh --exp stress-validation-8 --only "^stress-validation-8_" --static-chunks 8 --runs 5
run python3 $R/utility/aggregate.py --out $B
for g in onoff88 uniform88; do
    if python3 $R/utility/stress.py predict $g && python3 $R/utility/stress.py compare $g; then
        log "model $g: done"
    else
        log "model $g: REFUSED or failed (see above)"
    fi
done
log "${NAME^^}-END"
