#!/usr/bin/env bash
# The reporter runs again with the reporter on the target's partition (LAYOUT=target): idle and both
# calibrations, the four networks, the two L2 probes. Waits for a free GPU; every step is resumable.
R=/home/nvidia/f_scarfato/runner
source $R/utility/lib/guards.sh          # only for other_gpu_users; OUT is NOT exported
log() { echo "[$(date "+%F %T")] $*"; }
step() {
    until LAYOUT=target "$@"; do
        log "step aborted (busy GPU?): $*; retrying in 10 min"; sleep 600
        until other_gpu_users >/dev/null 2>&1; do sleep 300; done
    done
    log "step done: $*"
}
log "waiting for a free GPU"
until other_gpu_users >/dev/null 2>&1; do sleep 300; done
log "GPU free, starting"
step $R/utility/reporter.sh networks
step $R/utility/reporter.sh calibration-4sm
step $R/utility/reporter.sh calibration
step $R/utility/reporter.sh probes-4sm
log "REPORTER-TARGET-END"
