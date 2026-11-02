#!/usr/bin/env bash
# Option A (stressor on 4 SMs) + the L2 diagnostic. Waits for a free GPU; every step is resumable,
# so a guard abort (someone else on the GPU) just means wait and run the step again.
R=/home/nvidia/f_scarfato/runner
# OUT is only for the guard, and must not be exported: sweep.sh and ncu.sh take OUT as their output root
OUT=/dev/shm/fs-wait; export MOB=/mnt/nvme/f.scarfato/models/mobilenetv2-12/engine/mobilenetv2-12.engine
mkdir -p $OUT; source $R/utility/lib/guards.sh
log() { echo "[$(date '+%F %T')] $*"; }
step() {
    until "$@"; do
        log "step aborted (busy GPU?): $*; retrying in 10 min"; sleep 600
        until other_gpu_users >/dev/null 2>&1; do sleep 300; done
    done
    log "step done: $*"
}
log "waiting for a free GPU"
until other_gpu_users >/dev/null 2>&1; do sleep 300; done
log "GPU free, starting"
step $R/utility/sweep.sh --exp stress-sensitivity-4sm --static-chunks 8 --runs 5
log "CAMPAIGN-4SM-END"
