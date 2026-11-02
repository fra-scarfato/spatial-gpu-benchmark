#!/usr/bin/env bash
# After the target-layout reporter runs: redo saxpy4-L004 run 2 (chunk layout, with the binary the
# rest of that calibration used). Ends with QUEUE-END, which the root queue fs-ncu-queue.sh waits for.
# (The stressor variants - new itfproc, reporter.sh patterns, stress-pattern - were taken out.)
R=/home/nvidia/f_scarfato/runner
S=/mnt/nvme/f.scarfato/scratch
source $R/measure/lib/guards.sh          # only for other_gpu_users; OUT is NOT exported
log() { echo "[$(date "+%F %T")] $*"; }
free() { until other_gpu_users >/dev/null 2>&1; do sleep 300; done; }
retry() {
    until "$@"; do log "step aborted (busy GPU?): $*; retrying in 10 min"; sleep 600; free; done
    log "step done: $*"
}
log "waiting for REPORTER-TARGET-END"
until grep -q REPORTER-TARGET-END $S/fs-reporter-target.log; do sleep 120; done
retry env ONLY="^saxpy4-L004$" $R/measure/reporter.sh calibration-4sm
log "QUEUE-END"
