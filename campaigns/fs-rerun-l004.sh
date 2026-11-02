#!/usr/bin/env bash
# After the target-layout reporter runs: redo run 2 of saxpy4-L004 (chunk layout), which a co-tenant
# disturbed on 2026-10-07 (52k windows, mean 1136 ns); the bad files are in stress-reporter/rejected/.
R=/home/nvidia/f_scarfato/runner
source $R/measure/lib/guards.sh          # only for other_gpu_users; OUT is NOT exported
log() { echo "[$(date "+%F %T")] $*"; }
until grep -q REPORTER-TARGET-END /mnt/nvme/f.scarfato/scratch/fs-reporter-target.log; do sleep 120; done
until ONLY="^saxpy4-L004$" $R/measure/reporter.sh calibration-4sm; do
    log "aborted; retrying in 10 min"; sleep 600
    until other_gpu_users >/dev/null 2>&1; do sleep 300; done
done
log "RERUN-L004-END"
