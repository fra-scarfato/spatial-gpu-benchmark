#!/usr/bin/env bash
# Queued behind the 4-SM campaign: ncu.sh nets once the campaign has ended, nobody else has the GPU
# open and the MPS daemon is down (the nets mode uses green contexts). Needs root, for ncu.
R=/home/nvidia/f_scarfato/runner
L=/mnt/nvme/f.scarfato/scratch/fs-campaign-4sm.log
# OUT is only for the guard, and must not be exported: sweep.sh and ncu.sh take OUT as their output root
OUT=/dev/shm/fs-wait; export MOB=/mnt/nvme/f.scarfato/models/mobilenetv2-12/engine/mobilenetv2-12.engine
mkdir -p $OUT; source $R/utility/lib/guards.sh
log() { echo "[$(date "+%F %T")] $*"; }
log "waiting for CAMPAIGN-4SM-END"
until grep -q CAMPAIGN-4SM-END "$L"; do sleep 60; done
log "campaign ended; waiting for a free GPU and no MPS daemon"
until other_gpu_users >/dev/null 2>&1 && ! pgrep -f nvidia-cuda-mps >/dev/null; do sleep 60; done
log "running ncu.sh nets"
cd $R && ./utility/ncu.sh nets
chown -R nvidia: /mnt/nvme/f.scarfato/profiles/ncu
log "NCU-NETS-END"
