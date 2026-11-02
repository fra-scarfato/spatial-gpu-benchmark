#!/usr/bin/env bash
# Board-side campaign queue. Copy to <scratch>/<name>.sh, fill the STEPS section, run with
#   nohup bash <name>.sh > <name>.log 2>&1 &
# Contract: every step is guarded and retried while the GPU is busy; any other failure logs FAILED and
# stops (never let a crash be read as a result); the last line is <NAME>-END.
R=/home/nvidia/f_scarfato/runner            # board 2: ~/f_scarfato/runner
B=/mnt/nvme/f.scarfato/bench                 # board 2: ~/f_scarfato/data/bench (export DATA=~/f_scarfato/data)
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

# ---- STEPS: the geometry test (the user, 2026-10-28): the frozen L2 counter model on 4+12, 6+10,
# 12+4, then 8+4 and 8+4+4, with power. The sudo profiles (soc-mps-sizes) must be converted first.

n=$(grep -c '^soc-mps-.*-\(25\|37\|63\|75\),' /mnt/nvme/f.scarfato/profiles/derived/dram.csv)
[ "$n" -eq 24 ] || fail "dram.csv has $n of the 24 soc-mps-sizes rows: run the sudo step and profiles.py first"

sweep() { env POWER=1 $R/utility/sweep.sh --exp "$1" --only "^$1_" --static-chunks 8 --runs 5; }
step sweep stress-sizes
step sweep stress-split-412
step sweep stress-split-610
step sweep stress-split-124
run  python3 $R/utility/analysis/geometry.py inputs
run  python3 $R/utility/analysis/geometry.py splits
# The 8+4 and 8+4+4 predictions are frozen here, from the individual profiles only, before any of
# those runs exists (triple-predict refuses otherwise).
run  python3 $R/utility/analysis/geometry.py triple-predict
step sweep stress-triple
run  python3 $R/utility/analysis/geometry.py triple-compare
run  python3 $R/utility/analysis/geometry.py power

log "${NAME^^}-END"
