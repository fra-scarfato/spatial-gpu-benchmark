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

# ---- The L2 chapter, blocks 1a-1c (Daniel, 2026-10-10). Target slowdowns only (no reporter readings),
# so the floor is recorded in each sweep's manifest, not required.
#   kernels       each network alone, CUDA trace: kernels per inference (no root)
#   l2-presence   4 targets x {alone, spin K = 27000/1000/100/10 us, K = 100 us at P = 200/1000 us}
#   l2-gemm-sets  MobileNet x {alone, 288-cubed SGEMM over N = 1,2,3,4,5,8 sets, one launch per 115 us}
#   l2-read       MobileNet x {alone, spin K = 100 us, l2read K = 100 us at L = 0, 0.25, 0.5, 1}
step env WAIT_MIN=30 $R/utility/profile.sh kernels
step $R/utility/sweep.sh --exp l2-presence --only "^l2-presence_" --static-chunks 8 --runs 5
step $R/utility/sweep.sh --exp l2-gemm-sets --only "^l2-gemm-sets_" --static-chunks 8 --runs 5
step $R/utility/sweep.sh --exp l2-read --only "^l2-read_" --static-chunks 8 --runs 5
run python3 $R/utility/aggregate.py --out $B
log "${NAME^^}-END"
