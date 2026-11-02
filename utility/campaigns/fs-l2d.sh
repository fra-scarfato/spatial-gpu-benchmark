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

# ---- The L2 chapter, block 1d (Daniel, 2026-10-10): same DRAM traffic, different L2 pressure.
#   l2-vs-dram sweep     MobileNet x {alone, spin, compute_gemm, SAXPY L0.243 (22.3 GB/s), GEMM N=8, SAXPY L0.087 (8.0 GB/s)}
#   l2-vs-dram reporter  the same 5 co-runners, symmetric layout, windowed, floor required at 847 +- 8
step $R/utility/sweep.sh --exp l2-vs-dram --only "^l2-vs-dram_" --static-chunks 8 --runs 5
step env LAYOUT=target WINDOW=512 FLOOR_REF=847 $R/utility/reporter.sh l2-vs-dram
run python3 $R/utility/aggregate.py --out $B
log "${NAME^^}-END"
