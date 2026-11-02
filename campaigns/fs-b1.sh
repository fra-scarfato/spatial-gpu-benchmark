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

# ---- diagnostic: why board 1 and board 2 differ. Same session, 3 runs each, the reporter on the target's
# SMs (LAYOUT=target, WINDOW=512). own = this board's binaries; itf-b2 = the other board's itfproc;
# rep-b2 = the other board's gpu-latency. The two boards' builds differ only in the CUDA toolkit
# (13.2.1 V13.2.78 on board 1, 13.2.2 V13.2.86 on board 2); sources and clocks are identical.
X=$B/diag-xboard
log "co-tenants: $(uptime); $(ps -eo user,pcpu,comm --sort=-pcpu | sed -n 2,4p | tr '\n' ' ')"
D='^tb-D(000|010|100)$'
step env LAYOUT=target RUNS=3 OUTROOT=$X/own ONLY='^idle$' $R/utility/reporter.sh calibration
step env LAYOUT=target RUNS=3 OUTROOT=$X/own ONLY="$D" $R/utility/reporter.sh calibration-tb
step env LAYOUT=target RUNS=3 OUTROOT=$X/own ONLY='^resnet$' $R/utility/reporter.sh networks
step env LAYOUT=target RUNS=3 OUTROOT=$X/itf-b2 ITF=$S/itfproc.b2 ONLY="$D" $R/utility/reporter.sh calibration-tb
step env LAYOUT=target RUNS=3 OUTROOT=$X/rep-b2 REPORTER=$S/gpu-latency.b2 ONLY='^idle$' $R/utility/reporter.sh calibration
step env LAYOUT=target RUNS=3 OUTROOT=$X/rep-b2 REPORTER=$S/gpu-latency.b2 ONLY='^tb-D000$' $R/utility/reporter.sh calibration-tb
step env LAYOUT=target RUNS=3 OUTROOT=$X/rep-b2 REPORTER=$S/gpu-latency.b2 ONLY='^resnet$' $R/utility/reporter.sh networks
log "DIAG-DONE"
# ---- Burst-scrambled · Windowed · Symmetric (tbsw) on all four targets. Calibration and networks in one
# session, read by the windowed reporter on the target's SMs; then the four sensitivity curves.
for d in 0.30 1.00; do
    timeout -s TERM 10 $R/utility/probes/itfproc saxpy --period-us 1000 --duty $d --scramble-kb 2048 > $S/$NAME-smoke.log 2>&1
    log "smoke duty $d: $(grep -o "([0-9.]* GB/s)" $S/$NAME-smoke.log | tail -1) $(grep -o "achieved [0-9.]*" $S/$NAME-smoke.log | tail -1) $(grep -m1 -o "scramble:.*" $S/$NAME-smoke.log)"
    grep -q "scramble:" $S/$NAME-smoke.log && grep -q "achieved 0\.[1-9]\|achieved 1\." $S/$NAME-smoke.log || fail "smoke duty $d"
done
step env LAYOUT=target RUNS=5 OUTROOT=$B/stress-reporter-tbsw $R/utility/reporter.sh calibration-tbs
step env LAYOUT=target RUNS=5 OUTROOT=$B/stress-reporter-tbsw $R/utility/reporter.sh networks
step $R/utility/sweep.sh --exp stress-sensitivity-tbs --only "^stress-sensitivity-tbs_" --static-chunks 8 --runs 5
run python3 $R/utility/aggregate.py --out $B
run python3 $R/utility/stress.py predict tbsw
run python3 $R/utility/stress.py compare tbsw
run python3 $S/p99all.py
log "${NAME^^}-END"
