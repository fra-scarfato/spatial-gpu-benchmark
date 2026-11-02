#!/usr/bin/env bash
# Board-side campaign queue. Copy to <scratch>/<name>.sh, fill the STEPS section, run with
#   nohup bash <name>.sh > <name>.log 2>&1 &
# Contract: every step is guarded and retried while the GPU is busy; any other failure logs FAILED and
# stops (never let a crash be read as a result); the last line is <NAME>-END.
R=/home/nvidia/f_scarfato/runner            # board 2: ~/f_scarfato/runner
B=/mnt/nvme/f.scarfato/bench
S=$(dirname "$(readlink -f "$0")")
NAME=$(basename "$0" .sh)
source $R/measure/lib/guards.sh              # for other_gpu_users
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

# ---- Reboot test: is board 1's light-load floor (reporter alone 662 ns p99, 841 cycles) set at boot?
# The same configurations as the governor A/B of 2026-10-09 (diag-xboard/gov-nvhost_podgov), first boot
# after a reboot, LAYOUT=target, WINDOW=512, 3 runs each.
X=$B/diag-xboard/boot-$(date +%Y%m%d)
log "$(uptime); boot $(uptime -s); governor $(cat /sys/class/devfreq/17000000.gpu/governor)"
step env LAYOUT=target RUNS=3 OUTROOT=$X ONLY='^(idle|saxpy-L(000|010))$' $R/measure/reporter.sh calibration
step env LAYOUT=target RUNS=3 OUTROOT=$X ONLY='^tb-D(000|010)$' $R/measure/reporter.sh calibration-tb
step env LAYOUT=target RUNS=3 OUTROOT=$X ONLY='^(resnet|mobilenet)$' $R/measure/reporter.sh networks
run python3 $S/xb.py $B/diag-xboard/gov-nvhost_podgov $X
log "${NAME^^}-END"
