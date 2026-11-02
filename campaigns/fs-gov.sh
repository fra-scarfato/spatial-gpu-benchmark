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

# ---- GPU governor A/B. The same reporter set (LAYOUT=target, WINDOW=512, 3 runs each) under the
# current governor; run once under performance and once under the default nvhost_podgov, back to back.
# GOV names the output set; the governor actually in force is checked against it.
GOV=${GOV:?set GOV to the governor this run is meant for}
g=$(cat /sys/class/devfreq/17000000.gpu/governor); [ "$g" = "$GOV" ] || fail "governor is $g, expected $GOV"
X=$B/diag-xboard/gov-$GOV
log "governor $g, GPU min/max $(cat /sys/class/devfreq/17000000.gpu/min_freq)/$(cat /sys/class/devfreq/17000000.gpu/max_freq); $(uptime)"
step env LAYOUT=target RUNS=3 OUTROOT=$X ONLY='^(idle|saxpy-L(000|010|050|100))$' $R/utility/reporter.sh calibration
step env LAYOUT=target RUNS=3 OUTROOT=$X ONLY='^tb-D(000|010|100)$' $R/utility/reporter.sh calibration-tb
step env LAYOUT=target RUNS=3 OUTROOT=$X $R/utility/reporter.sh networks
g=$(cat /sys/class/devfreq/17000000.gpu/governor); [ "$g" = "$GOV" ] || fail "governor changed to $g during the run"
log "${NAME^^}-${GOV}-END"
