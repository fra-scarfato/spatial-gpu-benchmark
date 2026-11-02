#!/usr/bin/env bash
# fs-state.sh - what flips board 1's light-load memory latency between its two states?
#
# The reporter alone reads ~770 cycles per access right after a boot and ~805-840 later; loaded
# readings and network latencies do not move (2026-10-10). This sequence starts from a fresh boot and
# probes the state between single stimuli, so the trigger, and whether the state reverts by itself,
# can be read off one log.
#
# A probe: wait for an otherwise idle GPU, log the GPU/SoC rail power (tegrastats, no root), then run
# the reporter alone for 10 s on the whole device (no MPS, no partition: a one-thread pointer chase
# does not care) and print the median cycles per access, the median and p99 ns. low < 785 cycles.
# A stimulus: itfproc on the whole device for 10 s, then stopped, so a probe sees its after-effect
# and never overlaps it. Two pauses wait for the user's read-only root dump (fs-rootdump.sh).
R=/home/nvidia/f_scarfato/runner
S=/mnt/nvme/f.scarfato/scratch
D=$S/state-$(date +%Y%m%d-%H%M)
mkdir -p $D
source $R/utility/lib/guards.sh
log() { echo "[$(date '+%F %T')] $*"; }
n=0
probe() {
    until other_gpu_users >/dev/null 2>&1; do log "GPU busy, waiting"; sleep 30; done
    n=$((n + 1))
    local f=$D/p$(printf %02d $n)-$1.txt
    local vdd=$(timeout 2 tegrastats --interval 500 | head -1 | grep -o "VDD_GPU_SOC [0-9]*mW" | cut -d" " -f2)
    $R/utility/probes/gpu-latency/gpu-latency --sizes-kb 65536 --duration 10 --window 512 > $f 2>&1
    python3 - $f $1 $n "$vdd" <<'PY'
import sys, statistics as st
f, tag, n, vdd = sys.argv[1:]
r = [(float(x[2]), float(x[3])) for x in (l.split() for l in open(f)) if len(x) == 4 and x[0] == "65536"]
c = st.median(x[0] for x in r); ns = sorted(x[1] for x in r)
print("PROBE %2s %-12s cycles %6.1f  median %6.1f ns  p99 %6.1f ns  %4d windows  GPU_SOC %s  -> %s" % (
      n, tag, c, st.median(ns), ns[int(0.99 * (len(ns) - 1))], len(r), vdd, "LOW" if c < 785 else "HIGH"))
PY
}
stim() {   # stim <tag> <itfproc args...>
    local tag=$1; shift
    until other_gpu_users >/dev/null 2>&1; do log "GPU busy, waiting"; sleep 30; done
    timeout -s TERM 10 $R/utility/probes/itfproc "$@" > $D/stim-$tag.log 2>&1
    log "STIM $tag: itfproc $* for 10 s, $(grep -o "achieved [0-9.]*" $D/stim-$tag.log | tail -1) $(grep -o "([0-9.]* GB/s)" $D/stim-$tag.log | tail -1)"
}
rootgate() {
    log "ROOT-DUMP $1: please run   sudo bash $S/fs-rootdump.sh $D $1   (waiting up to 15 min)"
    for i in $(seq 90); do [ -f $D/rootdump-$1.txt ] && { log "root dump $1 received"; return; }; sleep 10; done
    log "root dump $1 skipped (timeout)"
}

log "boot $(uptime -s), up $(uptime -p); governor $(cat /sys/class/devfreq/17000000.gpu/governor); output $D"
log "1. baseline, a minute apart";            probe base; sleep 60; probe base; sleep 60; probe base
rootgate first
log "2. idle 5 min: does it flip by itself?"; sleep 300; probe idle5; probe idle5
log "3. resident kernel, no traffic";         stim level0 saxpy --level 0;    probe after-l0;   probe after-l0
log "4. light traffic";                       stim level01 saxpy --level 0.1; probe after-l01;  probe after-l01
log "5. full traffic";                        stim level1 saxpy --level 1.0;  probe after-l1;   probe after-l1
rootgate second
log "6. idle 15 min: does it revert?"
for k in 1 2 3 4 5; do sleep 180; probe revert$k; done
log "FS-STATE-END"
