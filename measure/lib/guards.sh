#!/bin/bash
# The preconditions, checked before every run: the clocks, and a quiet GPU and CPU.
# Sourced by sweep.sh and profile.sh. The functions read a few configuration variables, defaulted
# below when the caller does not set them.
#
# The board cannot be locked (no compute mode on Tegra, no MIG, one shared account), so
# quiet is measured rather than assumed: a known workload is timed and compared against a
# reference, and a run that lands outside the tolerance aborts instead of measuring.

# What the checks below read, defaulted here rather than assumed of the caller: the sweep sets these
# itself, and a script that sources this library without them died on "GPU_DEVFREQ: unbound variable"
# the moment set -u met check_clocks.
: "${GPU_DEVFREQ:=/sys/class/devfreq/17000000.gpu}"
: "${SENTINEL_REF_MS:=0.535}"
: "${SENTINEL_TOL:=0.15}"
: "${SENTINEL_INF:=500}"
# The GPU devfreq governor the campaign expects. It cannot move a locked clock: an A/B on board 1
# (2026-10-09, 36 reporter runs each) found performance and the Jetson default nvhost_podgov equal
# within noise. performance is kept because board 1 comes up with it and most campaigns ran under it.
: "${GPU_GOVERNOR:=performance}"

require_quiet_cpu() {
    local one
    # The 5-minute average, not the 1-minute: a single transient spike (a build, the
    # previous run's teardown) must not abort a campaign, while a co-tenant who is
    # actually working keeps the 5-minute average up for minutes.
    one=$(awk '{print $2}' /proc/loadavg)
    if awk -v l="$one" -v m="$CPU_LOAD_MAX" 'BEGIN{exit !(l > m)}'; then
        echo "[sweep] ABORT: 5-minute load average is $one (limit $CPU_LOAD_MAX)." >&2
        echo "        The submission-model arms measure host-side scheduling;" >&2
        echo "        something else on the CPU is a different experiment." >&2
        return 1
    fi
    return 0
}

check_clocks() {
    local ok=1 gmin gmax cmin cmax cpu
    if [ -r "$GPU_DEVFREQ/min_freq" ]; then
        gmin=$(cat "$GPU_DEVFREQ/min_freq")
        gmax=$(cat "$GPU_DEVFREQ/max_freq")
        printf '[sweep] GPU clocks: min=%s max=%s\n' "$gmin" "$gmax"
        # The governor is recorded on every check and compared with GPU_GOVERNOR (above); jetson_clocks
        # does not change it. A warning, not an abort: the A/B found no effect on the measurements.
        local gov; gov=$(cat "$GPU_DEVFREQ/governor" 2>/dev/null)
        printf '[sweep] GPU governor: %s\n' "${gov:-unreadable}"
        [ "$gov" = "$GPU_GOVERNOR" ] || echo "[sweep] WARNING: GPU governor is ${gov:-unreadable}, not the expected $GPU_GOVERNOR." >&2
        # GPU_FREQ (Hz): a block of a campaign meant for one locked frequency (measure/gpu-freq.sh).
        if [ -n "${GPU_FREQ:-}" ] && { [ "$gmin" != "$GPU_FREQ" ] || [ "$gmax" != "$GPU_FREQ" ]; }; then
            echo "[sweep] ABORT: the GPU clock is not locked at GPU_FREQ=$GPU_FREQ." >&2
            echo "        Fix with:  sudo measure/gpu-freq.sh $GPU_FREQ" >&2
            ok=0
        fi
        if [ "$gmin" != "$gmax" ]; then
            echo "[sweep] ABORT: the GPU clocks are NOT locked." >&2
            echo "        An idle governor would inflate every latency ~4x." >&2
            echo "        Fix with:  sudo jetson_clocks   (needed after every boot)" >&2
            ok=0
        fi
    else
        echo "[sweep] WARNING: cannot read $GPU_DEVFREQ — clocks unverified." >&2
    fi

    # The memory clock: every result here is a memory-contention result, and a fresh boot leaves the
    # EMC scaling (2133 instead of 3199 MHz was seen on 2026-10-10) until jetson_clocks pins it.
    local emin emax
    if [ -r /sys/class/devfreq/bwmgr/min_freq ]; then
        emin=$(cat /sys/class/devfreq/bwmgr/min_freq); emax=$(cat /sys/class/devfreq/bwmgr/max_freq)
        printf '[sweep] EMC clocks: min=%s max=%s\n' "$emin" "$emax"
        if [ "$emin" != "$emax" ]; then
            echo "[sweep] ABORT: the memory (EMC) clock is NOT locked.  Fix with: sudo jetson_clocks" >&2
            ok=0
        fi
    fi

    # Every core, not just cpu0: cpufreq is per-policy and a partly-locked board
    # is the sort of thing that produces a 5% mystery nobody can explain later.
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        [ -r "$cpu/cpufreq/scaling_min_freq" ] || continue
        cmin=$(cat "$cpu/cpufreq/scaling_min_freq")
        cmax=$(cat "$cpu/cpufreq/scaling_max_freq")
        if [ "$cmin" != "$cmax" ]; then
            echo "[sweep] ABORT: CPU clocks are NOT locked on $(basename "$cpu")" >&2
            echo "        (min=$cmin max=$cmax)  Fix with: sudo jetson_clocks" >&2
            ok=0
            break
        fi
    done
    [ "$ok" = 1 ]
}

# The board's login screen (gdm's Xorg, gnome-shell, mutter) holds the GPU devices but is idle. It
# runs as the gdm user, so a guard run as an ordinary user cannot read its file descriptors and never saw it:
# every campaign so far ran beside it. A guard run under sudo (profile.sh's counter sets) sees it,
# and ignores it here so that both see the same board.
is_desktop() {   # $1 = command line
    case "$1" in
        /usr/lib/xorg/Xorg\ *|/usr/bin/gnome-shell*|/usr/libexec/mutter-x11-frames*) return 0 ;;
    esac
    return 1
}

# Since 2026-11-11 the board has one account per person, and another user's /proc/<pid>/fd is
# unreadable, so the device nodes cannot tell whether that process holds the GPU. Its thread names
# can: they are public, and the CUDA driver starts threads named cuda-EvtHandlr and cuda0000... in
# every process that creates a context. Checked on this board: they are there from cudaFree(0) to
# exit, idle or busy. Sacrifice: another user's graphics-only GPU work (EGL, Vulkan) has no such
# thread and goes unseen; the only such user here is the desktop, which is ignored anyway.
cuda_threads() {   # $1 = /proc/<pid>
    grep -qs '^cuda' "$1"/task/*/comm
}

# other_gpu_users: 1, and one line per process (pid, user, command), if any process other than this
# campaign's own holds the GPU. A process whose open files are readable (ours, or any under sudo)
# is judged by its device nodes; any other process by its CUDA threads.
other_gpu_users() {
    local p pid f t cmd hit found=0
    for p in /proc/[0-9]*; do
        hit=""
        if [ -r "$p/fd" ]; then
            # Any of the device nodes counts; stop at the first one. A process can
            # exit mid-scan, which is normal, so a failed readlink just means skip.
            for f in "$p"/fd/*; do
                t=$(readlink "$f" 2>/dev/null) || continue
                case "$t" in
                    /dev/nvgpu/*|/dev/nvmap|/dev/dri/renderD*) hit=1; break ;;
                esac
            done
        else
            cuda_threads "$p" && hit=1
        fi
        [ -n "$hit" ] || continue

        pid=$(basename "$p")
        # cmdline is NUL-separated; turn the NULs into spaces for one readable
        # line. Empty means a kernel thread or a process already exiting.
        cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
        [ -n "$cmd" ] || continue
        # Ourselves, and our MPS daemon, which is attached but idle by construction. Another
        # user's MPS server runs that user's work, so it counts.
        case "$cmd" in
            *sweep.sh*|*nvidia-cuda-mps-control*|*nvidia-cuda-mps-server*) [ -O "$p" ] && continue ;;
        esac
        is_desktop "$cmd" && continue
        printf '          pid %-8s %-10s %s\n' "$pid" "$(stat -c %U "$p" 2>/dev/null)" "$cmd"
        found=1
    done
    [ "$found" = 0 ]
}

require_quiet_gpu() {
    local scratch="$OUT/_sentinel" got ok

    if ! other_gpu_users; then
        echo "[sweep] ABORT: another process has the GPU open (named above)." >&2
        echo "        Nothing was measured. Coordinate with the other user:" >&2
        echo "        a co-tenant benchmark inflates every latency several-fold." >&2
        echo "        (A compiler does not hold these devices, so if one is listed" >&2
        echo "        it really is attached to the GPU.)" >&2
        return 1
    fi

    rm -f "${scratch}_latencies.csv" "${scratch}_stats.csv"
    "$RUNNER" "$MOB" "$DS" --allocation whole --fixed 8 \
        --min-inferences "$SENTINEL_INF" --min-duration 0 \
        --csv "$scratch" >/dev/null 2>&1 || {
            echo "[sweep] ABORT: the sentinel run itself failed." >&2
            return 1
        }
    # Column 4 of the gpu_ms row is the median. Bash cannot do floating point,
    # so both the extraction and the comparison below are done by awk.
    got=$(awk -F, '$1=="gpu_ms"{print $4}' "${scratch}_stats.csv")
    if [ -z "$got" ]; then
        echo "[sweep] ABORT: the sentinel produced no result." >&2
        return 1
    fi
    # Tolerance test in awk for the same reason: bash arithmetic is integer-only.
    ok=$(awk -v g="$got" -v r="$SENTINEL_REF_MS" -v t="$SENTINEL_TOL" \
         'BEGIN{ d = (g>r ? g-r : r-g); print (d <= r*t) ? 1 : 0 }')
    printf '[sweep] sentinel: %.4f ms (reference %.4f ms +/- %d%%)\n' \
           "$got" "$SENTINEL_REF_MS" "$(awk -v t="$SENTINEL_TOL" 'BEGIN{printf "%d", t*100}')"
    if [ "$ok" != 1 ]; then
        echo "[sweep] ABORT: the GPU is BUSY or misconfigured." >&2
        echo "        sentinel ${got} ms vs reference ${SENTINEL_REF_MS} ms." >&2
        echo "        Another workload is sharing the board; coordinate with" >&2
        echo "        the other user and re-run. (Nothing was lost: the sweep" >&2
        echo "        is resumable, finished runs are skipped.)" >&2
        return 1
    fi
}

# ---- Provenance: which binaries ran, on which board state ---------------------------------------
# A rebuilt binary changes the measurement silently: on 2026-10-09 a rebuild of corunner made the
# Steady SAXPY 3.6 % slower and nothing noticed for a day. So every campaign script calls
# check_binaries once at start-up: it compares the binaries against probes/PINNED-<host>.sha256
# and refuses to run on a mismatch, and it writes a manifest (commit, binary hashes, governor, clocks)
# next to the results. Rebuilding on purpose means re-pinning (pin_binaries) and committing the pin
# file, so a change of instrument is always a visible, dated commit.
# Checklist: a new probe binary used by a campaign also goes into PINNED_BINARIES.
_GUARDS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PINNED_BINARIES="runner probes/corunner probes/gpu-latency/gpu-latency"
PIN_FILE="$_GUARDS_ROOT/probes/PINNED-$(hostname).sha256"

pin_binaries() {
    (cd "$_GUARDS_ROOT" && sha256sum $PINNED_BINARIES) > "$PIN_FILE" && cat "$PIN_FILE"
}

# check_binaries <dir> [script args...]: 1 (and a message) if a pinned binary differs; otherwise writes
# <dir>/manifest-<time>.txt and leaves its path in MANIFEST. ITF and REPORTER, when a caller
# overrides them with a binary built elsewhere, are recorded with their hashes but not pinned.
check_binaries() {
    local dir=$1; shift
    [ -r "$PIN_FILE" ] || { echo "[sweep] ABORT: no pin file $PIN_FILE (create it with pin_binaries)" >&2; return 1; }
    if ! (cd "$_GUARDS_ROOT" && sha256sum --quiet -c "$PIN_FILE") >&2; then
        echo "[sweep] ABORT: a binary differs from its pin ($PIN_FILE)." >&2
        echo "        If the rebuild was intended, re-pin: source measure/lib/guards.sh; pin_binaries; and commit." >&2
        return 1
    fi
    mkdir -p "$dir"
    MANIFEST="$dir/manifest-$(date +%Y%m%d-%H%M%S).txt"
    {
        echo "date      $(date '+%F %T')   host $(hostname)   boot $(uptime -s)"
        echo "commit    $(git -C "$_GUARDS_ROOT" rev-parse --short HEAD)$(git -C "$_GUARDS_ROOT" diff --quiet HEAD -- . ':!*.sha256' 2>/dev/null || echo ' (uncommitted changes)')"
        echo "command   $0 $*"
        echo "governor  $(cat "$GPU_DEVFREQ/governor" 2>/dev/null)   gpu $(cat "$GPU_DEVFREQ/min_freq")-$(cat "$GPU_DEVFREQ/max_freq")   emc $(cat /sys/class/devfreq/bwmgr/cur_freq 2>/dev/null)"
        echo "cuda      $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | grep -o "V[0-9][0-9.]*")   tensorrt $(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null)   l4t $(head -1 /etc/nv_tegra_release | grep -o 'R[0-9]*.*REVISION: [0-9.]*')"
        (cd "$_GUARDS_ROOT" && sha256sum $PINNED_BINARIES) | sed 's/^/binary    /'
        [ -n "${ITF:-}" ] && [ "$ITF" != "$_GUARDS_ROOT/probes/corunner" ] && echo "binary    $(sha256sum "$ITF")   (ITF override, not pinned)"
        [ -n "${REPORTER:-}" ] && [ "$REPORTER" != "$_GUARDS_ROOT/probes/gpu-latency/gpu-latency" ] && echo "binary    $(sha256sum "$REPORTER")   (REPORTER override, not pinned)"
    } > "$MANIFEST"
    echo "[sweep] manifest: $MANIFEST"
}

# floor_probe <tag>: 10 s of the reporter alone on the whole device, appended to MANIFEST, with the median
# cycles per access left in FLOOR_CYCLES (empty if the probe failed). The light-load memory latency
# drifts between sessions (771-847 cycles per access, 2026-10-05..10) while loaded readings and network
# latencies do not; this records which floor a campaign ran on. It needs the MPS daemon stopped (the
# callers stop it first) and never fails the campaign by itself.
floor_probe() {
    local f line; f=$(mktemp /dev/shm/fs-floor.XXXXXX)
    FLOOR_CYCLES=""
    if CUDA_MPS_PIPE_DIRECTORY=/nonexistent timeout 40 "$_GUARDS_ROOT/probes/gpu-latency/gpu-latency" \
           --sizes-kb 65536 --duration 10 --window 512 > "$f" 2>&1; then
        FLOOR_CYCLES=$(awk '$1 == 65536 && NF == 4 {print $3}' "$f" | sort -n | awk '{c[NR]=$1} END{if (NR) printf "%.1f", c[int((NR+1)/2)]}')
        line="floor     $(printf %-5s "$1") $FLOOR_CYCLES cycles median per access (reporter alone, whole device)"
    else
        line="floor     $1 probe failed: $(tail -1 "$f")"
    fi
    echo "$line" | tee -a "${MANIFEST:-/dev/null}"
    rm -f "$f"
}

# require_floor <tag>: when FLOOR_REF is set, 1 unless the floor is within FLOOR_TOL cycles of it. All the
# reporter readings one stress model is built from (calibration and co-runners) must share one floor
# (the user's rule, 2026-10-10): a campaign that finds the floor moved stops instead of mixing floors.
# Without FLOOR_REF the floor is only recorded.
require_floor() {
    floor_probe "$1"
    [ -z "${FLOOR_REF:-}" ] && return 0
    if [ -z "$FLOOR_CYCLES" ] || awk -v f="$FLOOR_CYCLES" -v r="$FLOOR_REF" -v t="${FLOOR_TOL:-8}" \
           'BEGIN{exit !(f < r - t || f > r + t)}'; then
        echo "[sweep] ABORT: the memory floor is ${FLOOR_CYCLES:-unknown} cycles, not $FLOOR_REF +/- ${FLOOR_TOL:-8}." >&2
        echo "        The readings of one model must share one floor; this campaign stops here." >&2
        return 1
    fi
}
