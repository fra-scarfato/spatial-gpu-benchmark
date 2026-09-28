#!/bin/bash
# The preconditions, checked before every run: the clocks, and a quiet GPU and CPU.
# Sourced by sweep.sh. The functions read the configuration variables sweep.sh defines.
#
# The board cannot be locked (no compute mode on Tegra, no MIG, one shared account), so
# quiet is measured rather than assumed: a known workload is timed and compared against a
# reference, and a run that lands outside the tolerance aborts instead of measuring.

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
        if [ "$gmin" != "$gmax" ]; then
            echo "[sweep] ABORT: the GPU clocks are NOT locked." >&2
            echo "        An idle governor would inflate every latency ~4x." >&2
            echo "        Fix with:  sudo jetson_clocks   (needed after every boot)" >&2
            ok=0
        fi
    else
        echo "[sweep] WARNING: cannot read $GPU_DEVFREQ — clocks unverified." >&2
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

other_gpu_users() {
    local p pid f t cmd hit found=0
    for p in /proc/[0-9]*; do
        # Any of the device nodes counts; stop at the first one. A process can
        # exit mid-scan, which is normal, so a failed readlink just means skip.
        hit=""
        for f in "$p"/fd/*; do
            t=$(readlink "$f" 2>/dev/null) || continue
            case "$t" in
                /dev/nvgpu/*|/dev/nvmap|/dev/dri/renderD*) hit=1; break ;;
            esac
        done
        [ -n "$hit" ] || continue

        pid=$(basename "$p")
        # cmdline is NUL-separated; turn the NULs into spaces for one readable
        # line. Empty means a kernel thread or a process already exiting.
        cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
        [ -n "$cmd" ] || continue
        # Ourselves, and the daemon, which is attached but idle by construction.
        case "$cmd" in
            *sweep.sh*|*nvidia-cuda-mps-control*|*nvidia-cuda-mps-server*) continue ;;
        esac
        printf '          pid %-8s %s\n' "$pid" "$cmd"
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
