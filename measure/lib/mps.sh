#!/bin/bash
# The MPS daemon and its static partitions, for the two-process arms.
# Sourced by sweep.sh, which owns the EXIT trap that calls cleanup.
#
# The pipe directory is the isolation boundary: a client joins a server only if it was
# told about the same directory, so a private one keeps this daemon invisible to anyone
# else on the shared board. Static mode is chosen at daemon start-up (-S) and cannot be
# turned on later, which is why switching modes restarts the daemon rather than setting a
# variable.

mps_ctl() {  # run one control command, print its output
    # nvidia-cuda-mps-control takes commands on stdin, one per line.
    printf '%s\n' "$1" | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        nvidia-cuda-mps-control 2>&1
}

mps_alive() {
    local out
    out=$(mps_ctl "get_default_active_thread_percentage" 2>/dev/null || true)
    [ -n "$out" ] && ! printf '%s' "$out" | grep -qi "cannot find"
}

mps_stop() {
    if mps_alive; then
        mps_ctl "quit" >/dev/null 2>&1 || true
        sleep 1
        if mps_alive; then
            echo "[sweep] WARNING: the MPS daemon did not shut down." >&2
            echo "        It uses the private pipe dir $MPS_PIPE, so nobody" >&2
            echo "        else is affected. Stop it manually with:" >&2
            echo "          printf 'quit\\n' | CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE nvidia-cuda-mps-control" >&2
        fi
    fi
    # Recreate the pipe and log dirs as well, so the next start begins from a
    # clean slate rather than inheriting a stale socket.
    rm -rf "$MPS_PIPE" "$MPS_LOG"
    MPS_MODE=off
}

mps_start() {  # $1 = on | static
    local mode="$1" flags="" out
    mkdir -p "$MPS_PIPE" "$MPS_LOG"
    # -S turns on static partitioning mode, and it can only be chosen when the
    # daemon starts; there is no way to enable it later. That is the reason
    # switching between MPS phases restarts the daemon instead of just setting
    # an environment variable.
    [ "$mode" = static ] && flags="-S"

    # setsid + stdin from /dev/null: the daemon must outlive this script's
    # session, otherwise it dies when the shell goes away (which is exactly
    # what happened the first time this was tried by hand).
    CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" \
        setsid nvidia-cuda-mps-control -d $flags </dev/null >/dev/null 2>&1

    sleep 2
    if ! mps_alive; then
        echo "[sweep] ABORT: the MPS daemon did not start (mode=$mode, flags='$flags')." >&2
        return 1
    fi
    out=$(mps_ctl "get_default_active_thread_percentage")
    echo "[sweep] MPS daemon up (mode=$mode, pipe=$MPS_PIPE, default ATP=$out)"

    if [ "$mode" = static ]; then
        # lspart answers "Static partitioning mode is not enabled" unless -S
        # took effect, which is the cheapest proof that the mode is really on
        # rather than merely accepted as a flag.
        out=$(mps_ctl "lspart")
        if printf '%s' "$out" | grep -qi "not enabled"; then
            echo "[sweep] ABORT: static partitioning mode is NOT enabled." >&2
            return 1
        fi
        echo "[sweep] static partitioning enabled; current partitions:"
        printf '          %s\n' "$out"
    fi
}

mps_ensure() {  # $1 = off | on | static
    [ "$1" = "$MPS_MODE" ] && return 0
    mps_stop
    if [ "$1" != off ]; then
        mps_start "$1" || return 1
    fi
    MPS_MODE="$1"
}

mps_create_partitions() {  # $1 = "50" | "50:25" | "50:25:25": the clients' shares, in order
    local spec="${1:-100}" uuid dev total pct chunks left out
    PART_LIST=()
    PART_A=""
    PART_B=""

    if [ "$STATIC_CHUNKS" -le 0 ]; then
        echo "[sweep] ABORT: the static jobs need --static-chunks <total>." >&2
        echo "        Start the daemon with -S and read 'lspart' for the" >&2
        echo "        device total. A chunk is 2 SMs on an integrated GPU," >&2
        echo "        so this 16-SM device should report 8." >&2
        return 1
    fi
    uuid=$(nvidia-smi -L 2>/dev/null | sed -n 's/.*UUID: *\(.*\))/\1/p' | head -1)
    [ -n "$uuid" ] || { echo "[sweep] ABORT: cannot read the device UUID." >&2; return 1; }
    dev="GPU-$uuid"

    # One percentage means two partitions covering the device, as before; a colon list means one
    # partition per share, and whatever is left over stays unassigned, which is what the 4-SM
    # co-runner arms need.
    case "$spec" in
        *:*) set -- $(printf '%s' "$spec" | tr ':' ' ') ;;
        *)   set -- "$spec" "$((100 - spec))" ;;
    esac

    total="$STATIC_CHUNKS"
    left="$total"
    for pct in "$@"; do
        chunks=$(( (pct * total + 50) / 100 ))
        if [ "$chunks" -lt 1 ] || [ "$chunks" -gt "$left" ]; then
            echo "[sweep] ABORT: $total chunks cannot express the split '$spec'." >&2
            return 1
        fi
        out=$(mps_ctl "sm_partition add $dev $chunks" | sed -n 's/^Partition \(.*\) created$/\1/p')
        if [ -z "$out" ]; then
            echo "[sweep] ABORT: could not read the partition id from sm_partition add." >&2
            echo "        Whatever it printed is above; the expected line is" >&2
            echo "        'Partition GPU-<uuid>/<id> created'." >&2
            return 1
        fi
        PART_LIST+=("$out")
        left=$(( left - chunks ))
        printf '[sweep] partition %s: %s%% -> %s chunks -> %s\n' \
               "${#PART_LIST[@]}" "$pct" "$chunks" "$out"
    done
    PART_A="${PART_LIST[0]}"
    [ "${#PART_LIST[@]}" -ge 2 ] && PART_B="${PART_LIST[1]}"
    out=$(mps_ctl "lspart")
    printf '%s\n' "$out" | sed 's/^/[sweep]   lspart: /'
    [ "$left" -gt 0 ] && printf '[sweep] %s chunks (of %s) left unassigned\n' "$left" "$total"
    return 0
}

mps_remove_partitions() {
    [ "$MPS_MODE" = static ] || return 0
    [ "${#PART_LIST[@]}" -gt 0 ] || return 0
    # A partition whose id starts with "/" cannot be removed: sm_partition rm answers "Failed to
    # remove partition" whatever form the id takes (CUDA 13.2 MPS, board 1, 2026-11-09). The id is the
    # base64 of the partition's chunk mask, so this hits a partition holding chunks 2-7, such as the
    # 14 SMs of 2+14 (mask 0xfe). Left in place, it would stop the next job from creating its own
    # partitions, so any failed removal stops the daemon and the next job starts a fresh one. The
    # chunk placement stays MPS's own; only the daemon is new.
    local part failed=0
    for part in "${PART_LIST[@]}"; do
        mps_ctl "sm_partition rm $part" | grep -q " removed$" || failed=1
    done
    PART_LIST=(); PART_A=""; PART_B=""
    if [ "$failed" = 1 ]; then
        echo "[sweep] a partition could not be removed: restarting the MPS daemon"
        mps_stop
    fi
}

cleanup() { mps_stop; }
