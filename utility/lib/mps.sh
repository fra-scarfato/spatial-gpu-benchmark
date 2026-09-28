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

mps_create_partitions() {  # $1 = the target's percentage, or "target:participant"
    # Two statements, not one: word expansion happens before the assignments of a single
    # local, so mp="${spec%%:*}" in the same declaration would expand the CALLER spec - the
    # whole job spec - and mp became "pair".
    local spec="${1:-100}" pp="" uuid dev total ca cb out
    local mp="${spec%%:*}"
    # A ratio of the form 50:25 asks for three partitions: 8 SMs to the target, 4 to the
    # co-runner, and the remaining 4 left unassigned. That is the shape the green-context arm
    # of the same measurement has, where a 50 % and a 25 % reservation also leave a quarter
    # of the device idle. A single percentage keeps the two partitions covering the device.
    [ "$spec" != "$mp" ] && pp="${spec##*:}"

    if [ "$STATIC_CHUNKS" -le 0 ]; then
        echo "[sweep] ABORT: the static jobs need --static-chunks <total>." >&2
        echo "        Start the daemon with -S and read 'lspart' for the" >&2
        echo "        device total. A chunk is 2 SMs on an integrated GPU," >&2
        echo "        so this 16-SM device should report 8." >&2
        return 1
    fi
    # nvidia-smi -L prints "GPU 0: Orin (nvgpu) (UUID: ef9ede97-...)", so lift
    # the UUID out of the parentheses. A client needs the full uuid.
    uuid=$(nvidia-smi -L 2>/dev/null | sed -n 's/.*UUID: *\(.*\))/\1/p' | head -1)
    [ -n "$uuid" ] || { echo "[sweep] ABORT: cannot read the device UUID." >&2; return 1; }

    # Split the device's chunks as the percentage asks. Bash arithmetic is
    # integer only, so the rounding is done by adding half the divisor before
    # the division; the remainder goes to B, which keeps the two partitions
    # covering the whole device (the protocol requires full capacity, nothing
    # may be left idle).
    total="$STATIC_CHUNKS"
    ca=$(( (mp * total + 50) / 100 ))
    if [ -n "$pp" ]; then
        cb=$(( (pp * total + 50) / 100 ))
    else
        cb=$(( total - ca ))
    fi
    if [ "$ca" -lt 1 ] || [ "$cb" -lt 1 ] || [ "$((ca + cb))" -gt "$total" ]; then
        echo "[sweep] ABORT: $total chunks cannot express the split '$spec'." >&2
        return 1
    fi

    # sm_partition wants the identifier WITH the GPU- prefix. The bare uuid
    # that nvidia-smi prints is rejected as "Invalid device identifier format",
    # which is easy to get wrong because the CLI accepts either form and only
    # tells you so at the point of use.
    dev="GPU-$uuid"

    printf '[sweep] sm_partition add %s %s (mobilenet) + %s (resnet)\n' "$dev" "$ca" "$cb"

    # The command PRINTS the identifier the client needs:
    #     Partition GPU-<uuid>/<id> created
    # so it is taken from there. That is exact, and it avoids parsing lspart --
    # whose columns shift, because it leaves the "free" cells blank and awk
    # then skips the empty fields, sliding used-chunks and used-SM left.
    #
    # The ORDER of these two calls is what identifies the partitions, so A is
    # always mobilenet. Nothing has to be inferred from sizes, which matters
    # because at 50/50 the two are the same size anyway.
    PART_A=$(mps_ctl "sm_partition add $dev $ca" | sed -n 's/^Partition \(.*\) created$/\1/p')
    PART_B=$(mps_ctl "sm_partition add $dev $cb" | sed -n 's/^Partition \(.*\) created$/\1/p')

    if [ -z "$PART_A" ] || [ -z "$PART_B" ]; then
        echo "[sweep] ABORT: could not read the partition ids from sm_partition add." >&2
        echo "        Whatever it printed is above; the expected line is" >&2
        echo "        'Partition GPU-<uuid>/<id> created'." >&2
        return 1
    fi

    # Printed for the record: the chunk and SM counts, and whether clients are
    # attached. Nothing below depends on parsing it.
    out=$(mps_ctl "lspart")
    printf '%s\n' "$out" | sed 's/^/[sweep]   lspart: /'

    printf '[sweep] A mobilenet %s%%: %s chunks -> %s\n' "$mp" "$ca" "$PART_A"
    printf '[sweep] B resnet    %s%%: %s chunks -> %s\n' "$((100-mp))" "$cb" "$PART_B"
}

mps_remove_partitions() {
    [ "$MPS_MODE" = static ] || return 0
    [ -n "$PART_A" ] || return 0
    # Removal fails while clients are still attached ("Partition %s in use"),
    # but the daemon is torn down immediately after, so this is best effort.
    mps_ctl "sm_partition rm $PART_A" >/dev/null 2>&1 || true
    mps_ctl "sm_partition rm $PART_B" >/dev/null 2>&1 || true
    PART_A=""; PART_B=""
}

cleanup() { mps_stop; }
