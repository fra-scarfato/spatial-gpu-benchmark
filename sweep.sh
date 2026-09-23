#!/usr/bin/env bash
#
# sweep.sh — run the measurement protocol and write one CSV per run.
#
# Five independent runs per configuration, each in a FRESH process (or pair of
# processes). The fresh process is part of the protocol, not an implementation
# detail: it resets the CUDA context, the stream pool, the green contexts and
# any MPS client state, so run k+1 cannot inherit run k's state.
#
#   Experiment 1  one network, isolated      whole | gc100 | gc75 | gc50 | gc25
#   Experiment 2  two networks, one process  baseline | 50/50 | 75/25 | 25/75
#   Experiment 3  two processes, MPS         mps_single | mps_off | mps_default
#                                            | mps_atp_* | mps_static_*
#
# Experiments 1 and 2 are IN-PROCESS: one runner invocation, one or two
# networks. Green contexts can only be made disjoint inside a single process
# (the joint cudaDevSmResourceSplit), so Experiment 2 has no two-process form.
#
# Experiment 3 is INTER-PROCESS: two independent runner invocations, because
# MPS is a service for multiple CUDA processes. That is a different process
# model, which is exactly why `mps_off` (two plain processes) is in the matrix:
# it is the baseline that bounds the process-model difference, so the green
# context and MPS numbers are never compared across an unmeasured gap.
#
# The run index is the OUTER loop and the configurations the inner one, so
# machine-state drift (thermals, memory fragmentation) spreads evenly across
# the configurations instead of favouring whichever one ran first.
#
# Two preconditions are checked before EVERY run, and both abort:
#   1. the clocks are locked. An idle governor drops the GPU to 306 MHz and
#      inflates every latency about 4x. `jetson_clocks` does not survive a
#      reboot, so this has to be checked rather than assumed.
#   2. nobody else is using the GPU. The board cannot be locked (no compute
#      mode on Tegra, no MIG, everybody shares the `nvidia` account), so the
#      only reliable test is to measure: a co-tenant benchmark was observed
#      inflating the median from 0.53 ms to 2.83 ms.
#
# Usage:
#   ./sweep.sh                     experiments 1 and 2 (~40 min)
#   ./sweep.sh --exp mps           a subset: each entry is matched as a
#   ./sweep.sh --exp isolated,mps  substring of the experiment name
#   ./sweep.sh --exp wq            the work queue experiment (not in the default
#                                  set: 60 runs, asked for explicitly)
#   ./sweep.sh --static-chunks 8   also run the static-partition jobs, passing
#                                  the device's TOTAL chunk count (read it off
#                                  `lspart`; a chunk is 2 SMs on an iGPU, so a
#                                  16-SM Orin has 8)
#   ./sweep.sh --dry-run           print every command, run nothing
#   ./sweep.sh --smoke             one fast run per configuration, to validate
#                                  the plumbing (writes to a SEPARATE directory:
#                                  it is not data)
#   ./sweep.sh --runs 3            fewer repetitions
#   ./sweep.sh --force             re-run configurations whose output exists
#
# The sweep is RESUMABLE: a run whose output already exists is skipped, so an
# interrupted sweep can simply be restarted.
set -euo pipefail

# BASH_SOURCE rather than $0: the script may be invoked through a symlink or
# from another directory. cd'ing into its directory and asking pwd gives an
# absolute path with no ".." left in it, which matters because the paths below
# are built by appending to it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA="$(cd "$HERE/.." && pwd)"

OUT="${OUT:-$DATA/bench}"       # results go in the data folder, outside the
                                # git repository
RUNNER="$HERE/runner"
DS="$DATA/dataset/imagenet-1k"
MOB="$DATA/models/mobilenetv2-12.engine"
RES="$DATA/models/resnet50-v2-7.engine"
# ResNet with multi-stream explicitly permitted, so it really issues two streams
# (main + one auxiliary). The default build leaves multi-stream off, so this is a
# different engine and the numbers are not comparable to the default build except
# where the job family below deliberately meets it.
RESAUX="$DATA/models/auxstreams/resnet50-v2-7-aux1.engine"

RUNS=5                          # independent runs per configuration
FIXED=32                        # fixed preprocessed input set
MIN_INF=5000                    # protocol: at least N inferences...
MIN_DUR=30                      # ...AND at least T seconds

# Where the GPU's frequency limits can be read. If min != max the governor is
# still free to clock the GPU down under us and every measurement is worthless.
GPU_DEVFREQ=/sys/class/devfreq/17000000.gpu

# Reference median for the sentinel: a fixed 500-inference MobileNet run on the
# whole GPU, clocks locked, board idle (measured 2026-09-17). If the sentinel
# drifts further than SENTINEL_TOL from this, something is stealing the GPU and
# the sweep stops. Re-measure it and pass --sentinel-ref if the board changes.
SENTINEL_REF_MS=0.535
SENTINEL_TOL=0.15
SENTINEL_INF=500

# MPS. The pipe directory is the isolation boundary: a client joins a server
# only if it was told about the same directory, so a private one keeps this
# daemon invisible to everyone else on the shared board.
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off                    # off | on | static

# Where the runner writes its CSVs while a run is in progress. /dev/shm is RAM.
#
# The guard watches the GPU; it does not watch the disk, and the disk is shared
# too. A CSV flush that ends up queued behind another writer can block for
# hundreds of milliseconds, and because the flush happens inside the measured
# window the GPU idles while it waits. Measured: 385 ms in one traced run, with
# the untraced tail identical whether the CSV went to disk or to RAM, so this is
# about removing a shared dependency rather than about a known bias. The files
# are moved into the results tree when the run finishes, so nothing downstream
# sees a difference.
TMPCSV="${TMPCSV:-/dev/shm/scarfato-csv}"

# 0 = skip the static-partition jobs. Otherwise it is the device's TOTAL chunk
# count, which only `lspart` knows: a chunk is 2 SMs on an integrated GPU, so a
# 16-SM Orin has 8.
STATIC_CHUNKS=0

EXPS="isolated,greenctx,mps"    # substrings. The work queue experiment is NOT
                                # in this list on purpose: it is 60 runs, so it
                                # is asked for explicitly with --exp wq
FORCE=0
DRY=0
SMOKE=0
OUT_GIVEN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY=1 ;;
        --smoke)         SMOKE=1 ;;
        --force)         FORCE=1 ;;
        # Value-taking options consume the NEXT argument, so they shift here as
        # well; the shift at the bottom of the loop then moves past the value.
        --runs)          RUNS="$2"; shift ;;
        --out)           OUT="$2"; OUT_GIVEN=1; shift ;;
        --exp)           EXPS="$2"; shift ;;
        --static-chunks) STATIC_CHUNKS="$2"; shift ;;
        --sentinel-ref)  SENTINEL_REF_MS="$2"; shift ;;
        # Print the header comment as the usage text. Stopping at the first
        # non-comment line means this never has to be kept in sync with however
        # long the header happens to be.
        #
        # Note that this is a single rule on purpose. sub() rewrites $0 in
        # place, so splitting it into "print the comments" and "exit on the
        # first non-comment" makes the second rule test the already-stripped
        # line -- which no longer looks like a comment -- and quit on line 2.
        # That produced an empty --help once; the two halves each work fine in
        # isolation, which is what makes it worth a warning.
        -h|--help)       awk 'NR>1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "$0"; exit 0 ;;
        *) echo "[sweep] unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$SMOKE" = 1 ]; then
    RUNS=1
    FIXED=8
    # Smoke results must never mix with real ones. The sweep is resumable, so a
    # smoke run left under a real name would later be skipped and silently kept.
    [ "$OUT_GIVEN" = 1 ] || OUT="$DATA/bench-smoke"
fi

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

# Who is holding the GPU's device nodes open? Returns non-zero if anyone is.
#
# This is the honest signal on Tegra. A CUDA process opens
# /dev/nvgpu/igpu0/ctrl and /dev/nvmap (plus the DRM render node) and keeps them
# open for its lifetime; a compiler opens none of them. Verified against our own
# runner.
#
# Two traps this avoids:
#   - /dev/nvidia* exists on this board but CUDA does NOT hold it, so a check for
#     it would silently match nothing. (It did, on the first attempt.)
#   - the MPS control daemon holds these nodes for its entire life without doing
#     any GPU work, so it has to be excluded or the guard would fire forever.
#
# This replaced a command-line pattern match, which produced a false positive in
# real use: a colleague's *compiler* was flagged because the source path it was
# compiling happened to contain "green_ctx". Names guess; file descriptors do
# not. The sentinel below is still the authority on whether the GPU is actually
# contended — this only answers the narrower question of who is attached to it.
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

# "Is the GPU ours right now?" Two tests, cheapest first: the device-holder
# scan above (fast and definitive about *attachment*) and then the sentinel
# (slow and definitive about *contention*). Non-zero if either says no.
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

# The daemon is a per-user service reached through a pipe directory. A private
# directory means other users are completely unaffected: an MPS client only
# joins a server whose pipe directory it was told about.
#
# One caveat shapes everything below: the control command `quit` TERMINATES the
# daemon. It is a teardown command, never a way to end a "session" of commands,
# so it must not be piped in accidentally — that silently kills the daemon and
# the next command then reports "Cannot find MPS control daemon process".
mps_ctl() {  # run one control command, print its output
    # nvidia-cuda-mps-control takes commands on stdin, one per line.
    printf '%s\n' "$1" | CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        nvidia-cuda-mps-control 2>&1
}

# There is no error code to test here, only the text of the reply, so the test
# is for the phrase the control tool uses when no daemon is listening.
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

# Called before a job is launched to bring the daemon into the mode that job
# needs. Doing nothing when the mode already matches is what keeps the
# unpartitioned experiments (all mode "off") from paying for a daemon restart
# before every single run.
mps_ensure() {  # $1 = off | on | static
    [ "$1" = "$MPS_MODE" ] && return 0
    mps_stop
    if [ "$1" != off ]; then
        mps_start "$1" || return 1
    fi
    MPS_MODE="$1"
}

# Static partitions, Experiment 3 phase 3.
#
# Partitions are created BEFORE any client connects: the driver rejects a
# static-partition request once the device already has active clients.
#
# A chunk is 2 SMs on an integrated GPU (Orin) on any architecture; it is 4 SMs
# for Ampere dGPUs under MPS v2 and 8 for Hopper+. So this 16-SM device exposes
# 8 chunks, which makes 50/50, 75/25 and 25/75 all exactly expressible -- at the
# SAME 2-SM granularity as a green context.
#
# lspart's legacy layout. Note that the GPU and the partition are separate
# columns, and that the GPU-<uuid>/<partition> identifier a client actually
# needs does NOT appear as such anywhere in the output:
#
#   GPU          Partition  free used  free used clients
#                             chunk chunk  SM   SM
#   GPU-74d43ed3 -           0   10     0   80 -
#   GPU-74d43ed3 Dx4AAAA...  -    5     -   40 Yes
#
# The device row (Partition = "-") carries the totals; every other row is a
# partition. The client-side identifier is the FULL gpu uuid and the partition
# string joined by "/".
PART_A=""
PART_B=""

mps_create_partitions() {  # $1 = the mobilenet percentage
    local mp="${1:-100}" uuid dev total ca cb out

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
    cb=$(( total - ca ))
    if [ "$ca" -lt 1 ] || [ "$cb" -lt 1 ]; then
        echo "[sweep] ABORT: $total chunks cannot express a ${mp}/$((100-mp)) split." >&2
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

# Teardown on any exit, including the abort paths, so an interrupted sweep
# cannot leave a daemon running on the shared board.
cleanup() { mps_stop; }
trap cleanup EXIT

# The run matrix.
#
# A job is a name plus a spec of the form
#      kind | mps | envA | envB | cmdA | cmdB | ratio
#   kind : solo (one process) | pair (two processes launched together)
#   mps  : off | on | static   (static also creates the partitions)
#   env* : extra environment for that client, e.g. the ATP variable
#   cmd* : the runner invocation, without the protocol flags
#   ratio: the mobilenet percentage; only the static jobs use it, to size the
#          two partitions, which are NOT equal at 75/25 or 25/75
#
# Pipe-separated and read back with `read`, which keeps empty fields
# unambiguous. Safe because no path contains a pipe.
declare -a JOB_NAME=() JOB_SPEC=()
add() { JOB_NAME+=("$1"); JOB_SPEC+=("$2"); }

# Experiment 1: one network, isolated.
# `whole` is the baseline; `gc100` is NOT redundant — it is the control that
# proves the green-context mechanism itself costs nothing, so a slowdown at 75%
# cannot be blamed on the mechanism.
for net in mobilenet resnet; do
    if [ "$net" = mobilenet ]; then eng="$MOB"; else eng="$RES"; fi
    add "isolated-scaling_${net}_whole" "solo|off|||$RUNNER $eng $DS --allocation whole|"
    for p in 100 75 50 25; do
        add "isolated-scaling_${net}_gc${p}" \
            "solo|off|||$RUNNER $eng $DS --allocation greenctx --sm-percent $p|"
    done
done

# Experiment 2: two networks in ONE process, disjoint green contexts.
#
# MobileNet + ResNet rather than mob-mob/res-res: that is the deployment being
# studied (a scheduler co-locates *different* workloads), and the mixed pair
# yields each network's co-run latency at 4, 8 and 12 SMs — exactly the points
# Experiment 1 measures in isolation.
add "concurrent-greenctx_baseline" \
    "solo|off|||$RUNNER --dataset $DS --allocation whole --net $MOB:100:mobilenet --net $RES:100:resnet|"

for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "concurrent-greenctx_${mp}_${rp}" \
        "solo|off|||$RUNNER --dataset $DS --allocation greenctx --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
done

# Experiment 3: MPS, TWO processes.
#
# Phase 1, the minimal comparison plus the control that makes it readable:
#   mps_single : one process under MPS, vs the isolated `whole` run
#   mps_off    : two processes, no MPS                  -> the 2-process baseline
#   mps_default: two processes under MPS                -> the co-scheduling effect
#
# On alignment: the two processes have no shared memory to rendezvous in, so
# each starts its timed phase when its own warm-up ends. The skew is the
# difference in pre-timed work — about 0.16 s (200 inferences: 0.11 s MobileNet
# vs 0.27 s ResNet) out of a 30 s window, so roughly 0.5%. A `--start-at`
# deadline would make it exact; it is not needed to see an effect of the size
# MPS would have to show in order to matter.
add "concurrent-mps_mps_single"  "solo|on|||$RUNNER $MOB $DS --allocation whole|"
add "concurrent-mps_mps_off"     "pair|off|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"
add "concurrent-mps_mps_default" "pair|on|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"

# Phase 2: active thread percentage. This is NOT spatial partitioning. ATP
# limits the fraction of the SM's thread slots a client may occupy, which scales
# throughput but does not reserve particular SMs.
for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "concurrent-mps_mps_atp_${mp}_${rp}" \
        "pair|on|CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$mp|CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$rp|$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole"
done

# Phase 3: static SM partitions, the actual spatial analogue of green contexts
# and the only part with an unknown. Documented behaviour: partitions are
# exclusive to their clients and cannot be over-allocated, and a client that
# does not set CUDA_MPS_SM_PARTITION FAILS to create a context — so a
# misconfigured run dies loudly instead of silently measuring nothing.
#
# Only added when --static-chunks is given, because the chunk count has to be
# read off `lspart` on the real board first.
if [ "$STATIC_CHUNKS" -gt 0 ]; then
    for split in 50:50 75:25 25:75; do
        mp="${split%%:*}"; rp="${split##*:}"
        add "concurrent-mps_mps_static_${mp}_${rp}" \
            "pair|static|||$RUNNER $MOB $DS --allocation whole|$RUNNER $RES $DS --allocation whole|$mp"
    done
fi

# The work queue experiment: Experiment 2's setup with the work queue half of
# the green context descriptor set EXPLICITLY instead of inherited.
#
# Experiment 2 never put a work queue resource in the descriptor at all, so every
# run of it used the driver default: all contexts sharing one set of queues, with
# a concurrency limit of CUDA_DEVICE_MAX_CONNECTIONS. Measured on this board that
# default is 8 normally and 2 under MPS — they are not the same, which is why
# this is an experiment rather than an assumption.
#
# The hypothesis is the guide's own warning: independent work on different
# streams that maps to the SAME work queue acquires a false dependence and is
# serialised even when SMs are free. MobileNet's kernels are short and numerous,
# so it submits far more often per unit of work than ResNet, and it is the
# network that should show the effect.
#
# Three knobs have to be distinguished, and only two of them are runner options:
#   --wq-scope  how the driver shares queues between contexts
#   --wq-limit  the concurrency declared to the driver as a HINT (not a stream
#               limit, and not a cap on anything)
#   CUDA_DEVICE_MAX_CONNECTIONS is a different thing again: it bounds how many
#   queues exist at all, is read from the environment before CUDA initialises,
#   and therefore lives in the env field of the spec. Both knobs are pinned in
#   every arm so the arms stay comparable.
#
# Five arms x two splits = the 2x2 factorial on (scope, limit) plus the
# limit-1 negative control. `devctx_8` must reproduce Experiment 2's numbers; if
# it does not, the code path is wrong and the rest is uninterpretable.
#
# Select with:  ./sweep.sh --exp wq
for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:8:device_ctx balanced:8:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               devctx:1:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "concurrent-wq_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
    done
done

# The unpartitioned pair: no green context, so no scope applies and only the
# queue BUDGET is set. These are Experiment 2's `baseline` run twice — once at
# the limit Experiment 2 actually had (8) and once at the one Experiment 3 had
# (2). That pair measures the process model with the budget held equal, which the
# Exp 2 versus Exp 3 comparison cannot do because those two differ in both at
# once. It is the cleanest pair in the whole matrix and it costs 10 runs.
for lim in 8 2; do
    add "concurrent-wq_unpart_${lim}" \
        "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation whole --net $MOB:100:mobilenet --net $RES:100:resnet|"
done

# The auxiliary-stream family: Experiment 2's pair again, but with ResNet built
# to allow one auxiliary stream, so it issues two streams of its own.
#
# ResNet is the only engine here with any intra-network parallelism: TensorRT
# says so in its build log ("Number of aux streams is 1", 7 events per
# inference) and measured against a trace the two streams overlap only about
# 4.5% of the time. MobileNet reports "cannot find an opportunity to leverage
# it" under every cap, so it never has more than one stream.
#
# That gives the pair three concurrent stream-ordered workloads - ResNet's two
# plus MobileNet's one - so the declared concurrency can finally be chosen from
# the measured parallelism rather than guessed. The family is built around three
# questions:
#   - the sharing scope at the coherent concurrency (limit 3), which is the
#     controlled comparison the whole work queue argument rests on;
#   - the same comparison at the per-client reading of coherence (limit 2), to
#     settle whether the hint is read per context or device-wide;
#   - the sweep he asked for: 1, the observed concurrency, and the device's
#     default (8) and maximum (32).
# Limit 8 at device_ctx is the attribution arm: it meets Experiment 4's
# devctx_8 rows at the same limit, scope and ratio, so the difference there is
# the auxiliary stream and nothing else.
for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:3:device_ctx balanced:3:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               balanced:1:green_ctx_balanced balanced:8:green_ctx_balanced \
               balanced:32:green_ctx_balanced devctx:8:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "concurrent-aux_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RESAUX:$rp:resnet|"
    done
done

# The aux-enabled engine on its own, no green context and nothing else changed:
# the reference its own pair rows are read against, and the row that says whether
# permitting the auxiliary stream changed ResNet's own latency at all.
add "concurrent-aux_aux1_whole" \
    "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=8||$RUNNER $RESAUX $DS --allocation whole|"

# ---------------------------------------------------------------------------
# The same three intra-process families, re-measured with ONE submitting
# thread.
#
# The pair has so far always been submitted by two threads, one per network.
# That is the naive application and it stays: every table above is built on it.
# It is also the source of the MobileNet tail, because two threads contend in
# the launch path of one CUDA context, a launch is occasionally delayed, and the
# GPU then idles inside the event window. MPS never showed the tail for the
# mirror-image reason: it is two processes, and two processes do not share that
# path.
#
# These arms change exactly one thing. A single thread drives both networks'
# timed loops, keeping one inference in flight per network — the same depth the
# threaded path has — so the two sets differ in the submitter and in nothing
# else: same engines, same green contexts, same joint SM split, same queue
# settings, same protocol. Measured effect on the pair at 75/25: MobileNet's
# tail falls from 0.348 to 0.050 ms and its median improves slightly, while
# throughput does not move.
#
# Select with:  ./sweep.sh --exp 1t
ST="--submit-threads 1"

for split in 50:50 75:25 25:75; do
    mp="${split%%:*}"; rp="${split##*:}"
    add "concurrent-greenctx-1t_${mp}_${rp}" \
        "solo|off|||$RUNNER --dataset $DS --allocation greenctx $ST --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
done
add "concurrent-greenctx-1t_baseline" \
    "solo|off|||$RUNNER --dataset $DS --allocation whole $ST --net $MOB:100:mobilenet --net $RES:100:resnet|"

for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:8:device_ctx balanced:8:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               devctx:1:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "concurrent-wq-1t_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx $ST --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RES:$rp:resnet|"
    done
done
for lim in 8 2; do
    add "concurrent-wq-1t_unpart_${lim}" \
        "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation whole $ST --net $MOB:100:mobilenet --net $RES:100:resnet|"
done

for split in 50:50 75:25; do
    mp="${split%%:*}"; rp="${split##*:}"
    for arm in devctx:3:device_ctx balanced:3:green_ctx_balanced \
               devctx:2:device_ctx balanced:2:green_ctx_balanced \
               balanced:1:green_ctx_balanced balanced:8:green_ctx_balanced \
               balanced:32:green_ctx_balanced devctx:8:device_ctx; do
        tag="${arm%%:*}"; rest="${arm#*:}"
        lim="${rest%%:*}"; scope="${rest##*:}"
        add "concurrent-aux-1t_${tag}_${lim}_${mp}_${rp}" \
            "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=$lim||$RUNNER --dataset $DS --allocation greenctx $ST --wq-scope $scope --wq-limit $lim --net $MOB:$mp:mobilenet --net $RESAUX:$rp:resnet|"
    done
done
# One network, so this arm has a single submitting thread in both sets; it is
# repeated here only so the one-thread family stands on its own.
add "concurrent-aux-1t_aux1_whole" \
    "solo|off|CUDA_DEVICE_MAX_CONNECTIONS=8||$RUNNER $RESAUX $DS --allocation whole|"

# Has this configuration already been measured?
#
# The number of stats files depends on how many NETWORKS a run has, not on how
# many PROCESSES it uses — and getting that wrong caused a real bug. Experiment 2
# is one process with two networks, so the runner writes two labelled files,
# while this check (keyed on kind=solo) looked for a single unlabelled one. Every
# Experiment 2 run therefore looked unfinished and was re-measured on every
# restart: 20 wasted runs, about 12 minutes.
#
# Keying on the shape of the output instead of the job kind is what makes that
# impossible. Either naming counts, a missing file means not-finished, and at
# least one of the two is always written.
#
# The stats file is also the completion marker, written only at the very end of a
# run. That is what makes an interrupted sweep safe to restart: a run killed
# mid-measurement leaves its latency CSV behind but no stats file, so it is not
# "done" and gets re-run as if it had never started.
job_done() {  # $1 = output prefix
    [ -f "$1_stats.csv" ] || \
        { [ -f "$1_mobilenet_stats.csv" ] && [ -f "$1_resnet_stats.csv" ]; }
}

# Partial output with no completion marker: somebody killed a run in flight.
interrupted() {  # $1 = output prefix
    job_done "$1" && return 1
    [ -f "$1_latencies.csv" ] || [ -f "$1_mobilenet_latencies.csv" ]
}

move_back() {  # $1 = tmpfs prefix, $2 = results prefix
    local f
    for f in "$1"*.csv; do
        [ -e "$f" ] || continue
        mv -f "$f" "$2${f#"$1"}"
    done
}

do_job() {  # $1 = spec, $2 = output prefix
    local spec="$1" out="$2"
    local kind mps envA envB cmdA cmdB ratio rc_a=0 rc_b=0
    # IFS is scoped to this one read, so it does not leak into the rest of the
    # function. `read` splits on the pipe and leaves empty fields empty.
    IFS='|' read -r kind mps envA envB cmdA cmdB ratio <<< "$spec"

    if [ "$mps" = static ]; then
        mps_create_partitions "$ratio" || return 1
        envA="CUDA_MPS_SM_PARTITION=$PART_A"
        envB="CUDA_MPS_SM_PARTITION=$PART_B"
    fi

    local penv=""
    [ "$mps" != off ] && penv="CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE"

    # The CSVs go to RAM during the run and are moved into the tree afterwards.
    local tmp="$TMPCSV/$(basename "$out")"

    # $penv, $envA and $cmdA are deliberately unquoted: each is a string that
    # must split into separate words for `env`. They are empty when unused, and
    # an empty unquoted expansion simply disappears. (That is also why the
    # shellcheck warnings here are disabled rather than "fixed".)
    if [ "$kind" = solo ]; then
        # shellcheck disable=SC2086
        env $penv $envA $cmdA $PROTO --csv "$tmp" > "$out.log" 2>&1 || return 1
        move_back "$tmp" "$out"
        return 0
    fi

    # Two processes, launched back to back so their warm-ups overlap as far as
    # the OS allows. $! must be read immediately, before the next command can
    # overwrite it, and each exit status is collected separately rather than
    # letting the failure of one mask the other.
    # shellcheck disable=SC2086
    env $penv $envA $cmdA $PROTO --csv "${tmp}_mobilenet" > "${out}_mobilenet.log" 2>&1 &
    local pid_a=$!
    # shellcheck disable=SC2086
    env $penv $envB $cmdB $PROTO --csv "${tmp}_resnet" > "${out}_resnet.log" 2>&1 &
    local pid_b=$!

    # `|| rc=$?` instead of letting the failure propagate, because `set -e`
    # would otherwise abort before we could report WHICH of the two failed.
    wait "$pid_a" || rc_a=$?
    wait "$pid_b" || rc_b=$?
    if [ "$rc_a" -ne 0 ] || [ "$rc_b" -ne 0 ]; then
        echo "[sweep] ABORT: run failed (mobilenet rc=$rc_a, resnet rc=$rc_b)." >&2
        echo "        See ${out}_mobilenet.log / ${out}_resnet.log" >&2
        return 1
    fi
    move_back "$tmp" "$out"
    return 0
}
mkdir -p "$OUT" "$TMPCSV"
printf '[sweep] output:  %s\n' "$OUT"
printf '[sweep] configs: %d, runs each: %d, total runs: %d, experiments: %s\n' \
       "${#JOB_NAME[@]}" "$RUNS" "$(( ${#JOB_NAME[@]} * RUNS ))" "$EXPS"

PROTO="--fixed $FIXED"
if [ "$SMOKE" = 1 ]; then
    PROTO="$PROTO --one-pass"
    echo "[sweep] SMOKE MODE: one fast run per configuration, results are NOT measurements"
else
    PROTO="$PROTO --min-inferences $MIN_INF --min-duration $MIN_DUR"
fi

if [ "$DRY" = 0 ]; then
    # Clear any daemon a previous sweep left behind. A dropped SSH connection
    # kills bash without running the EXIT trap, so a daemon can survive it; it
    # lives in our private pipe dir, so anything found there is ours to stop.
    # Doing it here, rather than relying on the restart when the first MPS job
    # starts, keeps the state known from the very first run.
    mps_stop

    check_clocks || exit 1
    # The clocks are a hard precondition, and so is require_quiet_gpu for real
    # runs. Smoke data is explicitly not a measurement, so there it only warns:
    # validating the plumbing on a busy board is the whole point of smoke mode.
    if ! require_quiet_gpu; then
        [ "$SMOKE" = 1 ] || exit 1
        echo "[sweep] WARNING: GPU is busy; proceeding because this is a SMOKE test."
    fi
fi

for r in $(seq 1 "$RUNS"); do
    for i in "${!JOB_NAME[@]}"; do
        # Two names, deliberately different:
        #   name  the full identity — for the log, and for the --exp filter
        #   out   the path the files go to, which is shorter because the
        #         directory already carries the experiment and the config
        # So the tree is bench/<experiment>/mobilenet_gc50/r3_stats.csv, rather
        # than one flat directory repeating the whole path in every file name.
        # The experiment folders say what the experiment is — a reader learns
        # nothing from "exp1", and the numbering can move as the thesis is
        # written while the description cannot.
        name="${JOB_NAME[$i]}_r${r}"
        exp="${JOB_NAME[$i]%%_*}"          # isolated-scaling
        cfg="${JOB_NAME[$i]#${exp}_}"      # mobilenet_gc50
        out="$OUT/$exp/$cfg/r${r}"

        # The --exp filter. Each entry is a SUBSTRING of the experiment name, so
        # `--exp mps` selects concurrent-mps without anyone having to remember
        # the full folder name, and the default list selects all of them.
        selected=0
        IFS=',' read -ra want <<< "$EXPS"
        for w in "${want[@]}"; do
            case "$exp" in *"$w"*) selected=1; break ;; esac
        done
        [ "$selected" = 1 ] || continue

        # Kind and mps are needed here for the done-check and the dry-run
        # display; the command fields are only used by the dry run.
        IFS='|' read -r kind mps _eA _eB _cA _cB _ratio <<< "${JOB_SPEC[$i]}"

        if job_done "$out" && [ "$FORCE" = 0 ]; then
            printf '[sweep] skip %s (already done)\n' "$name"
            continue
        fi

        # Start again from scratch, but say why: without this, a run appearing
        # twice in the log after a restart looks like a bug rather than the
        # resumability doing its job.
        if interrupted "$out"; then
            printf '[sweep] re-running %s (a previous attempt was interrupted)\n' "$name"
        fi

        if [ "$DRY" = 1 ]; then
            printf '  %-26s [%s/%s]\n' "$name" "$kind" "$mps"
            if [ "$kind" = solo ]; then
                printf '      A: %s %s --csv %s\n' "$_cA" "$PROTO" "$out"
            else
                printf '      A: %s %s --csv %s_mobilenet\n' "$_cA" "$PROTO" "$out"
                printf '      B: %s %s --csv %s_resnet\n' "$_cB" "$PROTO" "$out"
            fi
            continue
        fi

        # The per-run directory has to exist before the runner opens its CSVs.
        # Created here, after the dry-run branch, so a dry run stays a dry run.
        mkdir -p "$(dirname "$out")"

        # Re-checked before every run, because interference that starts
        # mid-sweep is exactly what a single startup check would miss.
        if [ "$SMOKE" = 0 ]; then
            require_quiet_gpu || exit 1
        fi

        # Bring the MPS daemon into the mode this job needs (off/on/static).
        mps_ensure "$mps" || exit 1

        printf '[sweep] %s\n' "$name"
        do_job "${JOB_SPEC[$i]}" "$out" || exit 1

        [ "$mps" = static ] && mps_remove_partitions
    done
done

rm -f "$OUT/_sentinel_latencies.csv" "$OUT/_sentinel_stats.csv"

echo
if [ "$DRY" = 1 ]; then
    echo "[sweep] dry run: nothing was executed."
elif [ "$SMOKE" = 1 ]; then
    echo "[sweep] smoke test complete — the plumbing works."
    echo "        Inspect a log, then run: ./sweep.sh"
else
    echo "[sweep] done. Aggregate with:  python3 $HERE/aggregate.py --out $OUT"
fi
