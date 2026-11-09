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
#   ./measure/sweep.sh --exp stress-triple            one family: a whole family name
#   ./measure/sweep.sh --exp isolated-scaling,stress-new   several, comma-separated
#   ./measure/sweep.sh --exp section       a group: section, appendix, mem or all
#   ./measure/sweep.sh --static-chunks 8   also run the static-partition jobs, passing
#                                  the device's TOTAL chunk count (read it off
#                                  `lspart`; a chunk is 2 SMs on an iGPU, so a
#                                  16-SM Orin has 8)
#   ./measure/sweep.sh --dry-run           print every command, run nothing
#   ./measure/sweep.sh --smoke             one fast run per configuration, to validate
#                                  the plumbing (writes to a SEPARATE directory:
#                                  it is not data)
#   ./measure/sweep.sh --runs 3            fewer repetitions
#   ./measure/sweep.sh --only REGEX        within the selected families, only the jobs whose
#                                  full name (family_config) matches the extended
#                                  regular expression, e.g. '^stress-sensitivity_mobilenet-'
#   ./measure/sweep.sh --force             re-run configurations whose output exists
#
# The sweep is RESUMABLE: a run whose output already exists is skipped, so an
# interrupted sweep can simply be restarted.
set -euo pipefail

# BASH_SOURCE rather than $0: the script may be invoked through a symlink or
# from another directory. cd'ing into its directory and asking pwd gives an
# absolute path with no ".." left in it, which matters because the paths below
# are built by appending to it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../runner/measure
ROOT="$(cd "$HERE/.." && pwd)"                          # .../runner, the repository

# The data is NOT in the repository and not on the system disk. The engines, the
# image set, the labels and the results live on the NVMe, which is a separate mount:
# the root filesystem is small and shared, and it once filled up under a campaign's
# worth of CSVs. Everything below is built by appending to this root, so DATA
# repoints the whole chain.
DATA="${DATA:-/mnt/nvme/f.scarfato}"

if [ ! -d "$DATA" ]; then
    echo "sweep.sh: data root not found: $DATA" >&2
    exit 1
fi

OUT="${OUT:-$DATA/bench}"       # results go in the data folder, outside the
                                # git repository
RUNNER="$ROOT/runner"
DS="$DATA/dataset/imagenet-1k"
MOB="$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine"
RES="$DATA/models/resnet50-v2-7/engine/resnet50-v2-7.engine"
# ResNet with multi-stream explicitly permitted, so it really issues two streams
# (main + one auxiliary). The default build leaves multi-stream off, so this is a
# different engine and the numbers are not comparable to the default build except
# where the job family below deliberately meets it.
RESAUX="$DATA/models/resnet50-v2-7/engine/resnet50-v2-7-aux1.engine"

# The networks added after the first campaign. Inception-v3 takes a 299x299 input
# and saturates at six concurrent stream-ordered workloads per client; the ViT is
# a serial chain, so it is here as a control rather than a subject.
INC="$DATA/models/inception_v3-onnx-float/engine/inception_v3.engine"
INC1="$DATA/models/inception_v3-onnx-float/engine/inception_v3-aux1.engine"
INC5="$DATA/models/inception_v3-onnx-float/engine/inception_v3-aux5.engine"
INC2="$DATA/models/inception_v3-onnx-float/engine/inception_v3-aux2.engine"
INC4="$DATA/models/inception_v3-onnx-float/engine/inception_v3-aux4.engine"
VIT="$DATA/models/vit-base-patch16-224/engine/vit-base-patch16-224.engine"
# RetinaNet-9 is a detector: ten output tensors and a 480x640 input, both of
# which the runner now honours. It adopts more streams than anything else here.
RN="$DATA/models/retinanet-9/engine/retinanet-9.engine"
RN1="$DATA/models/retinanet-9/engine/retinanet-9-aux1.engine"
RN8="$DATA/models/retinanet-9/engine/retinanet-9-aux8.engine"
RN16="$DATA/models/retinanet-9/engine/retinanet-9-aux16.engine"
# The validation networks (2026-10-13), measured after the p99 models were built on the four above:
# the four in INT8 (engines.mk int8, calibrated scales), and EfficientNet-B0. The ViT is $VIT.
MOBQ="$DATA/models/mobilenetv2-12/engine/mobilenetv2-12-int8.engine"
RESQ="$DATA/models/resnet50-v2-7/engine/resnet50-v2-7-int8.engine"
INCQ="$DATA/models/inception_v3-onnx-float/engine/inception_v3-int8.engine"
RNQ="$DATA/models/retinanet-9/engine/retinanet-9-int8.engine"
EFF="$DATA/models/efficientnet_b0/engine/efficientnet_b0.engine"

RUNS=5                          # independent runs per configuration
FIXED=32                        # fixed preprocessed input set
MIN_INF="${MIN_INF:-5000}"      # protocol: at least N inferences...
MIN_DUR="${MIN_DUR:-30}"        # ...AND at least T seconds. Both can be set from the environment: the
                                # equal-work runs (energy-same-n) use MIN_INF=10000 MIN_DUR=0, so that every
                                # network stops by itself after the same number of inferences.

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

SWEEP_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SWEEP_DIR/lib/guards.sh"
source "$SWEEP_DIR/lib/mps.sh"
trap cleanup EXIT

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

ONLY=""                        # --only: an extended regex on family_config
EXPS=""                         # --exp: whole family names, comma-separated; required
FORCE=0
DRY=0
SMOKE=0
OUT_GIVEN=0

# The submission-model arms are the only ones that care where the submitting
# thread runs and at what priority, so the core is a parameter (recorded in
# the configuration name, because a result you cannot re-derive is not one).
WITH_PRIORITY=0                  # --with-priority: measure them at all
PRIO=nice                        # nice | fifo, the mechanism if we do
SUBMIT_CORE="${SUBMIT_CORE:-6}"   # one core, for the pinned arms
SUBMIT_CORE_B="${SUBMIT_CORE_B:-7}"

# Named groups for --exp, so a stage is one command rather than a list of folders. The split
# is the one the tables make: `section` is every family whose rows the results section renders,
# `appendix` is the rest, which the history file and the diagnostic tools read. Kept in step
# with the generator by analysis/overview.py, which derives the same split from the source.
SECTION_FAMILIES="isolated-scaling pair-greenctx-1submit pair-mpsstatic-2submit pair-greenctx-auxstreams-2submit submission-model workqueue-declaration workqueue-aux-streams interference-additivity memory-budget-greenctx memory-interferers memory-mps-interferers memory-mps-budget memory-mps-interferers-4sm memory-two-threads"
APPENDIX_FAMILIES="pair-greenctx-2submit pair-greenctx-workqueue-2submit pair-greenctx-workqueue-1submit pair-greenctx-auxstreams-1submit cpu-affinity repeatability workqueue-three-clients"
MEM_FAMILIES="memory-budget-greenctx memory-interferers memory-mps-interferers memory-mps-budget memory-mps-interferers-4sm memory-two-threads"
ALL_FAMILIES="$SECTION_FAMILIES $APPENDIX_FAMILIES"

SWEEP_ARGS=("$@")   # the command line, for the manifest; the loop below consumes it
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)       DRY=1 ;;
        --smoke)         SMOKE=1 ;;
        --force)         FORCE=1 ;;
        # Value-taking options consume the NEXT argument, so they shift here as
        # well; the shift at the bottom of the loop then moves past the value.
        --runs)          RUNS="$2"; shift ;;
        --out)           OUT="$2"; OUT_GIVEN=1; shift ;;
        --exp)
            # A group name expands to its family list; anything else is a list of whole family
            # names (see selects below).
            case "$2" in
                section)  EXPS="$(printf '%s' "$SECTION_FAMILIES" | tr ' ' ',')" ;;
                appendix) EXPS="$(printf '%s' "$APPENDIX_FAMILIES" | tr ' ' ',')" ;;
                all)      EXPS="$(printf '%s' "$ALL_FAMILIES" | tr ' ' ',')" ;;
                mem)      EXPS="$(printf '%s' "$MEM_FAMILIES" | tr ' ' ',')" ;;
                *)        EXPS="$2" ;;
            esac
            shift ;;
        --only)          ONLY="$2"; shift ;;
        --static-chunks) STATIC_CHUNKS="$2"; shift ;;
        --with-priority) WITH_PRIORITY=1 ;;
        --submit-core)   SUBMIT_CORE="$2"; shift ;;
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

# A quiet CPU. These arms measure host-side scheduling, so a co-tenant doing work
# on the CPU is not background noise here, it is a different experiment -- and the
# GPU check cannot see it. Loose on purpose: our own run contributes about one, and
# this runs before the runner starts, so it sees the ambient load plus the previous
# run's teardown.
CPU_LOAD_MAX="${CPU_LOAD_MAX:-1.5}"
case "$EXPS" in *submit*) CPU_GUARD=1 ;; *) CPU_GUARD=0 ;; esac

# The priority arms need CAP_SYS_NICE, and nothing here tests for it: the runner
# refuses at startup with the remedy in the message. Failing on the first arm of
# the first configuration is early enough, and it keeps the knowledge in one place.
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
# "Is the GPU ours right now?" Two tests, cheapest first: the device-holder
# scan above (fast and definitive about *attachment*) and then the sentinel
# (slow and definitive about *contention*). Non-zero if either says no.
# The daemon is a per-user service reached through a pipe directory. A private
# directory means other users are completely unaffected: an MPS client only
# joins a server whose pipe directory it was told about.
#
# One caveat shapes everything below: the control command `quit` TERMINATES the
# daemon. It is a teardown command, never a way to end a "session" of commands,
# so it must not be piped in accidentally — that silently kills the daemon and
# the next command then reports "Cannot find MPS control daemon process".
# There is no error code to test here, only the text of the reply, so the test
# is for the phrase the control tool uses when no daemon is listening.
# Called before a job is launched to bring the daemon into the mode that job
# needs. Doing nothing when the mode already matches is what keeps the
# unpartitioned experiments (all mode "off") from paying for a daemon restart
# before every single run.
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

# Teardown on any exit, including the abort paths, so an interrupted sweep
# cannot leave a daemon running on the shared board.

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
source "$HERE/lib/matrix.sh"

# How many jobs the current selection contains. Counted before the loops, so
# every job line can say "N/M" and `tail -f` becomes a progress indicator on its
# own. A resumed sweep counts the jobs it skips: they are done, so they are
# progress, not gaps in the numbering.
#
# selects <family>: is the family one of the --exp entries? Whole names only: when an entry was a
# substring, --exp stress-sensitivity-tb also ran stress-sensitivity-tbs.
selects() {
    local w want
    IFS=',' read -ra want <<< "$EXPS"
    for w in "${want[@]}"; do [ "$1" = "$w" ] && return 0; done
    return 1
}
if [ -z "$EXPS" ]; then
    echo "[sweep] say what to run: --exp <family>[,<family>...] or a group (section, appendix, mem, all)" >&2
    exit 2
fi
SELECTED=0
job_no=0
for n in "${JOB_NAME[@]}"; do
    [ -n "$ONLY" ] && ! [[ $n =~ $ONLY ]] && continue
    selects "${n%%_*}" && SELECTED=$((SELECTED + 1))
done
printf '[sweep] %d jobs selected, %d runs each, %d runs total\n' \
       "$SELECTED" "$RUNS" "$((SELECTED * RUNS))"

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
#
# A pair run (two measured processes) writes one labelled stats file per network, <prefix>_<net>_stats.csv
# (net_label below), so it is done when two of them exist.
job_done() {  # $1 = output prefix
    [ -f "$1_stats.csv" ] || [ "$(compgen -G "$1_*_stats.csv" | wc -l)" -ge 2 ]
}

# Partial output with no completion marker: somebody killed a run in flight.
interrupted() {  # $1 = output prefix
    job_done "$1" && return 1
    compgen -G "$1_*latencies.csv" > /dev/null
}

# The label of a measured process in a pair run: the network named after the last ':' of its --net. The
# older pair families name no --net in their commands and keep the labels they always had, mobilenet for
# the first process and resnet for the second ($2).
net_label() {  # $1 = runner command, $2 = the label when it names no --net
    local l
    l=$(sed -n 's/.*--net [^ ]*:\([A-Za-z0-9_]*\).*/\1/p' <<< "$1")
    echo "${l:-$2}"
}

move_back() {  # $1 = tmpfs prefix, $2 = results prefix
    local f
    for f in "$1"*.csv; do
        [ -e "$f" ] || continue
        mv -f "$f" "$2${f#"$1"}"
    done
}

# A measured process that has not finished in this many seconds is stopped, and its run counts as
# not done (no stats file), so a restart measures it again. A run takes under 3 minutes; two TensorRT
# clients under MPS static partitions were seen to hang right after connecting (2 of 16 runs of
# stress-validation), spinning for hours instead of failing.
RUN_TIMEOUT="${RUN_TIMEOUT:-600}"

# POWER=1: each run also writes <out>_power.log, the board's three INA3221 rails sampled every 200 ms
# from the start of the first process to the end of the target. hwmon and not tegrastats, because
# tegrastats stamps its lines to the second; here each line starts with /proc/uptime (10 ms, the
# boot clock, which equals the runner's CLOCK_MONOTONIC stamps on a board that never suspends), so
# the analysis can cut exactly the target's timed window. The loop forks only sleep: read is a
# builtin, so the sampler costs the CPU next to nothing.
POWER="${POWER:-0}"
POWER_HW=""
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/in1_label" 2>/dev/null)" = VDD_GPU_SOC ] && POWER_HW="$h"
done
power_start() {  # $1 = output prefix; sets POWER_PID
    POWER_PID=""
    [ "$POWER" = 1 ] || return 0
    [ -n "$POWER_HW" ] || { echo "[sweep] ABORT: POWER=1 but no INA3221 with VDD_GPU_SOC" >&2; exit 1; }
    (
        local h="$POWER_HW" up _ v1 c1 v2 c2 v3 c3
        echo "uptime_s gpu_soc_mV gpu_soc_mA cpu_cv_mV cpu_cv_mA sys5v_mV sys5v_mA"
        while :; do
            read -r up _ < /proc/uptime
            read -r v1 < "$h/in1_input"; read -r c1 < "$h/curr1_input"
            read -r v2 < "$h/in2_input"; read -r c2 < "$h/curr2_input"
            read -r v3 < "$h/in3_input"; read -r c3 < "$h/curr3_input"
            echo "$up $v1 $c1 $v2 $c2 $v3 $c3"
            sleep 0.2
        done
    ) > "$1_power.log" 2>&1 &
    POWER_PID=$!
}
power_stop() {
    [ -n "${POWER_PID:-}" ] || return 0
    kill "$POWER_PID" 2>/dev/null || true
    wait "$POWER_PID" 2>/dev/null || true
    POWER_PID=""
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
        power_start "$out"
        # shellcheck disable=SC2086
        timeout -k 30 "$RUN_TIMEOUT" env $penv $envA $cmdA $PROTO --csv "$tmp" > "$out.log" 2>&1 ||
            { power_stop; return 1; }
        power_stop
        move_back "$tmp" "$out"
        return 0
    fi

    # One measured client plus a co-runner that is a SEPARATE PROCESS. This is the
    # shape the MPS arms need: MPS is a service for multiple processes, so a
    # co-runner inside the measured process would be the arrangement the MPS arm
    # exists to contrast with.
    #
    # The co-runner is started first, so it is in its steady state before the
    # target's timed phase begins, and it is stopped as soon as the target exits,
    # while the daemon and its partitions are still up. The lead is 3 s: long
    # enough for the co-runner's own allocation and warm-up, short enough not to
    # matter to a 30 s window, and it is before the target starts so it cannot
    # fall inside the measurement.
    #
    # Several co-runners: cmdB holds their commands separated by ';'. Under static MPS the i-th one
    # takes the i-th partition after the target's, and any beyond the last partition share it
    # (two clients on one partition). The first logs to _itf.log, the next to _itf2.log, ...
    #
    # A runner co-runner (one with --csv) is a victim too: its per-inference rows are kept as
    # <out>_co<k>.csv, the k-th co-runner, so the analysis can cut them to the target's timed window
    # (the co-runner starts 3 s earlier and is stopped after the target). Not *_latencies.csv, so no
    # script that reads targets picks them up. A stale file is removed first: a co-runner that fails to
    # start must leave nothing behind to be mistaken for its rows.
    if [ "$kind" = itf ]; then
        local cos c k=1 pids=() envk log csvs=() cp
        IFS=';' read -ra cos <<< "$cmdB"
        power_start "$out"
        for c in "${cos[@]}"; do
            cp=$(sed -n 's/.*--csv \([^ ]*\).*/\1/p' <<< "$c")
            csvs+=("$cp")
            [ -n "$cp" ] && rm -f "${cp}_latencies.csv"
            envk="$envB"
            if [ "$mps" = static ]; then
                local j=$k
                [ "$j" -ge "${#PART_LIST[@]}" ] && j=$(( ${#PART_LIST[@]} - 1 ))
                envk="CUDA_MPS_SM_PARTITION=${PART_LIST[$j]}"
            fi
            log="${out}_itf.log"; [ "$k" -gt 1 ] && log="${out}_itf$k.log"
            # shellcheck disable=SC2086
            timeout -k 30 "$RUN_TIMEOUT" env $penv $envk $c > "$log" 2>&1 &
            pids+=($!)
            k=$((k + 1))
        done
        sleep 3
        # shellcheck disable=SC2086
        timeout -k 30 "$RUN_TIMEOUT" env $penv $envA $cmdA $PROTO --csv "$tmp" > "$out.log" 2>&1 || rc_a=$?
        power_stop
        kill -TERM "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
        move_back "$tmp" "$out"
        for k in "${!csvs[@]}"; do
            cp="${csvs[$k]}"
            [ -n "$cp" ] && [ -f "${cp}_latencies.csv" ] && mv -f "${cp}_latencies.csv" "${out}_co$((k + 1)).csv"
        done
        return "$rc_a"
    fi

    # Two processes, launched back to back so their warm-ups overlap as far as
    # the OS allows. $! must be read immediately, before the next command can
    # overwrite it, and each exit status is collected separately rather than
    # letting the failure of one mask the other.
    #
    # Each file is labelled with its network (net_label). With POWER=1 the rails are sampled from before the two start until both have exited: the
    # equal-work runs (energy-same-n) measure the energy of the whole job, each network stopping by
    # itself.
    local la lb
    la=$(net_label "$cmdA" mobilenet)
    lb=$(net_label "$cmdB" resnet)
    power_start "$out"
    # shellcheck disable=SC2086
    timeout -k 30 "$RUN_TIMEOUT" env $penv $envA $cmdA $PROTO --csv "${tmp}_$la" > "${out}_$la.log" 2>&1 &
    local pid_a=$!
    # shellcheck disable=SC2086
    timeout -k 30 "$RUN_TIMEOUT" env $penv $envB $cmdB $PROTO --csv "${tmp}_$lb" > "${out}_$lb.log" 2>&1 &
    local pid_b=$!

    # `|| rc=$?` instead of letting the failure propagate, because `set -e`
    # would otherwise abort before we could report WHICH of the two failed.
    wait "$pid_a" || rc_a=$?
    wait "$pid_b" || rc_b=$?
    power_stop
    if [ "$rc_a" -ne 0 ] || [ "$rc_b" -ne 0 ]; then
        echo "[sweep] ABORT: run failed ($la rc=$rc_a, $lb rc=$rc_b)." >&2
        echo "        See ${out}_$la.log / ${out}_$lb.log" >&2
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
    printf '[sweep] protocol: %s\n' "$PROTO"
fi

if [ "$DRY" = 0 ]; then
    # Clear any daemon a previous sweep left behind. A dropped SSH connection
    # kills bash without running the EXIT trap, so a daemon can survive it; it
    # lives in our private pipe dir, so anything found there is ours to stop.
    # Doing it here, rather than relying on the restart when the first MPS job
    # starts, keeps the state known from the very first run.
    mps_stop

    check_clocks || exit 1
    # The instrument and the board state, recorded before anything runs (lib/guards.sh).
    check_binaries "$OUT/_manifests" "${SWEEP_ARGS[@]}" || exit 1
    floor_probe start
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

        selects "$exp" || continue
        [ -n "$ONLY" ] && ! [[ ${JOB_NAME[$i]} =~ $ONLY ]] && continue
      job_no=$((job_no + 1))

        # Kind and mps are needed here for the done-check and the dry-run
        # display; the command fields are only used by the dry run.
        IFS='|' read -r kind mps _eA _eB _cA _cB _ratio <<< "${JOB_SPEC[$i]}"

        if job_done "$out" && [ "$FORCE" = 0 ]; then
            printf '[sweep] %d/%d  skip %s (already done)\n' "$job_no" "$((SELECTED * RUNS))" "$name"
            continue
        fi

        # Start again from scratch, but say why: without this, a run appearing
        # twice in the log after a restart looks like a bug rather than the
        # resumability doing its job.
        if interrupted "$out"; then
            printf '[sweep] %d/%d  re-running %s (a previous attempt was interrupted)\n' "$job_no" "$((SELECTED * RUNS))" "$name"
        fi

        if [ "$DRY" = 1 ]; then
            printf '  %d/%d  %-26s [%s/%s]\n' "$job_no" "$((SELECTED * RUNS))" "$name" "$kind" "$mps"
            if [ "$kind" = solo ]; then
                printf '      A: %s %s --csv %s\n' "$_cA" "$PROTO" "$out"
            elif [ "$kind" = itf ]; then
                printf '      A: %s %s --csv %s   (measured)\n' "$_cA" "$PROTO" "$out"
                printf '      B: %s   (co-runner process, stopped when A exits)\n' "$_cB"
            else
                printf '      A: %s %s --csv %s_%s\n' "$_cA" "$PROTO" "$out" "$(net_label "$_cA" mobilenet)"
                printf '      B: %s %s --csv %s_%s\n' "$_cB" "$PROTO" "$out" "$(net_label "$_cB" resnet)"
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
            [ "$CPU_GUARD" = 1 ] && { require_quiet_cpu || exit 1; }
        fi

        # Bring the MPS daemon into the mode this job needs (off/on/static).
        mps_ensure "$mps" || exit 1

        printf '[sweep] %d/%d  %s\n' "$job_no" "$((SELECTED * RUNS))" "$name"
        do_job "${JOB_SPEC[$i]}" "$out" || exit 1

        [ "$mps" = static ] && mps_remove_partitions
    done
done

rm -f "$OUT/_sentinel_latencies.csv" "$OUT/_sentinel_stats.csv"
[ "$DRY" = 0 ] && { mps_stop; floor_probe end; }

echo
if [ "$DRY" = 1 ]; then
    echo "[sweep] dry run: nothing was executed."
elif [ "$SMOKE" = 1 ]; then
    echo "[sweep] smoke test complete — the plumbing works."
    echo "        Inspect a log, then run: ./measure/sweep.sh"
else
    echo "[sweep] done. $((SELECTED * RUNS)) of $((SELECTED * RUNS)) runs. Aggregate with:  python3 $ROOT/analysis/aggregate.py --out $OUT"
fi
