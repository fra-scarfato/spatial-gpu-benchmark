#!/bin/bash
# SoC-budget sweep: profile one configuration with the DRAM counters.
#
# The SoC counters are device-wide and cannot attribute to a client, so the way to obtain one
# network's pressure at one SM budget is to run only that network at that budget. This script does
# that under any of the three mechanisms, so the pressure curve can be measured twice - once under a
# green context and once under an MPS static partition - to check that the mechanism does not change
# it. A partition divides SMs and leaves DRAM and L2 shared, so it should not.
#
#   utility/soc-budget-sweep.sh greenctx   25 /path/to/mobilenetv2-12.engine:25
#   utility/soc-budget-sweep.sh mps-static 50 /path/to/mobilenetv2-12.engine:50 /path/to/retinanet-9.engine:25
#   utility/soc-budget-sweep.sh whole       0 /path/to/mobilenetv2-12.engine:100
#
#   mechanism  whole | greenctx | mps-static
#   budget     the largest share in the configuration, used to size the MPS partitions
#   spec       <engine path>:<percent>[:<label>], the share of the device that client may use
#
# Needs root for the counters, so run it with sudo. Profiling perturbs, so these are diagnostics:
# the window is shorter than the campaign protocol's and the write-up says so.
set -u

DATA="${DATA:-/mnt/nvme/f.scarfato}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${SOC_OUT:-$DATA/scratch/soc-budget}"
DATASET="$DATA/dataset/imagenet-1k"
NSYS="${NSYS:-/usr/local/bin/nsys}"
SECONDS_MEASURED="${SECONDS_MEASURED:-20}"
INFERENCES="${INFERENCES:-100}"

STATIC_CHUNKS="${STATIC_CHUNKS:-8}"
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off
[ "${1:-}" = "mps-static" ] && { . "$HERE/lib/mps.sh"; }

mechanism="${1:-}"; shift || true
budget="${1:-}"; shift || true
case "$mechanism" in
    whole|greenctx|mps-static) ;;
    *) echo "usage: $0 <whole|greenctx|mps-static> <budget-percent> <engine>:<percent>[:<label>]..." >&2; exit 2 ;;
esac
[ $# -ge 1 ] || { echo "usage: $0 <mechanism> <budget> <engine>:<percent>[:<label>]..." >&2; exit 2; }

# The spec carries a full engine path, which must never reach a file name: the label is taken from
# it, and everything outside a safe alphabet is folded to a dash.
safe() { printf '%s' "$1" | tr -c 'A-Za-z0-9._+-' '-' | sed 's/-\{2,\}/-/g; s/-$//'; }

name=""
nets=()
for spec in "$@"; do
    path="${spec%%:*}"
    rest="${spec#*:}"
    pct="${rest%%:*}"
    label="${rest#*:}"
    [ "$label" = "$rest" ] && label="$(basename "$path")"
    label="$(safe "${label%.engine}")"
    [ -f "$path" ] || { echo "  no engine at $path" >&2; exit 2; }
    nets+=("$path:$pct:$label")          # the share goes in the spec, not in --sm-percent
    name="${name}${name:+_}$(safe "$label")-$pct"
done
name="soc-$(safe "$mechanism")-$name"
mkdir -p "$OUT"

cleanup() { [ "$mechanism" = mps-static ] && mps_stop; }
trap cleanup EXIT

runner_args=(--submit-threads 1 --dataset "$DATASET" --fixed 32
             --min-inferences "$INFERENCES" --min-duration "$SECONDS_MEASURED")
if [ "$mechanism" = mps-static ]; then
    # under a static partition the runner's own share is meaningless: the partition confines the
    # client, which is why every MPS arm in the campaign passes 100 here
    for spec in "$@"; do
        p="${spec%%:*}"
        l="$(safe "$(basename "$p")")"
        runner_args+=(--net "$p:100:${l%.engine}")
    done
else
    for n in "${nets[@]}"; do runner_args+=(--net "$n"); done
fi

if [ "$mechanism" = mps-static ]; then
    # a static set covers the device in two parts, so a solo client needs the colon ratio, which
    # places its shares and leaves the rest unassigned
    ratio="$budget"
    [ $# -eq 1 ] && ratio="$budget:$((100 - budget))"
    mps_stop
    mps_start static || exit 1
    mps_create_partitions "$ratio" || exit 1
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE"
    export CUDA_MPS_SM_PARTITION="$PART_A"
    runner_args+=(--allocation whole)
elif [ "$mechanism" = greenctx ]; then
    runner_args+=(--allocation greenctx)
else
    runner_args+=(--allocation whole)
fi

echo "== $name (mechanism $mechanism, budget $budget %) =="
# The SoC counters are device-wide, but --soc-metrics cannot be combined with --trace=none, and
# --trace=cuda cannot instrument a client of the MPS daemon: its context belongs to the daemon, so
# the profiler waits for activity it will never see in that process, which is what blocked. So under
# MPS the client runs outside the profiler and the profiler samples the device while it runs - the
# counters do not care whose kernels they are.
if [ "$mechanism" = mps-static ]; then
    echo "   $ROOT/runner ${runner_args[*]} &   (client, not traced)"
    "$ROOT/runner" "${runner_args[@]}" --csv "/dev/shm/$name" > "$OUT/$name.log" 2>&1 &
    client=$!
    sleep 5                                   # let it load its engine and reach the timed phase
    echo "   $NSYS profile ... --output=$OUT/$name sleep $SECONDS_MEASURED   (device sampled)"
    "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
        --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
        --output="$OUT/$name" sleep "$SECONDS_MEASURED" > "$OUT/$name.nsys.log" 2>&1
    rc=$?
    wait "$client" || true
else
    echo "   $NSYS profile --soc-metrics=true --output=$OUT/$name.nsys-rep $ROOT/runner ${runner_args[*]}"
    "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
        --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
        --output="$OUT/$name" \
        "$ROOT/runner" "${runner_args[@]}" \
        --csv "/dev/shm/$name" > "$OUT/$name.log" 2>&1
    rc=$?
fi
echo "   exit $rc"
rm -f "/dev/shm/$name"*
grep -E "^\[info\] (network|partition plan|allocation)" "$OUT/$name.log" 2>/dev/null | sed 's/^/   /'
ls -l "$OUT/$name.nsys-rep" 2>/dev/null | awk '{printf "   %s  %.1f MB\n", $9, $5/1048576}'
