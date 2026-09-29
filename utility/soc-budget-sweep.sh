#!/bin/bash
# SoC-budget sweep: profile one configuration with the DRAM counters.
#
# The SoC counters are device-wide and cannot attribute to a client, so the way to obtain one
# network's pressure at one SM budget is to run only that network at that budget. This script does
# that under any of the three mechanisms, so the pressure curve can be measured twice - once under
# a green context and once under an MPS static partition - to check that the mechanism does not
# change it. A partition divides SMs and leaves DRAM and L2 shared, so it should not.
#
#   utility/soc-budget-sweep.sh greenctx   4 mobilenetv2-12:25
#   utility/soc-budget-sweep.sh mps-static 5 mobilenetv2-12:50 retinanet-9:25
#   utility/soc-budget-sweep.sh whole      0 mobilenetv2-12:100
#
#   mechanism  whole | greenctx | mps-static
#   budget     the largest share in the configuration, used to size the MPS partitions
#   net spec   <model>:<percent>[:<label>], the percent being the green-context share
#
# Needs root for the counters, so run it with sudo. Profiling perturbs, so these are diagnostics:
# the window is shorter than the campaign protocol's and the write-up says so.
set -u

DATA="${DATA:-/mnt/nvme/f.scarfato}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
OUT="${SOC_OUT:-$DATA/scratch/soc-budget}"
MODELS="$DATA/models"
DATASET="$DATA/dataset/imagenet-1k"
NSYS="${NSYS:-/usr/local/bin/nsys}"
SECONDS_MEASURED="${SECONDS_MEASURED:-20}"
INFERENCES="${INFERENCES:-100}"

# what the MPS static arms need, taken from the sweep's own library so the two cannot diverge
STATIC_CHUNKS="${STATIC_CHUNKS:-8}"
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off
if [ "$1" = "mps-static" ]; then
    # shellcheck source=lib/mps.sh
    . "$HERE/lib/mps.sh"
fi

mechanism="$1"; shift
budget="$1"; shift
[ $# -ge 1 ] || { echo "usage: $0 <whole|greenctx|mps-static> <budget-percent> <model>:<percent>[:<label>]..." >&2; exit 2; }

name="$mechanism-$budget"
nets=(); labels=()
for spec in "$@"; do
    model="${spec%%:*}"
    rest="${spec#*:}"
    pct="${rest%%:*}"
    label="${rest#*:}"
    [ "$label" = "$pct" ] && label="$model"
    nets+=("$MODELS/$model/engine/$model.engine:$pct:$label")
    name="$name-$(echo "$label" | tr -d '_')$pct"
done
mkdir -p "$OUT"

cleanup() { [ "$mechanism" = mps-static ] && mps_stop; }
trap cleanup EXIT

runner_args=(--submit-threads 1 --dataset "$DATASET" --fixed 32
             --min-inferences "$INFERENCES" --min-duration "$SECONDS_MEASURED")
# the networks go in every mechanism: under MPS static the allocation comes from the daemon's
# partition rather than from a green context, but the participants are declared the same way
for i in "${!nets[@]}"; do
    runner_args+=(--net "${nets[$i]}")
done
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

echo "== $name ($mechanism, budget $budget %) =="
"$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
    --soc-metrics=true --soc-metrics-set=t234 \
    --output="$OUT/$name" \
    "$ROOT/runner" "${runner_args[@]}" \
    --csv "/dev/shm/soc-$name" > "$OUT/$name.log" 2>&1
rc=$?
echo "  nsys exit $rc"
rm -f "/dev/shm/soc-$name"*
grep -E "^\[info\] (network|partition plan|allocation)" "$OUT/$name.log" | sed 's/^/    /'
ls -l "$OUT/$name.nsys-rep" 2>/dev/null | awk '{printf "    %s  %.1f MB\n", $9, $5/1048576}'
echo "  (the sqlite export and the analysis are done afterwards, not during the run)"
