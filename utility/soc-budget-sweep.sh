#!/bin/bash
# SoC-budget sweep: profile one configuration with the DRAM counters.
#
# The SoC counters are device-wide and cannot attribute to a client, so the way to obtain one
# network's pressure at one SM budget is to run only that network at that budget.
#
#   utility/soc-budget-sweep.sh greenctx 25 /path/to/mobilenetv2-12.engine:25
#   utility/soc-budget-sweep.sh greenctx 50 /path/to/mobilenetv2-12.engine:50 /path/to/resnet50.engine:50
#   utility/soc-budget-sweep.sh whole    0 /path/to/mobilenetv2-12.engine:100
#
#   mechanism  whole | greenctx
#   budget     the largest share in the configuration, for the record
#   spec       <engine path>:<percent>[:<label>], the share of the device that client may use
#
# A configuration whose trace already exists is skipped, so adding a point to the curve costs only
# that point. SOC_FORCE=1 re-measures.
#
# There is no MPS static path. nsys cannot instrument a client of the MPS daemon - the client's
# context belongs to the daemon, so --trace=cuda waits for activity it will never see in that
# process - and --soc-metrics cannot be combined with --trace=none to avoid that. The mechanism
# check for MPS static therefore rests on the latency and throughput agreement, which the campaign
# has, rather than on the counters.
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

mechanism="${1:-}"; shift || true
budget="${1:-}"; shift || true
case "$mechanism" in
    whole|greenctx) ;;
    *) echo "usage: $0 <whole|greenctx> <budget-percent> <engine>:<percent>[:<label>]..." >&2; exit 2 ;;
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

if [ -f "$OUT/$name.nsys-rep" ] && [ "${SOC_FORCE:-0}" != "1" ]; then
    echo "== $name: already traced, skipping (SOC_FORCE=1 to re-measure)"
    exit 0
fi

runner_args=(--submit-threads 1 --dataset "$DATASET" --fixed 32
             --min-inferences "$INFERENCES" --min-duration "$SECONDS_MEASURED")
for n in "${nets[@]}"; do runner_args+=(--net "$n"); done
[ "$mechanism" = greenctx ] && runner_args+=(--allocation greenctx) || runner_args+=(--allocation whole)

echo "== $name (mechanism $mechanism, budget $budget %) =="
echo "   $NSYS profile --soc-metrics=true --output=$OUT/$name.nsys-rep $ROOT/runner ${runner_args[*]}"
"$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
    --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
    --output="$OUT/$name" \
    "$ROOT/runner" "${runner_args[@]}" \
    --csv "/dev/shm/$name" > "$OUT/$name.log" 2>&1
rc=$?
echo "   exit $rc"
rm -f "/dev/shm/$name"*
grep -E "^\[info\] (network|partition plan|allocation)" "$OUT/$name.log" 2>/dev/null | sed 's/^/   /'
ls -l "$OUT/$name.nsys-rep" 2>/dev/null | awk '{printf "   %s  %.1f MB\n", $9, $5/1048576}'
