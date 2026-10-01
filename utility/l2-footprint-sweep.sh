#!/usr/bin/env bash
# l2-footprint-sweep.sh - does the co-runner's cache footprint, and not its DRAM traffic, cost the target?
#
#   sudo ./utility/l2-footprint-sweep.sh
#
# The memory subsection leaves a floor: about ten points of median penalty are present with a
# co-runner whose operands are L2-resident, on top of a term that follows its DRAM demand. Two
# candidates remain, and this sweep separates them by moving only the co-runner's operand footprint
# while its SM count, its kind and the protocol stay fixed:
#
#   the footprint is what matters      the penalty rises from A1 to E with no DRAM involved
#   the cache holds nothing to do       the penalty is flat until F, where the co-runner's own
#                                       DRAM traffic appears
#   neither                             the penalty is already at the floor in A1, where the
#                                       operands fit in L1 and almost nothing reaches the L2
#   the partition itself                A0, an idle participant, already costs something
#
# A0 and F are the poles. A0 provisions a partition and runs nothing in it, which nobody has
# measured. F is the only arm whose footprint exceeds the 4 MB cache, so it necessarily adds the
# co-runner's own DRAM traffic: it is the streaming pole, and its penalty should be read together
# with the memory kind's arms rather than as a point on the same axis.
#
# One traced run per arm gives everything: the target's median and inference count from the runner's
# own statistics, the device's DRAM integral from the counters, and the co-runner's achieved rate
# from its own report. The counter nearest this question, the L2's own traffic, is not available
# during contention - Nsight Compute quiesces the other clients around every kernel it measures -
# so the last step here uses it only to validate that the arms do what they claim: the L2 hit rate
# of the co-runner's own kernel should stay high through E and fall at F.
set -u

NSYS=${NSYS:-/usr/local/bin/nsys}
NCU=${NCU:-/usr/local/cuda/bin/ncu}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/scratch/l2-footprint}

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

# the same guards as every other measurement: the clocks, and a board nobody else is using
export OUT RUNNER MOB DS
RUNNER="$PWD/runner"
MOB="$MODEL"
DS="$DATA/dataset/imagenet-1k"
source utility/lib/guards.sh
check_clocks || exit 1

BASE=(--submit-threads 1 --dataset "$DS" --fixed 32 --min-inferences 5000 --min-duration 30)
METRICS="lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_bytes.sum,l1tex__t_sector_hit_rate.pct"

# label  interferer spec        operand footprint
ARMS=(
  "A0-idle       idle:25"
  "A1-l1         compute_gemm:25:64"
  "B-1mb         compute_gemm:25:288"
  "C-2mb         compute_gemm:25:400"
  "D-3mb         compute_gemm:25:512"
  "E-4mb         compute_gemm:25:576"
  "F-12mb        compute_gemm:25:1024"
)

run_arm() {
    local label=$1 spec=$2
    local kind=${spec%%:*} rest=${spec#*:}
    local pct=${rest%%:*} size=${rest#*:}
    local args=(--interferer "$kind:$pct")
    [ "$size" != "$rest" ] && args+=(--interferer-size "$size")

    require_quiet_gpu || exit 1
    echo
    echo "== $label : $spec =="
    "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
        --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
        --output "$OUT/$label.nsys-rep" \
        ./runner "${BASE[@]}" --csv "/dev/shm/$label" \
        --allocation greenctx --net "$MODEL:50:mobilenet" "${args[@]}" \
        > "$OUT/$label.log" 2>&1
    # a single-network run writes <prefix>_stats.csv unlabelled, a multi-network one appends the
    # network's label; take whichever is there
    for f in /dev/shm/${label}_stats.csv /dev/shm/${label}_latencies.csv /dev/shm/${label}_*_stats.csv; do
        [ -e "$f" ] && cp -f "$f" "$OUT/$(basename "$f")"
    done
    rm -f "/dev/shm/${label}"*
    # the record of the arm: the median and the co-runner's own rate, both from the runner
    awk -F, '$1=="gpu_ms"{printf "   target median %s ms over %s inferences\n", $4, $2}' \
        "$OUT/${label}_stats.csv" 2>/dev/null
    grep -E "working set|GEMM side|units:|arithmetic:|GB/s" "$OUT/$label.log" 2>/dev/null \
        | sed 's/^/   /' | head -5
}

for arm in "${ARMS[@]}"; do run_arm $arm; done

# the validation: the target's own kernels, profiled in isolation, on the same arms. This is not
# contention - the profiler quiesces the co-runner - it checks the arms do what they claim: the L2
# hit rate should be high while the operands fit in the cache and fall once they do not.
echo
echo "== validating the footprints with the profiler =="
printf "  %-14s %s\n" "arm" "L2 hit rate of the target's kernels"
for arm in "${ARMS[@]}"; do
    set -- $arm
    label=$1; spec=$2
    case "$spec" in idle:*) continue;; esac
    rest=${spec#*:}; size=${rest#*:}
    [ "$size" = "$rest" ] && size=512
    OUT="$OUT" ./utility/ncu.sh target "$size" >/dev/null 2>&1
    [ -f "$OUT/target-$size.csv" ] && cp -f "$OUT/target-$size.csv" "$OUT/$label-ncu.csv"
    awk -F'","' -v L="$label" \
        '/lts__t_sector_hit_rate/{printf "  %-14s %s %%\n", L, $NF; exit}' \
        "$OUT/$label-ncu.csv" 2>/dev/null
done

echo
echo "  traces, statistics and profiler output in $OUT"
echo "  the reading: python3 utility/l2-footprint-verdict.py"
