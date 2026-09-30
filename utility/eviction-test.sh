#!/usr/bin/env bash
# eviction-test.sh - does a co-runner that moves no DRAM make the target re-read from DRAM?
#
#   sudo ./utility/eviction-test.sh
#
# The question left open. MobileNet at 8 SMs is 14 % slower in its median when compute_gemm runs
# beside it on 4 SMs, and compute_gemm's operands are L2-resident, so it moves almost no DRAM. Two
# explanations remain:
#
#   eviction     the co-runner's lines push the target's out, so the target re-reads from DRAM.
#                Its own DRAM demand therefore rises, and the device's DRAM rate does NOT fall in
#                proportion as the target slows down.
#   arbitration  the lines stay and each access merely waits longer. The device's DRAM rate falls
#                with the target's rate, because the same amount of data is fetched, just later.
#
# The test is the ratio of the two rates against the ratio of the two inference rates:
#
#   same bytes per inference   ->  DRAM rate falls with the inference rate   ->  arbitration
#   more bytes per inference   ->  DRAM rate holds up while the rate drops   ->  eviction
#
# The runner writes its statistics to a prefix given with --csv, and it must: the default prefix
# drops bench_stats.csv into the working directory, and a root-owned file there cannot be reopened
# by anything else, which aborted the first attempt half a second in with "cannot open
# bench_latencies.csv". The prefix has to be in /dev/shm, not on the disk: a prefix on the NVMe
# makes the runner write a line per inference, which took MobileNet at 8 SMs from 1480 inferences
# per second to 434 and its p99 from 0.68 ms to 3.03. Measurement comes first, the copy afterwards.
#
# The counters cannot attribute to a client, but they do not need to here: the co-runner moves no
# DRAM, so the device's DRAM is the target's, and both runs use the same partition and the same
# protocol, so the two rates are directly comparable. The inference counts come from the runner's own
# statistics, which are copied out before anything deletes them, and the medians from the aggregate
# cross-check the ratio independently.
#
# ncu cannot do this: it serialises kernels, so during the profiled kernel the co-runner is not
# running. The evidence is in the earlier run - under mem_saxpy, a co-runner streaming 55 GB/s, the
# device-wide DRAM reading during the target's kernels was 4-15 %, the same as with no co-runner.
set -u

NSYS=${NSYS:-/usr/local/bin/nsys}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/scratch/eviction}

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

# The same preconditions every sweep checks, in the same library. Without them this measured a
# co-tenant benchmark: a plain run gave MobileNet at 8 SMs 2.27 ms where the corpus says 0.675,
# a factor of 3.4, because l.demartino's benchmark held the GPU at the time.
export OUT RUNNER MOB DS
RUNNER="$PWD/runner"
MOB="$MODEL"
DS="$DATA/dataset/imagenet-1k"
SENTINEL_REF_MS=${SENTINEL_REF_MS:-0.535}
SENTINEL_TOL=${SENTINEL_TOL:-0.15}
SENTINEL_INF=${SENTINEL_INF:-500}
source utility/lib/guards.sh
check_clocks || exit 1

BASE=(--submit-threads 1 --dataset "$DATA/dataset/imagenet-1k" --fixed 32
      --min-inferences 5000 --min-duration 30)

run_one() {   # $1 = label; rest are runner arguments
    local label=$1; shift
    local marker="$OUT/.marker"
    touch "$marker"
    # refuse to measure a board that is shared, rather than measuring it wrongly
    require_quiet_gpu || exit 1
    [ "${CPU_GUARD:-0}" = 1 ] && { require_quiet_cpu || exit 1; }
    echo
    echo "== $label =="
    echo "   $NSYS profile --soc-metrics … ./runner ${BASE[*]} $*"
    "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
        --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
        --output "$OUT/$label.nsys-rep" \
        ./runner "${BASE[@]}" --csv "/dev/shm/$label" "$@" > "$OUT/$label.log" 2>&1
    local rc=$?
    echo "   nsys exit $rc"
    # what the runner did, before the next sweep deletes /dev/shm
    local copied=0
    for f in /dev/shm/*_stats.csv /dev/shm/*_latencies.csv; do
        [ -e "$f" ] || continue
        [ "$f" -nt "$marker" ] || continue
        cp -f "$f" "$OUT/$label-$(basename "$f")" && copied=$((copied + 1))
    done
    echo "   runner statistics copied: $copied file(s)"
    grep -oE "[0-9]+ (inferences|inf/s)|median +[0-9.]+ ms" "$OUT/$label.log" 2>/dev/null \
        | tail -3 | sed 's/^/     /' || true
}

# the reference and the treatment: same partition, same protocol, the co-runner the only difference
run_one target8-alone   --allocation greenctx --net "$MODEL:50:mobilenet"
run_one target8-icompute --allocation greenctx --net "$MODEL:50:mobilenet" --interferer compute_gemm:25

# The co-runner on its own is not measurable: the runner refuses an interferer with no network
# ("no network given: pass <engine> or --net <engine>:<pct>"). Its own DRAM demand is checked
# instead by utility/ncu-who-moves-dram.sh, which profiles the pair and groups the kernels by
# stream, and for that purpose serialisation is harmless: measuring one client's own traffic does
# not need the other running.

echo
echo "  traces and statistics in $OUT"
echo "  the verdict needs the two DRAM series and the two inference counts, both of which are now here"
