#!/bin/bash
# The three kernel traces behind the per-inference decomposition, taken and exported.
#
# Kernel tracing needs no root (the SoC counters do, and live in soc-traces.sh). These are
# diagnostics and not campaign measurements: profiling perturbs the run, so they are quoted for
# the shape of the split, which reproduces the campaign, and not for absolute tails.
#
#   utility/take-traces.sh [output directory]
#
# Then reduce them with: python3 utility/itf-trace.py <that directory>
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NSYS=/usr/local/bin/nsys
DATA="${DATA:-/mnt/nvme/f.scarfato}"
OUT="${1:-$DATA/scratch/traces/trc}"
DATASET="$DATA/dataset/imagenet-1k"
MOB="$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine"

mkdir -p "$OUT" /dev/shm/scarfato-csv
cd "$ROOT"

run() {   # run <name> [extra runner arguments...]
    local name="$1"; shift
    echo "=== $name ==="
    "$NSYS" profile --force-overwrite=true --trace=cuda --sample=none --cpuctxsw=none \
        --output="$OUT/$name" \
        ./runner --submit-threads 1 --dataset "$DATASET" --allocation greenctx \
        --net "$MOB:50:mobilenet" "$@" \
        --fixed 32 --min-inferences 1000 --min-duration 2 \
        --csv "/dev/shm/scarfato-csv/trc-$name" > "$OUT/$name.log" 2>&1
    echo "  run: exit $?"
    "$NSYS" export --type sqlite --force-overwrite=true \
        --output="$OUT/$name.sqlite" "$OUT/$name.nsys-rep" > "$OUT/$name.export.log" 2>&1
    echo "  export: exit $?"
    rm -f "/dev/shm/scarfato-csv/trc-$name"*
}

run alone
run mem4     --interferer mem_saxpy:25
run compute4 --interferer compute_gemm:25

echo
echo "traces in $OUT"
echo "reduce with: python3 $ROOT/utility/itf-trace.py $OUT"
