#!/bin/bash
# One configuration of the additivity experiment under MPS static partitions, five runs.
#
# The sweep's job spec has room for two clients; this block needs three, because the question is
# what one target at 8 SMs takes from two co-runners at 4 SMs each. So the daemon and the
# partitions come from utility/lib/mps.sh, the same code the sweep uses, and the processes are
# launched and collected here.
#
#   utility/mps-additivity-block.sh [runs] <output name> <model>:<percent>:<label> [<model>:<percent>:<label> ...]
#
#   utility/mps-additivity-block.sh 5 mobilenet+retinanet+inception \
#       mobilenetv2-12:50:mobilenet retinanet-9:25:retinanet inception_v3-onnx-float:25:inception
#
# The first spec is the target. Each client is one process confined to its own partition, and each
# writes its own stats file, which is renamed to the r<k>_<label>_stats.csv layout the aggregate
# already reads. Results go to $MPS_BLOCK_OUT (default $DATA/bench/<family>).
set -u

DATA="${DATA:-/mnt/nvme/f.scarfato}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
FAMILY="${MPS_BLOCK_FAMILY:-interference-additivity-mps}"
OUT="${MPS_BLOCK_OUT:-$DATA/bench/$FAMILY}"
DATASET="$DATA/dataset/imagenet-1k"
MODELS="$DATA/models"
TMP="${TMPCSV:-/dev/shm/scarfato-csv}"

STATIC_CHUNKS="${STATIC_CHUNKS:-8}"
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off
# shellcheck source=lib/mps.sh
. "$HERE/lib/mps.sh"

runs="${1:-5}"; shift
outname="$1"; shift
[ $# -ge 1 ] || { echo "usage: $0 [runs] <output name> <model>:<percent>:<label> ..." >&2; exit 2; }

pcts=(); specs=(); labels=()
for spec in "$@"; do
    model="${spec%%:*}"; rest="${spec#*:}"
    pct="${rest%%:*}"; label="${rest##*:}"
    [ "$label" = "$rest" ] && label="$model"
    pcts+=("$pct")
    specs+=("$MODELS/$model/engine/$model.engine:100:$label")   # 100 %: the partition confines it
    labels+=("$label")
done

ratio="$(IFS=:; echo "${pcts[*]}")"
dest="$OUT/$outname"
mkdir -p "$dest" "$TMP"

cleanup() { mps_stop; }
trap cleanup EXIT

mps_stop
mps_start static || exit 1
mps_create_partitions "$ratio" || exit 1
[ "${#PART_LIST[@]}" -ge "${#specs[@]}" ] || { echo "  too few partitions for ${#specs[@]} clients"; exit 1; }

echo "== $outname: ${#specs[@]} clients, partitions ${PART_LIST[*]} =="
for k in $(seq 1 "$runs"); do
    pids=()
    for i in "${!specs[@]}"; do
        part="${PART_LIST[$i]}"
        label="${labels[$i]}"
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="$part" \
            "$ROOT/runner" --submit-threads 1 --dataset "$DATASET" --allocation whole \
            --net "${specs[$i]}" --fixed 32 \
            --min-inferences "${MPS_BLOCK_INF:-5000}" --min-duration "${MPS_BLOCK_DUR:-30}" \
            --csv "$TMP/${outname}-r${k}-${label}" > "$dest/r${k}-${label}.log" 2>&1 &
        pids+=($!)
    done
    rc=0
    for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
    for label in "${labels[@]}"; do
        for kind in stats latencies; do
            src="$TMP/${outname}-r${k}-${label}_${kind}.csv"
            [ -f "$src" ] && mv -f "$src" "$dest/r$k_${label}_$kind.csv"
        done
        rm -f "$TMP/${outname}-r${k}-${label}".*
    done
    printf '  run %s/%s done (rc=%s)  files: %s\n' "$k" "$runs" "$rc" \
        "$(ls "$dest" | grep -c "^r$k" )"
    [ "$rc" = 0 ] || exit 1
done
mps_remove_partitions || true
echo "  results in $dest"
