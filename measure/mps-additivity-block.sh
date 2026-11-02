#!/bin/bash
# The additivity experiment under MPS static partitions.
#
# Two modes:
#
#   one configuration, up to three clients, any shares:
#     measure/mps-additivity-block.sh 5 <name> <engine>:<percent>:<label> [...]
#
#   a whole block: the target alone, plus each co-runner, plus both, which is the four arms the
#   question needs and the same names the green-context family uses:
#     measure/mps-additivity-block.sh --block 5 <target engine>:<label> <A engine>:<label> <B engine>:<label>
#
# Three clients need three partitions; the sweep's job spec has room for two, so the daemon and the
# partitions come from measure/lib/mps.sh - the same library the sweep uses - and the processes are
# launched here. Each client is one process confined to its own partition, and each writes its own
# stats file, renamed to the r<k>_<label>_stats.csv layout the aggregate reads.
#
# For the block, one partition set of 50/25/25 serves all four arms: the alone arm uses the 50 %
# partition, the pairs use it with one of the 25 % ones, and the triple uses all three.
#
# Does not need root. Results go to $MPS_BLOCK_OUT (default $DATA/bench/<family>).
set -u

DATA="${DATA:-/mnt/nvme/f.scarfato}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
FAMILY="${MPS_BLOCK_FAMILY:-interference-additivity-mps}"
OUT="${MPS_BLOCK_OUT:-$DATA/bench/$FAMILY}"
DATASET="$DATA/dataset/imagenet-1k"
TMP="${TMPCSV:-/dev/shm/scarfato-csv}"
RUNS="${MPS_BLOCK_RUNS:-5}"
INFERENCES="${MPS_BLOCK_INF:-5000}"
DURATION="${MPS_BLOCK_DUR:-30}"

STATIC_CHUNKS="${STATIC_CHUNKS:-8}"
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off
# shellcheck source=lib/mps.sh
. "$HERE/lib/mps.sh"

MODE=one
if [ "${1:-}" = "--block" ]; then MODE=block; shift; fi
if [ "$#" -ge 1 ] && [ "$1" -gt 0 ] 2>/dev/null; then RUNS="$1"; shift; fi

# parse "engine:percent:label" or, for the block, "engine:label"
parse_spec() {   # $1 = spec, $2 = default percent ; sets SPEC_PATH SPEC_PCT SPEC_LABEL
    SPEC_PATH="${1%%:*}"
    local rest="${1#*:}"
    if [ "$MODE" = block ]; then
        SPEC_PCT="$2"; SPEC_LABEL="$rest"
    else
        SPEC_PCT="${rest%%:*}"; SPEC_LABEL="${rest##*:}"
        [ "$SPEC_LABEL" = "$rest" ] && SPEC_LABEL="${SPEC_PATH##*/}"
    fi
}

case "$MODE" in
one)
    [ $# -ge 1 ] || { echo "usage: $0 [runs] <name> <engine>:<percent>:<label> ..." >&2; exit 2; }
    NAME="$1"; shift
    CLIENTS=()
    for spec in "$@"; do
        parse_spec "$spec" 100
        CLIENTS+=("$SPEC_PATH:$SPEC_PCT:$SPEC_LABEL")
    done
    ;;
block)
    [ $# -eq 3 ] || { echo "usage: $0 --block [runs] <target>:<label> <A>:<label> <B>:<label>" >&2; exit 2; }
    parse_spec "$1" 50; TARGET_PATH="$SPEC_PATH"; TARGET_LABEL="$SPEC_LABEL"
    parse_spec "$2" 25; A_PATH="$SPEC_PATH"; A_LABEL="$SPEC_LABEL"
    parse_spec "$3" 25; B_PATH="$SPEC_PATH"; B_LABEL="$SPEC_LABEL"
    ;;
esac

mkdir -p "$OUT" "$TMP"
cleanup() { mps_stop; }
trap cleanup EXIT

mps_stop
mps_start static || exit 1
# one partition set serves every arm of the block, and the single-configuration mode asks for its
# own shares
if [ "$MODE" = block ]; then
    mps_create_partitions "50:25:25" || exit 1
else
    shares=""
    for c in "${CLIENTS[@]}"; do shares="$shares${shares:+:}$(echo "$c" | cut -d: -f2)"; done
    mps_create_partitions "$shares" || exit 1
fi

run_arm() {   # $1 = output name, then "<partition index>|<engine path>|<label>" per client
    local name="$1"; shift
    local dest="$OUT/$name" k rc part spec path label pids pid
    mkdir -p "$dest"
    printf '== %s: %s clients ==\n' "$name" "$#"
    for k in $(seq 1 "$RUNS"); do
        pids=()
        for spec in "$@"; do
            part="${PART_LIST[${spec%%|*}]}"
            path="$(echo "$spec" | cut -d'|' -f2)"
            label="$(echo "$spec" | cut -d'|' -f3)"
            env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="$part" \
                "$ROOT/runner" --submit-threads 1 --dataset "$DATASET" --allocation whole \
                --net "$path:100:$label" --fixed 32 \
                --min-inferences "$INFERENCES" --min-duration "$DURATION" \
                --csv "$TMP/${name}-r${k}-${label}" > "$dest/r${k}-${label}.log" 2>&1 &
            pids+=($!)
        done
        rc=0
        for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
        for spec in "$@"; do
            label="$(echo "$spec" | cut -d'|' -f3)"
            for kind in stats latencies; do
                [ -f "$TMP/${name}-r${k}-${label}_${kind}.csv" ] && \
                    mv -f "$TMP/${name}-r${k}-${label}_${kind}.csv" "$dest/r${k}_${label}_${kind}.csv"
            done
            rm -f "$TMP/${name}-r${k}-${label}".*
        done
        printf '   run %s/%s rc=%s\n' "$k" "$RUNS" "$rc"
        [ "$rc" = 0 ] || return 1
    done
}

if [ "$MODE" = block ]; then
    run_arm "${TARGET_LABEL}-alone" "0|$TARGET_PATH|$TARGET_LABEL" || exit 1
    run_arm "${TARGET_LABEL}+${A_LABEL}" "0|$TARGET_PATH|$TARGET_LABEL" "1|$A_PATH|$A_LABEL" || exit 1
    run_arm "${TARGET_LABEL}+${B_LABEL}" "0|$TARGET_PATH|$TARGET_LABEL" "2|$B_PATH|$B_LABEL" || exit 1
    run_arm "${TARGET_LABEL}+${A_LABEL}+${B_LABEL}" \
        "0|$TARGET_PATH|$TARGET_LABEL" "1|$A_PATH|$A_LABEL" "2|$B_PATH|$B_LABEL" || exit 1
else
    args=()
    i=0
    for c in "${CLIENTS[@]}"; do
        args+=("$i|${c%%:*}|$(echo "$c" | cut -d: -f3)")
        i=$((i + 1))
    done
    run_arm "$NAME" "${args[@]}" || exit 1
fi

mps_remove_partitions || true
echo "results in $OUT"
