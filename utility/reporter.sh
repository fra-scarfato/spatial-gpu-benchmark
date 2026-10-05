#!/usr/bin/env bash
# reporter.sh - the stress reporter beside a set of co-runners, under MPS static partitions.
#
#   utility/reporter.sh calibration          idle, then SAXPY on 8 SMs at levels 0, 0.1 ... 1
#   utility/reporter.sh networks             each network alone as the co-runner, on 4 SMs
#   utility/reporter.sh <set> --list         print the configurations, run nothing
#
# The reporter is RRZE-HPC's gpu-latency (utility/probes/gpu-latency): one thread chasing pointers
# through an L2-sized and a DRAM-sized buffer, in its own 2-SM partition, so it adds almost no
# traffic and reads the stress the co-runners put on the memory system as a latency. The co-runners
# are started first, the reporter 3 s later for 30 s; itfproc is then stopped, and a network
# co-runner finishes by itself, so its own statistics (occupancy, launch rate) are written too.
#
# Output: $DATA/bench/stress-reporter/<configuration>/r<k>_latency.txt, r<k>_c<i>.log and, for a
# network co-runner, r<k>_c<i>_stats.csv. A run whose latency file exists is skipped; every run is
# guarded like the campaign (clocks locked, nobody else on the GPU). MEASUREMENTS.md defines the
# quantities read from these files.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nvme/f.scarfato}"
OUTROOT="${OUTROOT:-$DATA/bench/stress-reporter}"
RUNS="${RUNS:-5}"
LEAD=3          # co-runners start this long before the reporter
DURATION=30     # the reporter's own measurement
DS="$DATA/dataset/imagenet-1k"
M="$DATA/models"
MOB="$M/mobilenetv2-12/engine/mobilenetv2-12.engine"
RES="$M/resnet50-v2-7/engine/resnet50-v2-7.engine"
INC="$M/inception_v3-onnx-float/engine/inception_v3.engine"
RET="$M/retinanet-9/engine/retinanet-9.engine"
RUNNER="$ROOT/runner"
ITF="$ROOT/utility/probes/itfproc"
REPORTER="$ROOT/utility/probes/gpu-latency/gpu-latency"
TMP=/dev/shm/fs-reporter

STATIC_CHUNKS=8
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off

# A network as a co-runner: its own process, the whole of its partition, and a run long enough to
# outlast the reporter, after which it exits and writes its statistics.
net() { echo "$RUNNER --submit-threads 1 --dataset $DS --fixed 32 --allocation whole --net $1:100:$2 --min-inferences 100 --min-duration $((LEAD + DURATION + 2))"; }

# The configurations of each set, one per line: name|co-runner shares|co-runner commands (;-separated).
# The reporter always takes one 2-SM chunk after the co-runners' partitions.
configs() {
    case "$1" in
    calibration)
        echo "idle||"
        local p
        for p in 000 010 020 030 040 050 060 070 080 090 100; do
            echo "saxpy-L$p|50|$ITF saxpy --level $((10#$p / 100)).$(printf '%02d' $((10#$p % 100)))"
        done
        ;;
    networks)
        echo "mobilenet|25|$(net "$MOB" mobilenet)"
        echo "resnet|25|$(net "$RES" resnet)"
        echo "inception|25|$(net "$INC" inception)"
        echo "retinanet|25|$(net "$RET" retinanet)"
        ;;
    *) return 1 ;;
    esac
}

set_name="${1:-}"
configs "$set_name" >/dev/null || { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
if [ "${2:-}" = "--list" ]; then
    configs "$set_name" | while IFS='|' read -r name shares cmds; do
        printf '%-14s shares %-6s %s\n' "$name" "${shares:-none}" "${cmds:-(no co-runner)}"
    done
    exit 0
fi

[ -x "$REPORTER" ] && [ -x "$ITF" ] || { echo "reporter.sh: build the probes first: make -C utility/probes" >&2; exit 1; }
export OUT="$TMP" MOB DS
mkdir -p "$TMP"
source "$ROOT/utility/lib/guards.sh"
source "$ROOT/utility/lib/mps.sh"
trap 'mps_stop; rm -rf "$TMP"' EXIT
check_clocks || exit 1

while IFS='|' read -r name shares cmds; do
    dest="$OUTROOT/$name"
    mkdir -p "$dest"
    IFS=';' read -r -a co <<< "$cmds"
    for k in $(seq 1 "$RUNS"); do
        [ -f "$dest/r${k}_latency.txt" ] && { echo "== $name r$k: done, skipped"; continue; }
        require_quiet_gpu || exit 1
        # A fresh daemon for every run: removing partitions is only best effort (lib/mps.sh), and
        # a partition left behind makes the next run's set fail to fit.
        mps_ensure static || exit 1
        mps_create_partitions "${shares}${shares:+:}12" || exit 1
        echo "== $name r$k: ${#co[@]} co-runner(s)"
        pids=(); stop=()
        for i in "${!co[@]}"; do
            csv=""
            case "${co[$i]}" in "$RUNNER"*) csv="--csv $TMP/r${k}_c$i" ;; esac
            # shellcheck disable=SC2086  # a command line, split into words on purpose
            env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="${PART_LIST[$i]}" \
                ${co[$i]} $csv > "$dest/r${k}_c$i.log" 2>&1 </dev/null &
            pids+=($!)
            case "${co[$i]}" in "$ITF"*) stop+=($!) ;; esac
        done
        [ "${#co[@]}" -gt 0 ] && sleep "$LEAD"
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="${PART_LIST[${#co[@]}]}" \
            "$REPORTER" --sizes-kb 2048,65536 --duration "$DURATION" > "$TMP/latency.txt" 2>&1
        rc=$?
        for pid in "${stop[@]}"; do kill -TERM "$pid" 2>/dev/null; done
        for pid in "${pids[@]}"; do wait "$pid"; done
        for f in "$TMP"/r"${k}"_c*_stats.csv; do
            [ -e "$f" ] && mv -f "$f" "$dest/"
        done
        rm -f "$TMP"/r"${k}"_c*
        mps_ensure off
        # The latency file is the completion marker, written last and only after a clean run.
        [ "$rc" = 0 ] && mv -f "$TMP/latency.txt" "$dest/r${k}_latency.txt" || { echo "   reporter failed (rc=$rc)"; exit 1; }
        awk '!/^#/{s[$1]+=$4; n[$1]++} END{for (k in s) printf "   %s KB: %.1f ns mean\n", k, s[k]/n[k]}' "$dest/r${k}_latency.txt"
    done
done < <(configs "$set_name")
