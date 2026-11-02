#!/usr/bin/env bash
# reporter.sh - the stress reporter beside a set of co-runners, under MPS static partitions.
#
#   measure/reporter.sh calibration          idle, then SAXPY on 8 SMs at levels 0 ... 1
#   measure/reporter.sh networks             each network alone as the co-runner, on 4 SMs
#   measure/reporter.sh calibration-4sm      SAXPY on 4 SMs at levels 0 ... 1: the networks' geometry
#   measure/reporter.sh probes-4sm           mem_saxpy (all DRAM) and compute_gemm (L2-resident), 4 SMs
#   measure/reporter.sh l2-vs-dram           the L2 chapter's 1d co-runners and their DRAM-matched SAXPY
#   measure/reporter.sh calibration-onoff    idle, spin, ON/OFF, uniform, smooth and peak SAXPY on 8 SMs (stress-onoff's)
#   measure/reporter.sh networks-8sm         each network alone as the co-runner, on 8 SMs (8 + 8)
#   measure/reporter.sh calibration-smooth   idle, spin and smooth SAXPY on 8 SMs: the validation set's ruler
#   measure/reporter.sh networks-new-8sm     the validation pairs' ten networks as the co-runner, on 8 SMs
#   measure/reporter.sh compose-4sm          two smooth SAXPY on 4 SMs each, four levels each, and each alone
#   measure/reporter.sh <set> --list         print the configurations, run nothing
#
# The reporter is RRZE-HPC's gpu-latency (probes/gpu-latency): one thread chasing pointers
# through an L2-sized and a DRAM-sized buffer, in its own 2-SM partition, so it adds almost no
# traffic and reads the stress the co-runners put on the memory system as a latency. The co-runners
# are started first and the reporter 3 s later. Beside itfproc (or alone) it runs for 30 s and
# itfproc is then stopped; beside networks it runs until they have completed the campaign's protocol
# (5000 inferences and 30 s), so their own statistics (occupancy, rate) are written too.
#
# Output: $DATA/bench/stress-reporter/<configuration>/r<k>_latency.txt, r<k>_c<i>.log and, for a
# network co-runner, r<k>_c<i>_stats.csv. A run whose latency file exists is skipped; every run is
# guarded like the campaign (clocks locked, nobody else on the GPU). MEASUREMENTS.md defines the
# quantities read from these files.
#
# LAYOUT=target puts the reporter where the target sits in the sweeps: its partition (8 SMs, chunks
# {0,1,4,5}) is created first, so every co-runner gets the chunks it has beside a target (SAXPY on 8
# SMs {2,3,6,7}, a 4-SM co-runner {2,6}). The reporter is one thread, so the size of its partition
# changes nothing but its place. Output then goes to $DATA/bench/stress-reporter-target.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nvme/f.scarfato}"
LAYOUT="${LAYOUT:-chunk}"             # chunk: the reporter on one 2-SM chunk after the co-runners;
                                     # target: on the target's 8 SMs, created first
case "$LAYOUT" in
    chunk)  OUTROOT="${OUTROOT:-$DATA/bench/stress-reporter}" ;;
    target) OUTROOT="${OUTROOT:-$DATA/bench/stress-reporter-target}" ;;
    *) echo "reporter.sh: LAYOUT is chunk or target" >&2; exit 2 ;;
esac
RUNS="${RUNS:-5}"
LEAD=3          # co-runners start this long before the reporter
DURATION=30     # the reporter's own measurement
SIZES_KB="${SIZES_KB:-65536}"        # the reporter's chain: DRAM-sized (an L2-sized one is evicted
                                     # outright by any co-runner, so it carries no level)
ITERS="${ITERS:-0}"                  # dereferences per sample; 0 = upstream's max(LEN, 100000)
WINDOW="${WINDOW:-512}"              # in-kernel windows of this many dereferences (~0.3 ms); 0 =
                                     # upstream's per-launch samples, which average bursts away
ONLY="${ONLY:-}"                     # a regular expression: run only the configurations it matches
DS="$DATA/dataset/imagenet-1k"
M="$DATA/models"
MOB="$M/mobilenetv2-12/engine/mobilenetv2-12.engine"
RES="$M/resnet50-v2-7/engine/resnet50-v2-7.engine"
INC="$M/inception_v3-onnx-float/engine/inception_v3.engine"
RET="$M/retinanet-9/engine/retinanet-9.engine"
MOBQ="$M/mobilenetv2-12/engine/mobilenetv2-12-int8.engine"      # the validation networks (sweep.sh)
RESQ="$M/resnet50-v2-7/engine/resnet50-v2-7-int8.engine"
INCQ="$M/inception_v3-onnx-float/engine/inception_v3-int8.engine"
RETQ="$M/retinanet-9/engine/retinanet-9-int8.engine"
EFF="$M/efficientnet_b0/engine/efficientnet_b0.engine"
VIT="$M/vit-base-patch16-224/engine/vit-base-patch16-224.engine"
RUNNER="$ROOT/runner"
ITF="${ITF:-$ROOT/probes/itfproc}"             # overridable: e.g. a binary built elsewhere
REPORTER="${REPORTER:-$ROOT/probes/gpu-latency/gpu-latency}"
TMP=/dev/shm/fs-reporter

STATIC_CHUNKS=8
MPS_PIPE="${MPS_PIPE:-/tmp/mps-$USER}"
MPS_LOG="${MPS_LOG:-/tmp/mps-log-$USER}"
MPS_MODE=off

# A network as a co-runner: its own process, the whole of its partition, and the campaign's protocol
# (at least 5000 inferences and 30 s). The reporter runs for as long as it does, and the network then
# exits by itself and writes its statistics.
# The smooth SAXPY at duty $1 (025 ... 100): 100 us periods, so the reporter's windows see it uniform.
smooth() { echo "$ITF saxpy --period-us 100 --duty $((10#$1 / 100)).$(printf '%02d' $((10#$1 % 100))) --burst 2"; }
net() { echo "$RUNNER --submit-threads 1 --dataset $DS --fixed 32 --allocation whole --net $1:100:$2 --min-inferences 5000 --min-duration $DURATION"; }

# The configurations of each set, one per line: name|co-runner shares|co-runner commands (;-separated).
# The reporter always takes one 2-SM chunk after the co-runners' partitions.
configs() {
    case "$1" in
    calibration)
        echo "idle||"
        local p
        # finer below 0.2, where every real network falls
        for p in 000 002 004 006 008 010 015 020 030 040 050 060 070 080 090 100; do
            echo "saxpy-L$p|50|$ITF saxpy --level $((10#$p / 100)).$(printf '%02d' $((10#$p % 100)))"
        done
        ;;
    calibration-4sm)
        # The same levels with the stressor on 4 SMs, chunks {0,4} like a network co-runner here
        # (and like the stressor of stress-sensitivity-4sm beside its target): calibration and use
        # then share one geometry. L is relative to full speed on these 4 SMs.
        local p
        for p in 000 002 004 006 008 010 015 020 030 040 050 060 070 080 090 100; do
            echo "saxpy4-L$p|25|$ITF saxpy --level $((10#$p / 100)).$(printf '%02d' $((10#$p % 100)))"
        done
        ;;
    calibration-tb)
        # The baseline's stressor: time-based SAXPY on 8 SMs, all threads on together for D of every
        # 1 ms, D = 0.1 ... 1.0; D = 0 is the resident kernel with no traffic. Use with WINDOW=0 (one
        # mean per run, the unpatched reporter) and LAYOUT=target.
        echo "tb-D000|50|$ITF saxpy --level 0"
        local d
        for d in 010 020 030 040 050 060 070 080 090 100; do
            echo "tb-D$d|50|$ITF saxpy --period-us 1000 --duty $((10#$d / 100)).$(printf '%02d' $((10#$d % 100)))"
        done
        ;;
    calibration-tbs)
        # calibration-tb with the SAXPY's lines scrambled within 2 MB windows (stress-sensitivity-tbs).
        echo "tbs-D000|50|$ITF saxpy --level 0"
        local d
        for d in 010 020 030 040 050 060 070 080 090 100; do
            echo "tbs-D$d|50|$ITF saxpy --period-us 1000 --duty $((10#$d / 100)).$(printf '%02d' $((10#$d % 100))) --scramble-kb 2048"
        done
        ;;
    probes-4sm)
        # Two co-runners whose cost to MobileNet under MPS on 4 SMs is already measured
        # (memory-mps-interferers-4sm): one whose traffic all reaches DRAM, one that hits the L2.
        echo "memsaxpy-4sm|25|$ITF mem_saxpy"
        echo "computegemm-4sm|25|$ITF compute_gemm"
        ;;
    l2-vs-dram)
        # The co-runners of the l2-vs-dram sweep family (matrix.sh), on 4 SMs: does the reporter
        # tell an L2-using co-runner from a SAXPY moving the same DRAM traffic?
        echo "l2d-spin-K100|25|$ITF spin --kernel-us 100"
        echo "l2d-computegemm|25|$ITF compute_gemm"
        echo "l2d-saxpy-L0243|25|$ITF saxpy --level 0.243"
        echo "l2d-gemm-N8|25|$ITF gemm --side 288 --sets 8 --period-us ${GEMM_PERIOD_US:-115}"
        echo "l2d-saxpy-L0087|25|$ITF saxpy --level 0.087"
        ;;
    calibration-onoff)
        # The stressors of the stress-onoff sweep family (matrix.sh), the same arguments, on 8 SMs:
        # the calibrations of stress.py's onoff and uniform geometries. Run with LAYOUT=target and
        # OUTROOT=$DATA/bench/stress-reporter-onoff, and the networks set into the same directory.
        echo "idle||"
        echo "spin|50|$ITF spin --kernel-us 100"      # level 0: resident, no DRAM traffic
        echo "onoff-D025|50|$ITF saxpy --onoff-ms 400 --duty 0.25"
        echo "onoff-D050|50|$ITF saxpy --onoff-ms 400 --duty 0.50"
        echo "onoff-D075|50|$ITF saxpy --onoff-ms 400 --duty 0.75"
        echo "onoff-D100|50|$ITF saxpy --onoff-ms 400 --duty 1.00"
        echo "uniform-D025|50|$ITF saxpy --mb 32 --onoff-ms 3.1 --duty 0.312"
        echo "uniform-D050|50|$ITF saxpy --mb 32 --onoff-ms 3.0 --duty 0.537"
        echo "uniform-D075|50|$ITF saxpy --mb 32 --onoff-ms 3.03 --duty 0.744"
        echo "smooth-D025|50|$ITF saxpy --period-us 100 --duty 0.25 --burst 2"
        echo "smooth-D050|50|$ITF saxpy --period-us 100 --duty 0.50 --burst 2"
        echo "smooth-D075|50|$ITF saxpy --period-us 100 --duty 0.75 --burst 2"
        echo "smooth-D100|50|$ITF saxpy --period-us 100 --duty 1.00 --burst 2"
        echo "peak-T0010|50|$ITF saxpy --period-us 100 --duty 0.10 --burst 2"
        echo "peak-T0100|50|$ITF saxpy --period-us 1000 --duty 0.10 --burst 2"
        echo "peak-T0300|50|$ITF saxpy --period-us 3000 --duty 0.10 --burst 2"
        echo "peak-T1000|50|$ITF saxpy --period-us 10000 --duty 0.10 --burst 2"
        echo "peak-T3000|50|$ITF saxpy --period-us 30000 --duty 0.10 --burst 2"
        ;;
    networks-8sm)
        # Daniel's 8 + 8 pairs (stress-validation-8): each network on the co-runner's 8 SMs.
        echo "mobilenet-8sm|50|$(net "$MOB" mobilenet)"
        echo "resnet-8sm|50|$(net "$RES" resnet)"
        echo "inception-8sm|50|$(net "$INC" inception)"
        echo "retinanet-8sm|50|$(net "$RET" retinanet)"
        ;;
    calibration-smooth)
        # The validation set's ruler (2026-10-13): calibration-onoff's zero and smooth stressors only,
        # taken again in the session of the networks it reads (networks-new-8sm), into the same
        # OUTROOT=$DATA/bench/stress-reporter-new, with LAYOUT=target.
        echo "idle||"
        echo "spin|50|$ITF spin --kernel-us 100"
        local d
        for d in 025 050 075 100; do
            echo "smooth-D$d|50|$(smooth $d)"
        done
        ;;
    compose-4sm)
        # Daniel's composition test (2026-10-22): does the stress of two co-runners add up? Two smooth
        # SAXPY on 4 SMs each beside the reporter on 8, four duties each, every combination, and each
        # duty alone on the first 4-SM partition (chunks {2,6}). The second, the remaining 4 SMs, has no
        # alone runs: the combinations hold every pair in both orders, which tests the two partitions
        # against each other. The spins are the zero, one resident co-runner and two. Run with
        # LAYOUT=target, after calibration-smooth into the same OUTROOT: the 8-SM ruler turns each
        # reading into an equivalent stress (analysis/compose.py). top8, the plain kernel back
        # to back on 8 SMs (ON/OFF at 100 %, ~135 GB/s), extends that ruler above smooth-D100 (~111):
        # two 4-SM stressors at high duty can read heavier than the smooth ruler's top.
        echo "top8|50|$ITF saxpy --onoff-ms 400 --duty 1.00"
        echo "spin4|25|$ITF spin --kernel-us 100"
        echo "spin4+spin4|25:25|$ITF spin --kernel-us 100;$ITF spin --kernel-us 100"
        local a b
        for a in 025 050 075 100; do
            echo "s4-D$a|25|$(smooth $a)"
            for b in 025 050 075 100; do
                echo "s4-D$a+s4-D$b|25:25|$(smooth $a);$(smooth $b)"
            done
        done
        ;;
    networks-new-8sm)
        # Every network of the validation pairs as the co-runner on 8 SMs: the six new ones and the
        # four FP16 ones again, so that all readings and the ruler come from one session.
        echo "mobilenet-8sm|50|$(net "$MOB" mobilenet)"
        echo "resnet-8sm|50|$(net "$RES" resnet)"
        echo "inception-8sm|50|$(net "$INC" inception)"
        echo "retinanet-8sm|50|$(net "$RET" retinanet)"
        echo "mobilenetint8-8sm|50|$(net "$MOBQ" mobilenetint8)"
        echo "resnetint8-8sm|50|$(net "$RESQ" resnetint8)"
        echo "inceptionint8-8sm|50|$(net "$INCQ" inceptionint8)"
        echo "retinanetint8-8sm|50|$(net "$RETQ" retinanetint8)"
        echo "efficientnet-8sm|50|$(net "$EFF" efficientnet)"
        echo "vit-8sm|50|$(net "$VIT" vit)"
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
configs "$set_name" >/dev/null || { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
if [ "${2:-}" = "--list" ]; then
    configs "$set_name" | while IFS='|' read -r name shares cmds; do
        printf '%-14s shares %-6s %s\n' "$name" "${shares:-none}" "${cmds:-(no co-runner)}"
    done
    exit 0
fi

[ -x "$REPORTER" ] && [ -x "$ITF" ] || { echo "reporter.sh: build the probes first: make -C probes" >&2; exit 1; }
export OUT="$TMP" MOB DS
mkdir -p "$TMP"
source "$ROOT/measure/lib/guards.sh"
source "$ROOT/measure/lib/mps.sh"
trap 'mps_stop; rm -rf "$TMP"' EXIT
check_clocks || exit 1
# The instrument and the board state, recorded before anything runs (lib/guards.sh).
check_binaries "$OUTROOT/_manifests" "$@" || exit 1
require_floor start || exit 1

while IFS='|' read -r name shares cmds; do
    [ -n "$ONLY" ] && ! [[ $name =~ $ONLY ]] && continue
    dest="$OUTROOT/$name"
    mkdir -p "$dest"
    # The floor again before each configuration that still has runs to do (FLOOR_REF: lib/guards.sh),
    # so a campaign never mixes floors; the probe needs the MPS daemon stopped.
    todo=0
    for k in $(seq 1 "$RUNS"); do [ -f "$dest/r${k}_latency.txt" ] || todo=1; done
    [ "$todo" = 1 ] && { mps_stop; require_floor "$name" || exit 1; }
    IFS=';' read -r -a co <<< "$cmds"
    for k in $(seq 1 "$RUNS"); do
        [ -f "$dest/r${k}_latency.txt" ] && { echo "== $name r$k: done, skipped"; continue; }
        require_quiet_gpu || exit 1
        # A fresh daemon for every run: removing partitions is only best effort (lib/mps.sh), and
        # a partition left behind makes the next run's set fail to fit.
        mps_ensure static || exit 1
        if [ "$LAYOUT" = target ]; then
            mps_create_partitions "50${shares:+:}${shares}" || exit 1
            rpart=0; coff=1                  # the reporter first, the co-runners after it
        else
            mps_create_partitions "${shares}${shares:+:}12" || exit 1
            rpart=${#co[@]}; coff=0
        fi
        echo "== $name r$k: ${#co[@]} co-runner(s)"
        # The guard sees the board only before a run. A co-tenant who starts during one would go
        # unnoticed, so a watcher polls every 2 s for GPU users other than this run's own processes
        # (all under f_scarfato or, for binaries in the data scratch, f.scarfato), and a run it flags is set
        # aside under rejected/ and not counted.
        rm -f "$TMP/.intruded"
        ( while sleep 2; do
              o=$(other_gpu_users 2>/dev/null | grep -v "f[._]scarfato")
              [ -n "$o" ] && { echo "$o" > "$TMP/.intruded"; break; }
          done ) & wpid=$!
        pids=(); stop=()
        for i in "${!co[@]}"; do
            csv=""
            case "${co[$i]}" in "$RUNNER"*) csv="--csv $TMP/r${k}_c$i" ;; esac
            # shellcheck disable=SC2086  # a command line, split into words on purpose
            env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="${PART_LIST[$((i + coff))]}" \
                ${co[$i]} $csv > "$dest/r${k}_c$i.log" 2>&1 </dev/null &
            pids+=($!)
            case "${co[$i]}" in "$ITF"*) stop+=($!) ;; esac
        done
        [ "${#co[@]}" -gt 0 ] && sleep "$LEAD"
        # Beside networks the reporter runs until they have finished their protocol; beside itfproc,
        # or alone, for DURATION. Its output is flushed sample by sample, so stopping it loses nothing.
        dur="$DURATION"
        [ "${#stop[@]}" -lt "${#co[@]}" ] && dur=100000
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="${PART_LIST[$rpart]}" \
            "$REPORTER" --sizes-kb "$SIZES_KB" --duration "$dur" --iters "$ITERS" --window "$WINDOW" \
            > "$TMP/latency.txt" 2>&1 &
        rpid=$!
        if [ "$dur" = "$DURATION" ]; then
            wait "$rpid"; rc=$?
        else
            for pid in "${pids[@]}"; do wait "$pid"; done
            kill -TERM "$rpid" 2>/dev/null; wait "$rpid" 2>/dev/null; rc=0
            grep -q "^[0-9]" "$TMP/latency.txt" || rc=1
        fi
        for pid in "${stop[@]}"; do kill -TERM "$pid" 2>/dev/null; done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
        for f in "$TMP"/r"${k}"_c*_stats.csv; do
            [ -e "$f" ] && mv -f "$f" "$dest/"
        done
        rm -f "$TMP"/r"${k}"_c*
        mps_ensure off
        kill "$wpid" 2>/dev/null; wait "$wpid" 2>/dev/null
        if [ -s "$TMP/.intruded" ]; then
            echo "   REJECTED: another process used the GPU during the run: $(head -1 "$TMP/.intruded" | tr -s " ")"
            mkdir -p "$OUTROOT/rejected/$name"
            mv -f "$TMP/latency.txt" "$OUTROOT/rejected/$name/r${k}_latency.txt" 2>/dev/null
            mv -f "$dest"/r"${k}"_c* "$OUTROOT/rejected/$name/" 2>/dev/null
            exit 1                      # the caller's retry waits for a free GPU and resumes here
        fi
        # The latency file is the completion marker, written last and only after a clean run.
        [ "$rc" = 0 ] && mv -f "$TMP/latency.txt" "$dest/r${k}_latency.txt" || { echo "   reporter failed (rc=$rc)"; exit 1; }
        awk '!/^#/{s[$1]+=$4; n[$1]++} END{for (k in s) printf "   %s KB: %.1f ns mean\n", k, s[k]/n[k]}' "$dest/r${k}_latency.txt"
    done
done < <(configs "$set_name")
mps_stop
require_floor end || exit 1
