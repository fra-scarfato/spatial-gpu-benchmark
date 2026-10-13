#!/usr/bin/env bash
# profile.sh - every profiled run of this work, from one place.
#
#   utility/profile.sh traces           CUDA kernel traces of the target beside each co-runner (no root)
#   utility/profile.sh kernels          CUDA kernel traces of each network alone: kernels per inference (no root)
#   sudo utility/profile.sh soc-budget  DRAM counters: each network alone on four budgets, three pairs
#   sudo utility/profile.sh soc-itf     DRAM counters: the target alone and beside compute_gemm
#   sudo utility/profile.sh soc-l2      DRAM counters: the L2 chapter's co-runners under MPS static
#   sudo utility/profile.sh soc-onoff   DRAM counters: ON/OFF and uniform SAXPY, for 10 ms series
#   sudo utility/profile.sh footprint   DRAM counters: compute_gemm's operand side varied, and idle
#   sudo utility/profile.sh calibration DRAM counters during devprobe's device-to-device copies
#   utility/profile.sh kernels-new      kernels: the validation networks (INT8, EfficientNet, ViT) (no root)
#   sudo utility/profile.sh soc-new     DRAM counters: each validation network alone on 8 SMs
#   utility/profile.sh <set> --list     print the arms and their commands, run nothing
#
# A profiled run is a diagnostic, not a campaign measurement: the profiler perturbs the run, so
# these are one run per arm and are read for shapes, ratios and per-kernel structure, never for a
# latency that a table reports. The campaign is sweep.sh's.
#
# Output: $DATA/profiles/<set>/<arm>.nsys-rep, <arm>.log (the runner's own report) and
# <arm>_stats.csv. An arm whose report exists is skipped unless FORCE=1. Every arm is guarded like
# the campaign (clocks locked, nobody else on the GPU, the sentinel within tolerance); with
# WAIT_MIN=<n> an arm waits up to n minutes for the board to become quiet instead of aborting.
#
# Read them with: python3 utility/profiles.py
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${DATA:-/mnt/nvme/f.scarfato}"
NSYS="${NSYS:-/usr/local/bin/nsys}"
WAIT_MIN="${WAIT_MIN:-0}"
DS="$DATA/dataset/imagenet-1k"
M="$DATA/models"
MOB="$M/mobilenetv2-12/engine/mobilenetv2-12.engine"
RES="$M/resnet50-v2-7/engine/resnet50-v2-7.engine"
INC="$M/inception_v3-onnx-float/engine/inception_v3.engine"
RET="$M/retinanet-9/engine/retinanet-9.engine"
# The validation networks (sweep.sh), as engine:label; the label is the engine's name, as soc-budget's.
NEW="$M/mobilenetv2-12/engine/mobilenetv2-12-int8.engine:mobilenetv2-12-int8
$M/resnet50-v2-7/engine/resnet50-v2-7-int8.engine:resnet50-v2-7-int8
$M/inception_v3-onnx-float/engine/inception_v3-int8.engine:inception_v3-int8
$M/retinanet-9/engine/retinanet-9-int8.engine:retinanet-9-int8
$M/efficientnet_b0/engine/efficientnet_b0.engine:efficientnet_b0
$M/vit-base-patch16-224/engine/vit-base-patch16-224.engine:vit-base-patch16-224"
RUNNER="$ROOT/runner"

set_name="${1:-}"; shift || true
LIST=0; [ "${1:-}" = "--list" ] && LIST=1

# The protocols. The campaign's 30 s and 5000 inferences for everything that is compared with the
# campaign; the budget curve keeps the 20 s it was taken with, since its quantity is bytes per
# inference and not a latency.
LONG=(--min-inferences 5000 --min-duration 30)
SHORT=(--min-inferences 100 --min-duration 20)
TRACE=(--trace=cuda --sample=none --cpuctxsw=none)
SOC=("${TRACE[@]}" --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000)

# The arms of each set, one per line: name|protocol|runner arguments. The target is MobileNet on 8
# of the 16 SMs wherever there is a target; a co-runner gets 4 unless its name says otherwise.
arms() {
    local T="--allocation greenctx --net $MOB:50:mobilenet"
    case "$1" in
    traces)
        echo "alone|LONG|$T"
        echo "mem4|LONG|$T --interferer mem_saxpy:25"
        echo "compute4|LONG|$T --interferer compute_gemm:25"
        echo "spin4|LONG|$T --interferer spin:25"
        echo "resnet4|LONG|$T --net $RES:25:resnet"
        ;;
    kernels)
        # Each network alone on 8 SMs: how many kernels one inference launches, which sets how many
        # kernel boundaries a co-runner's presence can cost it (the L2 chapter's presence experiment).
        local m
        for m in "$MOB:mobilenet" "$RES:resnet" "$INC:inception" "$RET:retinanet"; do
            echo "alone-${m#*:}|LONG|--allocation greenctx --net ${m%%:*}:50:${m#*:}"
        done
        ;;
    kernels-new)
        # kernels for the validation networks, which counter-model.py's presence term and
        # ncu.sh nets-new need (kernels per inference).
        local m
        for m in $NEW; do
            echo "alone-${m#*:}|LONG|--allocation greenctx --net ${m%%:*}:50:${m#*:}"
        done
        ;;
    soc-new)
        # The validation networks' DRAM demand alone on 8 SMs (green context 50 %), soc-budget's
        # 50 % arm under the same name, so profiles.py's dram.csv gets their rows.
        local m
        for m in $NEW; do
            echo "soc-greenctx-${m#*:}-50|SHORT|--allocation greenctx --net ${m%%:*}:50:${m#*:}"
        done
        ;;
    soc-budget)
        local m p
        for m in "$MOB:mobilenetv2-12" "$RES:resnet50-v2-7" "$INC:inception_v3" "$RET:retinanet-9"; do
            for p in 100 75 50 25; do
                echo "soc-greenctx-${m#*:}-$p|SHORT|--allocation greenctx --net ${m%%:*}:$p:${m#*:}"
            done
        done
        echo "soc-greenctx-mobilenetv2-12-50_resnet50-v2-7-25|SHORT|--allocation greenctx --net $MOB:50:mobilenetv2-12 --net $RES:25:resnet50-v2-7"
        echo "soc-greenctx-mobilenetv2-12-50_resnet50-v2-7-50|SHORT|--allocation greenctx --net $MOB:50:mobilenetv2-12 --net $RES:50:resnet50-v2-7"
        echo "soc-greenctx-mobilenetv2-12-50_retinanet-9-25|SHORT|--allocation greenctx --net $MOB:50:mobilenetv2-12 --net $RET:25:retinanet-9"
        ;;
    soc-itf)
        echo "alone|LONG|$T"
        echo "compute4|LONG|$T --interferer compute_gemm:25"
        ;;
    soc-l2)
        # The inputs of the L2 chapter's block 1d. Each co-runner runs on 4 SMs under MPS static as
        # in its campaign family, alone for 20 s and beside MobileNet on 8, to answer: is l2read
        # L2-resident (DRAM about zero)? Does the GEMM's DRAM traffic grow with its working set?
        # Which Steady level matches each co-runner's DRAM rate?
        local I="$ROOT/utility/probes/itfproc" c n
        local cs=("spin-K100:spin --kernel-us 100" "l2read-L050:l2read --kernel-us 100 --level 0.50"
                  "l2read-L100:l2read --kernel-us 100 --level 1.00" "compute_gemm:compute_gemm")
        for n in 1 2 3 4 5 8; do
            cs+=("gemm-N$n:gemm --side 288 --sets $n --period-us ${GEMM_PERIOD_US:-115}")
        done
        echo "mobilenet-alone|ITF|--allocation whole --net $MOB:50:mobilenet|-"
        for c in "${cs[@]}"; do
            echo "alone-${c%%:*}|ITF|-|$I ${c#*:}"
            echo "mobilenet-${c%%:*}|ITF|--allocation whole --net $MOB:50:mobilenet|$I ${c#*:}"
        done
        ;;
    soc-onoff)
        # Daniel's 10 ms DRAM series: is the traffic of ON/OFF concentrated in ON, is the uniform
        # variant uniform, and do the two reach the same mean? Each stressor on 8 SMs (50:50, the
        # stress-onoff geometry), alone for 20 s; then MobileNet next to the two at 50 %.
        local I="$ROOT/utility/probes/itfproc" c
        local cs=(
            "onoff-D025:saxpy --onoff-ms 400 --duty 0.25"
            "onoff-D050:saxpy --onoff-ms 400 --duty 0.50"
            "onoff-D075:saxpy --onoff-ms 400 --duty 0.75"
            "onoff-D100:saxpy --onoff-ms 400 --duty 1.00"
            "uniform-D025:saxpy --mb 32 --onoff-ms 3.1 --duty 0.312"
            "uniform-D050:saxpy --mb 32 --onoff-ms 3.0 --duty 0.537"
            "uniform-D075:saxpy --mb 32 --onoff-ms 3.03 --duty 0.744"
        )
        for c in "${cs[@]}"; do
            echo "alone-${c%%:*}|ITF|-|$I ${c#*:}"
        done
        echo "mobilenet-alone|ITF|--allocation whole --net $MOB:50:mobilenet|-"
        for c in "${cs[1]}" "${cs[5]}"; do
            echo "mobilenet-${c%%:*}|ITF|--allocation whole --net $MOB:50:mobilenet|$I ${c#*:}"
        done
        ;;
    calibration)
        echo "devprobe|NONE|"
        ;;
    footprint)
        echo "A0-idle|LONG|$T --interferer idle:25"
        local a
        for a in A1-l1:64 B-1mb:288 C-2mb:400 D-3mb:512 E-4mb:576 F-12mb:1024; do
            echo "${a%%:*}|LONG|$T --interferer compute_gemm:25 --interferer-size ${a#*:}"
        done
        ;;
    *) return 1 ;;
    esac
}

flags_for() { case "$1" in traces|kernels|kernels-new) echo TRACE ;; *) echo SOC ;; esac; }

arms "$set_name" >/dev/null || {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
}

OUT="$DATA/profiles/$set_name"
if [ "$LIST" = 1 ]; then
    arms "$set_name" | while IFS='|' read -r name proto args; do
        printf '%-50s %s %s\n' "$name" "$proto" "$args"
    done
    exit 0
fi

if [ "$(flags_for "$set_name")" = SOC ] && [ "$(id -u)" != 0 ]; then
    echo "profile.sh: the $set_name set reads the SoC counters, which need root: sudo $0 $set_name" >&2
    exit 1
fi

mkdir -p "$OUT"
export OUT RUNNER MOB DS
source "$ROOT/utility/lib/guards.sh"
check_clocks || exit 1
check_binaries "$OUT/_manifests" "$set_name" || exit 1

wait_quiet() {   # the campaign's guard, retried once a minute for up to WAIT_MIN minutes
    local n=0
    until require_quiet_gpu; do
        n=$((n + 1))
        [ "$n" -gt "$WAIT_MIN" ] && { echo "[profile] the board stayed busy; stopping" >&2; return 1; }
        echo "[profile] waiting for a quiet board ($n/$WAIT_MIN min)"
        sleep 60
    done
}

# The guard only sees the board before an arm. A co-tenant who starts during one would go unnoticed,
# so a watcher polls the GPU device nodes every two seconds while the arm runs, and an arm it flags
# is moved to rejected/ and taken again. A process is this arm's own when it descends from this
# script, or when it is the Nsight agent of a session whose nsys does: the agent is started as a
# separate process and names its session after that nsys's pid.
descends_from_me() {   # $1 = pid
    local p=$1
    while [ -n "$p" ] && [ "$p" -gt 1 ]; do
        [ "$p" = "$$" ] && return 0
        p=$(awk '/^PPid:/{print $2}' "/proc/$p/status" 2>/dev/null)
    done
    return 1
}
intruders() {
    local p f t cmd pid
    for p in /proc/[0-9]*; do
        pid=${p#/proc/}
        for f in "$p"/fd/*; do
            t=$(readlink "$f" 2>/dev/null) || continue
            case "$t" in /dev/nvgpu/*|/dev/nvmap|/dev/dri/renderD*) ;; *) continue ;; esac
            cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null)
            [ -n "$cmd" ] || break
            descends_from_me "$pid" && break
            if [[ $cmd =~ --session-name\ profile-([0-9]+) ]] && descends_from_me "${BASH_REMATCH[1]}"; then
                break
            fi
            case "$cmd" in *nvidia-cuda-mps*) break ;; esac
            is_desktop "$cmd" && break
            echo "$pid $cmd"; return 0
        done
    done
    return 1
}
watch_gpu() { while sleep 2; do intruders >> "$OUT/.intruded" && return; done; }

# A single-network run writes <prefix>_stats.csv, a multi-network one <prefix>_<label>_stats.csv.
keep_stats() {   # $1 = arm
    local f
    for f in /dev/shm/fs-"$1"_*stats.csv; do
        [ -e "$f" ] && cp -f "$f" "$OUT/$1${f#/dev/shm/fs-$1}"
    done
    rm -f /dev/shm/fs-"$1"_*
    awk -F, '$1=="gpu_ms"{printf "   median %s ms over %s inferences\n", $4, $2}' "$OUT/$1"*_stats.csv 2>/dev/null
}

# The ITF arms (soc-l2, soc-onoff) run as sweep.sh's do_job does: an MPS static daemon on a private
# pipe, the target in a 50 % partition and the itfproc co-runner in a 25 % one (50 % for soc-onoff), the co-runner started 3 s
# ahead and stopped with SIGTERM when the target exits. nsys runs only the target, which is enough
# because the SoC counters are device-wide. With no target, nsys runs the co-runner alone for 20 s.
# The daemon lives for one arm only, so the guard and its sentinel always see a board without MPS.
# Under sudo, nsys starts its application as the invoking user unless told otherwise, and an MPS
# server only takes clients of its own user: --run-as=root keeps the daemon, the co-runner and the
# target under one user (without it the target waits forever for the server).
MPS_PIPE=/tmp/mps-profile MPS_LOG=/tmp/mps-log-profile STATIC_CHUNKS=8 MPS_MODE=off
ITF_SHARES=50:25; [ "$set_name" = soc-onoff ] && ITF_SHARES=50:50
source "$ROOT/utility/lib/mps.sh"
run_itf() {   # $1 = arm, $2 = "<target runner arguments>|<itfproc command>", either may be "-"
    local tgt="${2%%|*}" itf="${2#*|}" pid="" rc
    [ "$tgt" = - ] && tgt=""
    [ "$itf" = - ] && itf=""
    mps_stop                          # a daemon left by an interrupted run would block the start
    mps_start static > "$OUT/$1.mps.log" && mps_create_partitions "$ITF_SHARES" >> "$OUT/$1.mps.log" ||
        { mps_stop; return 1; }
    # shellcheck disable=SC2086  # tgt and itf are lists of words by construction
    if [ -z "$tgt" ]; then
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="$PART_B" \
            "$NSYS" profile --force-overwrite=true --run-as=root "${FLAGS[@]}" --output "$OUT/$1" \
            timeout -s TERM 20 $itf > "$OUT/$1.log" 2>&1 </dev/null
        rc=$?
        [ "$rc" = 124 ] && rc=0       # the 20 s timeout is how this arm ends
    else
        if [ -n "$itf" ]; then
            env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="$PART_B" \
                $itf > "$OUT/$1_itf.log" 2>&1 </dev/null &
            pid=$!
            sleep 3
        fi
        env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" CUDA_MPS_SM_PARTITION="$PART_A" \
            "$NSYS" profile --force-overwrite=true --run-as=root "${FLAGS[@]}" --output "$OUT/$1" \
            "$RUNNER" --submit-threads 1 --dataset "$DS" --fixed 32 "${SHORT[@]}" $tgt \
            --csv "/dev/shm/fs-$1" > "$OUT/$1.log" 2>&1 </dev/null
        rc=$?
        [ -n "$pid" ] && { kill -TERM "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; }
        keep_stats "$1"
    fi
    mps_stop
    return "$rc"
}

declare -n FLAGS="$(flags_for "$set_name")"
"$NSYS" --version | head -1
while IFS='|' read -r name proto args; do
    if [ -f "$OUT/$name.nsys-rep" ] && [ "${FORCE:-0}" != 1 ]; then
        echo "== $name: profiled already, skipped (FORCE=1 to repeat)"
        continue
    fi
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    wait_quiet </dev/null || exit 1
    echo "== $name"
    rm -f "$OUT/.intruded"
    watch_gpu & WATCH=$!
    if [ "$proto" = NONE ]; then      # the calibration: devprobe's copies, not the runner
        make -s -C "$ROOT/utility/probes" devprobe
        "$NSYS" profile --force-overwrite=true "${FLAGS[@]}" --output "$OUT/$name" \
            "$ROOT/utility/probes/devprobe" > "$OUT/$name.log" 2>&1 </dev/null
        echo "   exit $?"
    elif [ "$proto" = ITF ]; then
        run_itf "$name" "$args"
        echo "   exit $?"
    else
    declare -n P="$proto"
    # shellcheck disable=SC2086  # args is a list of words by construction
    "$NSYS" profile --force-overwrite=true "${FLAGS[@]}" --output "$OUT/$name" \
        "$RUNNER" --submit-threads 1 --dataset "$DS" --fixed 32 "${P[@]}" $args \
        --csv "/dev/shm/fs-$name" > "$OUT/$name.log" 2>&1 </dev/null
    echo "   exit $?"
    keep_stats "$name"
    fi
    kill "$WATCH" 2>/dev/null; wait "$WATCH" 2>/dev/null
    [ -s "$OUT/.intruded" ] || break
    echo "   REJECTED: another process used the GPU during the run: $(head -1 "$OUT/.intruded")"
    mkdir -p "$OUT/rejected/$attempt"
    mv -f "$OUT/$name".* "$OUT/$name"_* "$OUT/rejected/$attempt/" 2>/dev/null
  done
done < <(arms "$set_name")
rm -f "$OUT/.intruded"
rm -f "$OUT"/_sentinel_*

# Under sudo everything above belongs to root; hand it back to the account that owns the data.
[ "$(id -u)" = 0 ] && chown -R "${SUDO_USER:-nvidia}:" "$OUT" 2>/dev/null
exit 0
