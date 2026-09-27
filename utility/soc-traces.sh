#!/usr/bin/env bash
#
# soc-traces.sh — profile the pair configurations with the Orin SoC counters on.
#
# Why these two, and not others: both rows put the same target on the same 8 SMs and
# differ only in the co-runner's budget, four SMs against eight. The difference between
# the traces is therefore the co-runner's traffic and nothing else, which is what the
# memory-contention question needs: whether the target's latency rise coincides with the
# co-runner's DRAM read and write throughput.
#
# The counters need root, because unprivileged nsys answers ERR_NVGPUCTRPERM:
#
#     sudo ~/f_scarfato/runner/utility/soc-traces.sh          # both configurations
#     sudo ~/f_scarfato/runner/utility/soc-traces.sh res8     # only the 8+8 one
#
# Two things make the absolute path of nsys explicit here. Three installations exist on
# this board — /usr/local/bin/nsys (2026.3.1), /opt/nvidia/nsight-systems/2026.3.1 and
# 2025.6.3 — and a viewer older than the recorder may fail to place the SoC samples,
# which is what makes the DRAM rows look empty. The version used is printed, so a
# trace's provenance is never in doubt. Override with NSYS=.
#
# Everything it writes goes to $DATA/scratch; its only deletion is its own temporary
# export, and it never removes anything else.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"                          # .../runner
DATA="${DATA:-/mnt/nvme/f.scarfato}"
NSYS="${NSYS:-/usr/local/bin/nsys}"

OUT="$DATA/scratch"
RUNNER="$ROOT/runner"
DS="$DATA/dataset/imagenet-1k"
MOB="$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine"
RES="$DATA/models/resnet50-v2-7/engine/resnet50-v2-7.engine"

[ "$(id -u)" = 0 ] || {
    echo "soc-traces.sh: run me with sudo — the SoC counters need root." >&2
    echo "               unprivileged nsys answers ERR_NVGPUCTRPERM." >&2
    exit 1
}
[ -x "$NSYS" ]   || { echo "soc-traces.sh: no nsys at $NSYS" >&2; exit 1; }
[ -x "$RUNNER" ] || { echo "soc-traces.sh: no runner at $RUNNER" >&2; exit 1; }
[ -d "$DS" ]     || { echo "soc-traces.sh: no dataset at $DS" >&2; exit 1; }

mkdir -p "$OUT"
echo "nsys: $NSYS"
"$NSYS" --version | head -1 | sed 's/^/       /'

# Five seconds of measurement per trace rather than one pass: the counters sample at
# 10 kHz, so a pass of 32 inferences would leave the average resting on a few hundred
# samples. Five seconds is also short enough to keep a root session brief on a board
# shared with another user.
profile() {  # $1 = name; the rest are --net arguments
    local name="$1"; shift
    echo
    echo "=== $name ==="
    printf '    %s\n' "$(IFS=' '; echo "$*")"
    # `--` before the application: nsys stops parsing its own options there, so the
    # runner's --net and --allocation cannot be mistaken for profiler flags.
    "$NSYS" profile \
        --soc-metrics=true --soc-metrics-set=t234 --soc-metrics-frequency=10000 \
        --force-overwrite=true \
        -o "$OUT/$name" \
        -- "$RUNNER" --submit-threads 1 --dataset "$DS" --allocation greenctx \
        "$@" --fixed 32 --min-inferences 2000 --min-duration 5 \
        --csv "$OUT/csv-$name" > "$OUT/$name.nsys.log" 2>&1
    printf '    trace: %s (%s bytes)\n' "$OUT/$name.nsys-rep" "$(stat -c %s "$OUT/$name.nsys-rep")"
    check_offset "$name"
}

# Did the counters land on the workload's clock? A viewer draws the SoC rows wherever
# their timestamps say they belong, so an offset of tens of seconds puts them past the
# end of the activity and the rows look empty. Reporting it here turns that from a
# mystery in the GUI into a number.
check_offset() {  # $1 = name
    local name="$1" db="$OUT/$name.check.sqlite"
    rm -f "$db"
    if ! "$NSYS" export --type sqlite --output "$db" "$OUT/$name.nsys-rep" > /dev/null 2>&1; then
        echo "    alignment: export failed, cannot check"
        return
    fi
    python3 - "$db" <<'PY'
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
try:
    k = c.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
except Exception:
    k = None
rows = c.execute("select * from SOC_METRICS").fetchall()
if not rows:
    print("    alignment: no SoC samples at all")
elif k is None:
    print("    alignment: no kernels to compare against")
else:
    off = (min(r[0] for r in rows) - k) / 1e9
    verdict = "aligned" if abs(off) < 1.0 else "MISALIGNED, the DRAM rows will look empty"
    print("    alignment: first SoC sample minus first kernel = {:+.3f} s   ({})".format(off, verdict))
PY
    rm -f "$db"
}

WANT="${*:-res4 res8}"
case " $WANT " in
    *" res4 "*) profile mob8-res4-soc --net "$MOB:50:mobilenet" --net "$RES:25:resnet" ;;
esac
case " $WANT " in
    *" res8 "*) profile mob8-res8-soc --net "$MOB:50:mobilenet" --net "$RES:50:resnet" ;;
esac

# The interferer arms. Two traces, same target and same 4 SMs for the participant, so
# the only difference is whether it moves memory or does arithmetic — which is the whole
# question, and the one thing ncu cannot answer on this board.
case " $WANT " in
    *" itf "*)
        profile itf-mem-saxpy --net "$MOB:50:mobilenet" --interferer mem_saxpy:25 ;;
esac
case " $WANT " in
    *" itf "*)
        profile itf-compute-gemm --net "$MOB:50:mobilenet" --interferer compute_gemm:25 ;;
esac

# nsys ran as root, so everything it wrote belongs to root. Hand it to the account that
# owns the runner, or the files are readable but not manageable.
OWNER="${SUDO_USER:-nvidia}"
chown -R "$OWNER:$(id -gn "$OWNER")" "$OUT" 2>/dev/null || true

cat <<'MSG'

Done. Nothing outside the scratch directory was touched.
MSG
