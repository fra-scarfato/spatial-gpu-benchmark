#!/usr/bin/env bash
# ncu-diagnose.sh - can ncu measure anything on this board, and can it watch the caches?
#
#   sudo ./utility/ncu-diagnose.sh
#
# The eight earlier ncu runs returned 800 metric values of "n/a" each, and they asked for
# dram__bytes_read.sum and dram__bytes_write.sum. That may have been the whole failure:
#
#   ncu --query-metrics --chip ga10b    -> 4636 base metrics, 1930 of them lts__ (the L2),
#                                          and ZERO dram__ metrics. On this chip the DRAM counters
#                                          are mcc__dram_throughput_op_read and _op_write.
#
# So a name that exists on a discrete GPU does not exist here, and ncu answers a metric it cannot
# resolve with "n/a" rather than an error. This script therefore asks for metrics verified present
# in the ga10b list, of which the caches are the interesting ones:
#
#   l1tex__t_sector_hit_rate                       L1 hit rate
#   l1tex__t_sectors_pipe_lsu_mem_global_op_ld.*   L1 global-load sectors, and their misses
#   lts__t_sector_hit_rate                         L2 hit rate  <- the one that matters
#   lts__t_bytes.sum                               bytes the L2 served
#   mcc__dram_throughput_op_read/write             DRAM occupancy in percent of peak
#
# Two separate things are being tested and they must not be confused:
#
#   * privilege. As an ordinary user the counter library answers "no permission ... on the target
#     vGPU device". Whether root is refused too is decided by attempt 0, in thirty seconds.
#   * the metric name. If counters are reachable at all, the cache metrics above give what the
#     interferer question needs: whether a co-runner evicts the target's lines. A falling L2 hit
#     rate under a co-runner that moves no DRAM is eviction; a steady hit rate with a slower kernel
#     is arbitration or latency instead.
#
# Three more things were never varied and each has a plausible failure for a silent "n/a":
# clock control (Tegra refuses to let ncu lock the clocks), cache control (the default flushes
# caches before every replay pass), and replay mode (kernel replay saves and restores memory between
# passes; application replay re-runs the whole application instead).
set -u

NCU=${NCU:-/usr/local/cuda/bin/ncu}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/scratch/ncu-diag}

METRIC=${METRIC:-l1tex__t_sector_hit_rate,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,lts__t_sector_hit_rate,lts__t_sector_op_read_hit_rate,lts__t_bytes.sum,lts__t_bytes_equiv_l1sectormiss.sum}

# The DRAM counters this chip does have, in the same currency as the SoC counters: percent of peak.
# They are a cross-check of the copy calibration, which is the weakest number in the SoC analysis.
DRAM_METRIC=${DRAM_METRIC:-mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed,mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed}

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

RUNNER_ARGS=(--submit-threads 1 --dataset "$DATA/dataset/imagenet-1k" --fixed 32
             --min-inferences 200 --min-duration 2)
FLAGS="--clock-control none --cache-control none"

say() { echo; echo "== $* =="; }

say "the board: privileges, and whether this is a vGPU"
cat /proc/driver/nvidia/version 2>/dev/null | sed 's/^/   /'
grep -iE "profil|vgpu" /proc/driver/nvidia/params 2>/dev/null | sed 's/^/   /' || true
ls /sys/bus/mdev/devices 2>/dev/null | sed 's/^/   mdev: /' || true

say "attempt 0: does root see the metric list? (thirty seconds, and it decides the privilege question)"
"$NCU" --query-metrics > "$OUT/query-metrics.txt" 2>&1
if grep -qi "permission" "$OUT/query-metrics.txt"; then
    grep -iE "error|permission" "$OUT/query-metrics.txt" | head -4 | sed 's/^/   /'
    echo "   REFUSED even to root: no metric can be measured, and no flag changes that."
    echo "   Saved to $OUT/query-metrics.txt"
    exit 0
fi
echo "   root sees $(grep -c . "$OUT/query-metrics.txt") lines, so the counters are reachable."

verdict() {  # $1 = label, $2 = csv
    python3 - "$1" "$2" <<'PY'
import csv, os, sys
label, path = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("   %-22s no output file" % label); sys.exit()
rows = numeric = 0
seen = []
for line in open(path, errors="replace"):
    if not line.startswith('"'):
        continue
    f = next(csv.reader([line]))
    if len(f) < 15 or not f[12].strip():
        continue
    rows += 1
    try:
        float(f[14]); numeric += 1
        if len(seen) < 3:
            seen.append("%s=%s" % (f[12].split(".")[0][:34], f[14][:9]))
    except ValueError:
        pass
print("   %-22s %2d values, %2d numeric  %s" % (
    label, rows, numeric, "COUNTERS WORK" if numeric else "n/a"))
for s in seen:
    print("        " + s)
PY
}

attempt() {  # $1 = label, $2 = trace flags, rest = runner args
    local label=$1 trace=$2; shift 2
    echo
    echo "   -> $label"
    "$NCU" --launch-count 1 --metrics "$METRIC" --csv --log-file "$OUT/$label.csv" \
           $trace ./runner "${RUNNER_ARGS[@]}" "$@" > "$OUT/$label.err" 2>&1
    verdict "$label" "$OUT/$label.csv"
    grep -iE "^==(ERROR|WARNING)" "$OUT/$label.err" | head -2 | sed 's/^/      /' || true
}

say "the cache metrics, one launch of one real inference each"
echo "   (cache metrics are an event replay, so a launch count of 1 keeps this quick)"
attempt whole-default    ""                             --allocation whole   --net "$MODEL:100:mobilenet"
attempt whole-noclocks   "--clock-control none"         --allocation whole   --net "$MODEL:100:mobilenet"
attempt whole-nocache    "--cache-control none"         --allocation whole   --net "$MODEL:100:mobilenet"
attempt whole-both       "$FLAGS"                       --allocation whole   --net "$MODEL:100:mobilenet"
attempt whole-appreplay  "$FLAGS --replay-mode application" --allocation whole --net "$MODEL:100:mobilenet"
attempt greenctx-both    "$FLAGS"                       --allocation greenctx --net "$MODEL:50:mobilenet"
attempt greenctx-app     "$FLAGS --replay-mode application" --allocation greenctx --net "$MODEL:50:mobilenet"

say "the DRAM counters this chip has, percent of peak, for the calibration cross-check"
"$NCU" --launch-count 1 --metrics "$DRAM_METRIC" --csv --log-file "$OUT/mcc-dram.csv" \
       $FLAGS ./runner "${RUNNER_ARGS[@]}" --allocation whole --net "$MODEL:100:mobilenet" \
       > "$OUT/mcc-dram.err" 2>&1
python3 - "$OUT/mcc-dram.csv" <<'PY'
import csv, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("   mcc-dram: no output"); sys.exit()
found = False
for line in open(path, errors="replace"):
    if line.startswith('"'):
        f = next(csv.reader([line]))
        if len(f) >= 15 and f[12].startswith("mcc"):
            print("   %-58s %s" % (f[12][:58], f[14])); found = True
if not found:
    print("   mcc-dram: still nothing numeric")
PY

say "done"
echo "  logs and CSVs in $OUT"
echo "  a line saying COUNTERS WORK means the traffic and the hit rates are measurable after all"
