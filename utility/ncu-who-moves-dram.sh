#!/usr/bin/env bash
# ncu-who-moves-dram.sh - of the two clients in a pair, which one is asking the DRAM for data?
#
#   sudo ./utility/ncu-who-moves-dram.sh
#
# One unverified claim stands behind the whole contention argument: that compute_gemm's operands are
# L2-resident, so a co-runner costing the target 14 % asks the DRAM for nothing. That claim comes
# from the interferer's own accounting of what it asked for. This measures it.
#
# ncu serialises kernels, and for this question that costs nothing: the traffic a kernel generates
# on its own is the same whether or not another kernel runs beside it. What serialisation breaks is
# the interference itself, which is why it cannot answer the contention question - and this script
# does not try to.
#
# The two clients are separated by stream, which the CSV carries: the interferer runs in its own
# green context and its own stream, so grouping the rows by stream divides the pair cleanly.
set -u

NCU=${NCU:-/usr/local/cuda/bin/ncu}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/scratch/ncu-pair-dram}

METRICS="mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed,mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__t_bytes.sum"

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

# Two corrections after the first attempt blocked.
#
# The interferer's kernels are long and rare, so a window of consecutive launches misses them: a
# plain CUDA trace of the same pair shows 94545 TensorRT kernels against 3776 SGEMMs, and the SGEMM
# is named ampere_sgemm_64x64_nn. Filtering by that name instruments only the kernels wanted.
#
# And application replay cannot be used beside a second thread at all: it requires every pass to
# execute the same kernels in the same order, an interferer thread makes the count differ between
# passes, and ncu then retries until it is killed - "Unexpected number of profiled kernels ...
# Check the --app-replay-match options". Kernel replay, which is the default, has no such
# requirement, and a handful of kernels is seconds of work rather than minutes.
"$NCU" --metrics "$METRICS" --csv --log-file "$OUT/pair.csv" \
       --launch-count 5 --kernel-name regex:ampere_sgemm_64x64_nn --clock-control none \
       ./runner --submit-threads 1 --dataset "$DATA/dataset/imagenet-1k" --fixed 32 \
       --min-inferences 200 --min-duration 2 \
       --allocation greenctx --net "$MODEL:50:mobilenet" --interferer compute_gemm:25 \
       > "$OUT/pair.err" 2>&1

python3 - "$OUT/pair.csv" <<'PY'
import csv, collections, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print("   no output; see the .err beside it"); sys.exit()
by_stream = collections.defaultdict(lambda: collections.defaultdict(list))
names = collections.defaultdict(set)
for line in open(path, errors="replace"):
    if not line.startswith('"'):
        continue
    f = next(csv.reader([line]))
    if len(f) < 15 or f[12] == "Metric Name":
        continue
    try:
        v = float(f[14])
    except ValueError:
        continue
    key = (f[5], f[6])                      # context, stream
    by_stream[key][f[12].split(".")[0]].append(v)
    names[key].add(f[4][:40])

print("  %-14s %6s %10s %10s %9s  %s" % ("ctx/stream", "kern", "dram rd %", "dram wr %", "L2 hit %", "example kernel"))
print("  " + "-" * 104)
for key in sorted(by_stream, key=lambda k: -len(by_stream[k]["mcc__dram_throughput_op_read"])):
    m = by_stream[key]
    n = len(m["mcc__dram_throughput_op_read"])
    rd = sum(m["mcc__dram_throughput_op_read"]) / n
    wr = sum(m["mcc__dram_throughput_op_write"]) / n if m["mcc__dram_throughput_op_write"] else 0.0
    hit = m.get("lts__t_sector_hit_rate")
    example = sorted(names[key])[0] if names[key] else "?"
    print("  %-14s %6d %10.2f %10.2f %9s  %s" % (
        "%s/%s" % key, n, rd, wr,
        "%.1f" % (sum(hit) / len(hit)) if hit else "?", example))
PY

echo
echo "  the stream with few long kernels at near-zero DRAM is the interferer; the other is the target"
