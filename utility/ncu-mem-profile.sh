#!/usr/bin/env bash
# ncu-mem-profile.sh - does a co-runner evict the target's lines, or only get in its way?
#
#   sudo ./utility/ncu-mem-profile.sh
#
# This is the measurement that was thought to be impossible on this board. ncu does work here: the
# metric names just have to be ones this chip has. ga10b exposes no dram__ metric at all - its DRAM
# counters are mcc__dram_throughput_op_read/write - and ncu answers an unresolvable metric with "n/a"
# rather than an error, which is how eight runs came back with 800 n/a values each.
#
# The question: MobileNet at 8 SMs is 17 % slower in its median when compute_gemm runs beside it on
# 4 SMs, and compute_gemm moves no DRAM because its operands are L2-resident. Two explanations, and
# the hit rates separate them:
#
#   eviction      the co-runner's lines displace the target's, so the target's L2 hit rate falls and
#                 it re-reads from DRAM. Signature: lts__t_sector_hit_rate down, mcc__dram up.
#   arbitration   the lines stay, but every access waits longer. Signature: hit rates flat, the
#                 kernel slower, DRAM unchanged.
#
# The L1 is the control. It is per-SM and the two clients hold disjoint SMs, so no co-runner can
# evict the target's L1 lines: that hit rate should not move. If it does, the explanation is not
# cache capacity at all and the conclusion changes.
#
# Four conditions, the same fixed 32 inputs and the same protocol as every other measurement, and
# application replay rather than kernel replay because it was tested above and the whole thing then
# costs seconds per condition rather than minutes.
set -u

NCU=${NCU:-/usr/local/cuda/bin/ncu}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/scratch/ncu-mem}

# every name verified present in `ncu --query-metrics --chip ga10b`
METRICS="l1tex__t_sector_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_bytes.sum,lts__t_bytes_equiv_l1sectormiss.sum,mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed,mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed"

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

BASE=(--submit-threads 1 --dataset "$DATA/dataset/imagenet-1k" --fixed 32
      --min-inferences 200 --min-duration 2)

# the launch window: a fifth of an inference, from the middle of it. MobileNet is 55 kernels per
# inference, so the first twenty after skipping three hundred are whole work, not warm-up.
WINDOW=(--launch-skip 300 --launch-count 20 --replay-mode application --clock-control none)

run_one() {  # $1 = label; then the runner arguments
    local label=$1; shift
    echo
    echo "== $label =="
    "$NCU" --metrics "$METRICS" --csv --log-file "$OUT/$label.csv" "${WINDOW[@]}" \
           ./runner "${BASE[@]}" "$@" > "$OUT/$label.err" 2>&1
    python3 - "$label" "$OUT/$label.csv" <<'PY'
import csv, os, sys
label, path = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("   no output file; see %s.err" % label); sys.exit()
rows = numbers = 0
kernels = set()
for line in open(path, errors="replace"):
    if not line.startswith('"'):
        continue
    f = next(csv.reader([line]))
    if len(f) < 15 or not f[12].strip():
        continue
    rows += 1
    kernels.add(f[4])
    try:
        float(f[14]); numbers += 1
    except ValueError:
        pass
print("   %d metric values, %d numeric, %d distinct kernels" % (rows, numbers, len(kernels)))
PY
    grep -iE "^==(ERROR)" "$OUT/$label.err" | head -2 | sed 's/^/   /' || true
}

# the reference and the treatment, same SM budget, differing only in the co-runner
run_one target8-alone    --allocation greenctx --net "$MODEL:50:mobilenet"
run_one target8-compute  --allocation greenctx --net "$MODEL:50:mobilenet" --interferer compute_gemm:25
run_one target8-mem      --allocation greenctx --net "$MODEL:50:mobilenet" --interferer mem_saxpy:25
# and the whole device, to compare against the SoC counter's reading of the same network
run_one target-whole     --allocation whole    --net "$MODEL:100:mobilenet"

echo
echo "  CSVs in $OUT; the analysis attributes rows to the target and to the interferer by kernel name"
