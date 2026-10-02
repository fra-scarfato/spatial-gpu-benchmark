#!/usr/bin/env bash
# ncu.sh - the one Nsight Compute driver, in the three modes this work needs.
#
#   sudo ./utility/ncu.sh diagnose        does ncu work here, and under which flags
#   sudo ./utility/ncu.sh target [side]   the target's per-kernel L1 and L2 statistics, in a pair
#   sudo ./utility/ncu.sh pair   [side]   the co-runner's own DRAM demand, filtered to its kernel
#
# Three scripts had grown around this profiler, differing only in the flags they pass. What they
# share is worth stating once, because it is a property of the board rather than of the flags:
#
#   * ncu works here. The metric names just have to be ones this chip has: ga10b exposes 4636 base
#     metrics and not one beginning with dram__, because its DRAM counters are mcc__dram_throughput_*
#     and ncu answers a metric it cannot resolve with "n/a" rather than an error.
#   * the permission error - "no permission ... on the target vGPU device 0" - appears only for an
#     unprivileged user. Under root the counters are reachable.
#   * ncu quiesces the other clients around every kernel it measures, so it cannot observe
#     contention at all. Its use here is per-kernel behaviour and single-client demand, never the
#     interference itself, which is why the SoC counters carry every contention measurement.
#
# target is the validation the footprint sweep needs: the L2 hit rate of the target's own kernels
# should be high on the small arms and fall when the co-runner leaves the cache. pair answers
# whether the compute participant really asks the DRAM for little, which the displacement test
# subtracts and the demand table quotes.
set -u

NCU=${NCU:-/usr/local/cuda/bin/ncu}
DATA=${DATA:-/mnt/nvme/f.scarfato}
MODEL=${MODEL:-$DATA/models/mobilenetv2-12/engine/mobilenetv2-12.engine}
OUT=${OUT:-$DATA/profiles/ncu}
MODE=${1:-}
SIZE=${2:-512}

mkdir -p "$OUT"
cd "$(dirname "$0")/.." || exit 1

BASE=(--submit-threads 1 --dataset "$DATA/dataset/imagenet-1k" --fixed 32
      --min-inferences 200 --min-duration 2)
GREEN=(--allocation greenctx --net "$MODEL:50:mobilenet" --interferer "compute_gemm:25"
       --interferer-size "$SIZE")
# the names below were all verified present in `ncu --query-metrics --chip ga10b`
CACHE="l1tex__t_sector_hit_rate.pct,l1tex__t_sector_pipe_lsu_mem_global_op_ld_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,lts__t_sector_hit_rate.pct,lts__t_sector_op_read_hit_rate.pct,lts__t_bytes.sum,lts__t_bytes_equiv_l1sectormiss.sum"
DRAM="mcc__dram_throughput_op_read.avg.pct_of_peak_sustained_elapsed,mcc__dram_throughput_op_write.avg.pct_of_peak_sustained_elapsed"

report() {  # $1 = csv, $2 = label column header
    python3 - "$1" "$2" <<'PY'
import csv, os, sys
path, hdr = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    print("   no output; see the .err beside it"); sys.exit()
per, names = {}, {}
for line in open(path, errors="replace"):
    if not line.startswith('"'):
        continue
    f = next(csv.reader([line]))
    if len(f) < 15 or f[12] == "Metric Name":
        continue
    key = (f[5], f[6], f[4])          # context, stream, kernel
    per.setdefault(key, {})[f[12]] = f[14]
    names.setdefault(key, f[4])
print("   %s" % hdr)
for key in sorted(per, key=lambda k: -len(per[k])):
    m = per[key]
    got = " ".join("%s=%s" % (k.split("__")[-1].split(".")[0][:14], v)
                   for k, v in list(m.items())[:4])
    print("     ctx%s st%-4s %-40s %s" % (key[0], key[1], names[key][:40], got))
PY
}

case "$MODE" in
diagnose)
    echo "== the board, and whether root sees the counters =="
    grep -iE "profil|vgpu" /proc/driver/nvidia/params 2>/dev/null | sed 's/^/  /' || true
    "$NCU" --query-metrics > "$OUT/query-metrics.txt" 2>&1
    if grep -qi permission "$OUT/query-metrics.txt"; then
        grep -iE "error|permission" "$OUT/query-metrics.txt" | head -3 | sed 's/^/  /'
        echo "  REFUSED even to root: no metric can be measured here."
        exit 0
    fi
    echo "  root sees $(grep -c . "$OUT/query-metrics.txt") metrics"
    echo
    echo "== the flag combinations, one launch of one real inference each =="
    for f in "" "--clock-control none" "--cache-control none" \
             "--clock-control none --cache-control none" \
             "--clock-control none --cache-control none --replay-mode application"; do
        label=$(echo "flags${f}" | tr ' -' '__')
        "$NCU" --launch-count 1 --metrics "$DRAM" --csv --log-file "$OUT/$label.csv" \
               $f ./runner "${BASE[@]}" --allocation whole --net "$MODEL:100:mobilenet" \
               > "$OUT/$label.err" 2>&1
        n=$(grep -c 'mcc__' "$OUT/$label.csv" 2>/dev/null)
        v=$(awk -F'","' '/mcc__dram_throughput_op_read/{print $NF; exit}' "$OUT/$label.csv" 2>/dev/null)
        printf "  %-46s %s values, first reading %s\n" "${f:-defaults}" "$n" "${v:-none}"
    done
    ;;
target)
    echo "== the target's own kernels, with the co-runner present but quiesced by the profiler =="
    "$NCU" --metrics "$CACHE" --csv --log-file "$OUT/target-$SIZE.csv" \
           --launch-skip 300 --launch-count 20 --kernel-name "regex:.*xmma.*" --clock-control none \
           ./runner "${BASE[@]}" --csv /dev/shm/ncu-target "${GREEN[@]}" \
           > "$OUT/target-$SIZE.err" 2>&1
    rm -f /dev/shm/ncu-target*
    report "$OUT/target-$SIZE.csv" "kernel, L1 hit, L1 load sectors, L2 hit, L2 bytes"
    echo "   the L2 hit rate should be high while the co-runner's operands fit in the cache, and"
    echo "   fall once they do not: this validates the footprint, not the contention"
    ;;
pair)
    echo "== the co-runner's own demand, filtered to its kernel and profiled alone =="
    "$NCU" --metrics "$DRAM,lts__t_sector_hit_rate.pct,lts__t_bytes.sum" --csv \
           --log-file "$OUT/pair-$SIZE.csv" --launch-count 5 \
           --kernel-name "regex:.*sgemm.*" --clock-control none \
           ./runner "${BASE[@]}" --csv /dev/shm/ncu-pair "${GREEN[@]}" \
           > "$OUT/pair-$SIZE.err" 2>&1
    rm -f /dev/shm/ncu-pair*
    report "$OUT/pair-$SIZE.csv" "kernel, DRAM read %, write %, L2 hit"
    echo "   at a side of 512 this reads about 4.98 % of peak read and 4.0 % write"
    ;;
*)
    sed -n '3,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
    ;;
esac
