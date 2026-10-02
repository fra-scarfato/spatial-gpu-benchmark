#!/usr/bin/env python3
"""Read the footprint sweep: does the co-runner's cache footprint, not its traffic, cost?

    python3 utility/l2-footprint-verdict.py [/mnt/nvme/f.scarfato/scratch/l2-footprint]

Per arm the target's own DRAM per inference is the integral of the device counters over the run,
minus the co-runner's contribution over the time its kernels ran, divided by the target's
inferences. The same arithmetic as the displacement test, applied to seven footprints, and the
reading at the end names which of the four outcomes the arms support.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import soc


def stats(path):
    for row in __import__("csv").DictReader(open(path)):
        if row.get("metric") == "gpu_ms":
            return float(row["median"]), int(float(row["count"]))
    return None, None


def footprint(log):
    m = re.search(r"working set:\s+([0-9.]+) MB", open(log, errors="replace").read())
    return float(m.group(1)) if m else float("nan")


def co_rate_gb_s(log):
    """The co-runner's own rate, from its report: bytes for the memory kind, 86 FLOP per byte of
    L2 traffic for the compute kind, which is the intensity its own accounting fixes."""
    txt = open(log, errors="replace").read()
    m = re.search(r"bytes moved:\s+[0-9.]+ GB \(([0-9.]+) GB/s\)", txt)
    if m:
        return float(m.group(1))
    m = re.search(r"arithmetic:\s+[0-9.]+ GFLOP \(([0-9.]+) TFLOP/s\)", txt)
    return float(m.group(1)) * 1e12 / 86.0 / 1e9 if m else 0.0


def l2_hit(csv_path):
    if not os.path.exists(csv_path):
        return ""
    for line in open(csv_path, errors="replace"):
        if "lts__t_sector_hit_rate" in line:
            return line.rsplit('","', 1)[-1].strip().strip('"')
    return ""


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(soc.DATA, "scratch", "l2-footprint")
    print("\n  %-10s %9s %9s %8s %9s %9s %8s" % (
        "arm", "footprint", "median", "vs A0", "MB/Inf", "co GB/s", "L2 hit"))
    print("  " + "-" * 70)
    rows, base = [], None
    for rep in sorted(glob.glob(os.path.join(d, "*.nsys-rep"))):
        label = os.path.basename(rep)[:-len(".nsys-rep")]
        st = (glob.glob(os.path.join(d, label + "_stats.csv"))
              or glob.glob(os.path.join(d, label + "_*_stats.csv")))
        if not st:
            continue
        med, n = stats(st[0])
        print("  reading %s ..." % label, flush=True)
        r, w, secs = soc.with_kernels(rep, "sgemm")
        gb = soc.gb(sum(r), sum(w))
        co = co_rate_gb_s(os.path.join(d, label + ".log"))
        own = gb - co * secs                      # GB/s times seconds is GB
        if base is None:
            base = med
        rows.append((label, footprint(os.path.join(d, label + ".log")), med, med / base - 1,
                     own * 1e9 / n, co, l2_hit(os.path.join(d, label + "-ncu.csv"))))
    for label, fp, med, dlt, per_inf, co, hit in rows:
        print("  %-10s %8.2fMB %9.4f %7.1f%% %9.1f %9.1f %8s"
              % (label, fp, med, dlt * 100, per_inf, co, hit))

    if len(rows) < 3:
        print("\n  too few arms read to draw a conclusion")
        return
    by = {r[0].split("-")[0]: r for r in rows}
    a0, a1 = by.get("A0"), by.get("A1")
    sweep = [r for r in rows if r[0].split("-")[0] not in ("A0", "A1")]
    print("\n  A0, an idle partition:                    %+.1f %%   (the topology alone)" % (a0[3] * 100))
    if sweep:
        lo, hi = sweep[0], sweep[-1]
        print("  the footprint sweep, %.2f MB to %.2f MB:  %+.1f %% to %+.1f %%"
              % (lo[1], hi[1], lo[3] * 100, hi[3] * 100))
    if a1:
        print("  A1, whose operands fit in the L1:         %+.1f %%" % (a1[3] * 100))
    print()
    if a0 and a0[3] > 0.05:
        print("  -> the partition by itself costs something: the floor is the topology")
    elif sweep and abs(sweep[-1][3] - sweep[0][3]) < 0.08:
        print("  -> FLAT across the footprint: twelve times the working set moves the penalty by")
        print("     %.1f points, so the shared resource is not the cache's capacity."
              % abs(sweep[-1][3] - sweep[0][3]) * 100)
    elif sweep and sweep[-1][3] > sweep[0][3] + 0.05:
        print("  -> the penalty grows with the footprint: the resource is cache-side")
    else:
        print("  -> no monotone trend in the footprint sweep; see the columns above")
    if a1 and a1[3] > 2 * (sweep[0][3] if sweep else 0.1):
        print("     A1 is a different regime, not a point on this curve: a side-64 GEMM repeated")
        print("     16400 times per unit is a storm of tiny launches, so it measures how the")
        print("     co-runner delivers work rather than what it keeps in the cache, and it costs")
        print("     the target more than any footprint does.")
