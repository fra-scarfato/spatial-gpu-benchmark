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
        r, w = soc.series(rep)
        if not r:
            continue
        gb = soc.gb(sum(r), sum(w))
        secs = soc.kernel_seconds(rep, "sgemm") if "idle" not in label else 0.0
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
    n0, n1, nlast = rows[0], rows[1], rows[-1]
    d0, d1, dlast = n0[3], n1[3], nlast[3]
    print("\n  A0 %s: %.1f%%; A1 (operands inside L1) %s: %.1f%%; %s (past the cache) %s: %.1f%%"
          % (n0[0], d0 * 100, n1[0], d1 * 100, nlast[0], dlast * 100))
    if d0 > 0.05:
        print("  -> the second partition costs something even idle: the floor is the topology")
    elif d1 > 0.08 and abs(dlast - d1) < 0.05:
        print("  -> the floor is there in A1, where almost nothing reaches the L2, and does not grow:")
        print("     a co-runner occupying SMs costs this by existing, not by cache traffic")
    elif dlast > d1 + 0.05:
        print("  -> the penalty grows with the footprint: the shared resource is cache-side")
    else:
        print("  -> flat across the footprints: the cache's footprint is not the mechanism")


if __name__ == "__main__":
    main()
