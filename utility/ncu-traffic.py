#!/usr/bin/env python3
"""Turn the ncu per-kernel DRAM counters into bytes per inference, per network and budget.

The runner's own rate times this is the traffic a network applies, with no device-wide differencing:
the reason ncu was worth the three attempts it took to invoke it.

    utility/ncu-traffic.py [/tmp/ncu-<label>-<pct>.csv ...]
    utility/ncu-traffic.py /mnt/nvme/f.scarfato/scratch/ncu

Three things happen here rather than in a generic parser:

  * the ncu CSV carries one row per (kernel, metric), and the first two lines are profiler chatter
    rather than quoted fields
  * the kernels per inference is not assumed: the kernel-name sequence repeats once per inference,
    so the period is found from the sequence itself, the same way the kernel traces were sliced
  * bytes per inference is the mean bytes per launch times that period, which is robust to where in
    the cycle the profiler started, unlike summing the first p launches
"""
import collections
import csv
import glob
import os
import re
import statistics
import sys

METRICS = ("dram__bytes_read.sum", "dram__bytes_write.sum")


def read_ncu(path):
    """[(kernel name, metric, value)] for one ncu CSV export."""
    rows = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith('"'):
                continue                      # ==PROF== chatter and the header
            fields = next(csv.reader([line]))
            if len(fields) < 15 or fields[12] not in METRICS:
                continue
            try:
                rows.append((fields[4], fields[12], float(fields[14])))
            except ValueError:
                # "n/a" is what a counter that could not be collected looks like, and it is a
                # finding rather than an absence: the profiler ran and got nothing.
                rows.append((fields[4], fields[12], None))
    return rows


def period(names, lo=10, hi=200):
    """Kernels per inference: the shift at which the kernel-name sequence agrees with itself."""
    n = min(len(names), 400)
    best, p_best = -1.0, 0
    for p in range(lo, min(hi, n - 1)):
        agree = sum(1 for i in range(n - p) if names[i] == names[i + p]) / (n - p)
        if agree > best:
            best, p_best = agree, p
    return p_best, best


def main():
    args = sys.argv[1:]
    dest = None
    if "--csv-out" in args:
        i = args.index("--csv-out")
        dest = args[i + 1]
        del args[i:i + 2]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "*.csv"))) if os.path.isdir(a) else [a]
    if not files:
        files = sorted(glob.glob("/tmp/ncu-*.csv"))
    if not files:
        sys.exit("  no ncu CSVs found")

    print("\n  %-26s %-6s %-7s %-12s %-12s %s" % (
        "configuration", "kern", "cycle", "read/inf", "write/inf", "per inference"))
    print("  " + "-" * 84)
    out = []
    for path in files:
        rows = read_ncu(path)
        if not rows:
            print("  %-26s no metric rows" % os.path.basename(path))
            continue
        if all(r[2] is None for r in rows):
            print("  %-30s ran, but every one of %d metric values is n/a: the counters were "
                  "not collected" % (os.path.basename(path), len(rows)))
            continue
        names = [r[0] for r in rows if r[1] == METRICS[0]]
        kpi, agree = period(names)
        total = collections.Counter()
        for _, metric, value in rows:
            if value is None:
                continue
            total[metric] += value
        launched = len(names)
        read_b = total[METRICS[0]] / launched * kpi
        write_b = total[METRICS[1]] / launched * kpi
        m = re.match(r"ncu-(.+)-(\d+)\.csv$", os.path.basename(path))
        label, pct = (m.group(1), m.group(2)) if m else (os.path.basename(path), "?")
        out.append((label, int(pct), kpi, read_b, write_b))
        print("  %-26s %-6d %-7s %-12s %-12s %.1f MB" % (
            "{} at {}%".format(label, pct), launched,
            "{}{}".format(kpi, "" if agree > 0.9 else "?"),
            "{:.2f} MB".format(read_b / 1e6), "{:.2f} MB".format(write_b / 1e6),
            (read_b + write_b) / 1e6))

    if dest:
        with open(dest, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["network", "sm_percent", "kernels_per_inference",
                        "dram_read_bytes_per_inference", "dram_write_bytes_per_inference"])
            for label, pct, kpi, r, wr in out:
                w.writerow([label, pct, kpi, "{:.0f}".format(r), "{:.0f}".format(wr)])
        print("\n  wrote {}".format(dest))


if __name__ == "__main__":
    main()
