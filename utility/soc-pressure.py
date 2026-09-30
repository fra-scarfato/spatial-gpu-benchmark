#!/usr/bin/env python3
"""The DRAM pressure a network applies at a given SM budget, from the SoC counters.

    utility/soc-pressure.py [<dir of .nsys-rep>]

Exports each trace to sqlite if it has not been already, reads the DRAM read and write counters
while that network was the only client running, converts them to GB/s, and divides by the rate the
runner reported in the run's own log to get bytes per inference — which is what ncu would have given
had the board allowed its counters to be collected.

Two things about the counters themselves:

  * the sqlite export reports `None` for every column name, so `SOC_METRICS` is read by position:
    (start_ns, end_ns, domain, index, value), and the counter names come from
    `TARGET_INFO_SOC_METRICS`, which is how the DRAM rows are found rather than assumed
  * read and write percentages are on different scales, so each is converted against its own
    calibration point. The one measured here is a D2D copy: 48 % of peak read and 83 % of peak write
    at 91.5 GB/s counted, from `devprobe` and the copy trace in the scratch directory.

Peak and p95 are reported rather than a mean over a window, because the windows of the CUDA and the
SoC clocks do not line up - they are about 14 s apart in the export - so anything that needs a window
boundary would need that offset to be right, and neither the peak nor the p95 does.
"""
import csv
import glob
import os
import re
import sqlite3
import statistics
import subprocess
import sys

NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")
COPY_READ_PCT, COPY_WRITE_PCT, COPY_GB_S = 48.0, 83.0, 91.5


def export(rep):
    """sqlite in a directory this user can write, exporting only if it is not there yet.

    Not beside the trace: those were written by a root-owned run, so the directory is root's and
    reading the trace is possible while creating anything next to it is not.
    """
    tmp = os.environ.get("SOC_SQLITE_DIR",
                         os.path.join(os.environ.get("DATA", "/mnt/nvme/f.scarfato"),
                                      "scratch/soc-sqlite"))
    os.makedirs(tmp, exist_ok=True)
    db = os.path.join(tmp, os.path.basename(rep)[:-len(".nsys-rep")] + ".sqlite")
    if not os.path.exists(db):
        r = subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                            "--output", db, rep], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(db):
            sys.exit("  export failed for {}\n  {}".format(rep, (r.stderr or r.stdout)[-300:]))
    return db


def cached_series(rep):
    """The series from a small CSV if it is there, otherwise from a fresh export."""
    cache_dir = os.environ.get("SOC_SERIES_DIR",
                              os.path.join(os.environ.get("DATA", "/mnt/nvme/f.scarfato"),
                                           "scratch/soc-series"))
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, os.path.basename(rep)[:-len(".nsys-rep")] + ".csv")
    if os.path.exists(cache):
        rows = csv.reader(open(cache))
        next(rows, None)                      # the header line, read_pct,write_pct
        r, w = [], []
        for row in rows:
            if len(row) >= 2:
                r.append(float(row[0]))
                w.append(float(row[1]))
        return {"read": r, "write": w}
    db = export(rep)
    series = dram_series(db)
    os.remove(db)
    with open(cache, "w", newline="") as fh:
        fh.write("read_pct,write_pct\n")
        for a, b in zip(series["read"], series["write"]):
            fh.write("{:.4f},{:.4f}\n".format(a, b))
    return series


def dram_series(db):
    """{'read': [values], 'write': [values]} from the SoC counters, found by name."""
    con = sqlite3.connect(db)
    names = {}
    try:
        for row in con.execute("SELECT * FROM TARGET_INFO_SOC_METRICS"):
            for i, v in enumerate(row):
                if isinstance(v, str) and ("DRAM" in v or "dram" in v):
                    # the index is the integer column nearest it
                    for w in row:
                        if isinstance(w, int):
                            names[w] = v
                            break
    except sqlite3.Error:
        pass
    out = {"read": [], "write": []}
    for row in con.execute("SELECT * FROM SOC_METRICS"):
        # by position: start_ns, end_ns, domain, index, value
        if len(row) < 5:
            continue
        index, value = row[3], row[4]
        name = names.get(index)
        if name is None:
            # no name table: fall back to the indices established on this board
            if index == 6:
                name = "DRAM Read Throughput"
            elif index == 7:
                name = "DRAM Write Throughput"
            else:
                continue
        if "Read" in name:
            out["read"].append(value)
        elif "Write" in name:
            out["write"].append(value)
    con.close()
    return out


def rate_from_log(log):
    """The inferences per second the runner reported, from its own log."""
    try:
        hits = re.findall(r"([0-9.]+) inf/s", open(log, errors="replace").read())
    except OSError:
        return None
    return float(hits[-1]) if hits else None


def pct(values, q):
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1)))] if v else float("nan")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/f.scarfato/scratch/soc-budget"
    reps = sorted(glob.glob(os.path.join(d, "*.nsys-rep")))
    if not reps:
        sys.exit("  no traces in " + d)
    out_dir = os.environ.get("DATA", "/mnt/nvme/f.scarfato") + "/bench"
    rows = []
    print("\n  %-34s %-8s %-8s %-10s %-10s %-9s %s" % (
        "configuration", "read %", "write %", "GB/s", "inf/s", "B/inf", "samples"))
    print("  " + "-" * 96)
    for rep in reps:
        name = os.path.basename(rep)[:-len(".nsys-rep")]
        s = cached_series(rep)
        rate = rate_from_log(rep[:-len(".nsys-rep")] + ".log")
        r_peak, w_peak = max(s["read"] or [float("nan")]), max(s["write"] or [float("nan")])
        # the mean over the samples that are doing something, which needs no window boundary and
        # is a fairer proxy for sustained pressure than the peak
        busy = lambda v: statistics.mean([x for x in v if x > 1.0]) if [x for x in v if x > 1.0] else 0.0
        r_busy, w_busy = busy(s["read"]), busy(s["write"])
        # each counter against its own calibration point, since the two scales differ
        read_gb = r_peak / COPY_READ_PCT * COPY_GB_S
        write_gb = w_peak / COPY_WRITE_PCT * COPY_GB_S
        total_gb = read_gb + write_gb
        per_inf = (total_gb * 1e9 / rate) if rate else float("nan")
        rows.append((name, r_peak, w_peak, total_gb, rate, per_inf, r_busy, w_busy))
        print("  %-34s %-8.2f %-8.2f %-10.1f %-10s %-9s %d   busy %.1f/%.1f" % (
            name, r_peak, w_peak, total_gb, "{:.0f}".format(rate) if rate else "?",
            "{:.2e}".format(per_inf) if rate else "?", len(s["read"]), r_busy, w_busy))

    dest = os.path.join(out_dir, "soc-pressure.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "dram_read_peak_pct", "dram_write_peak_pct",
                    "gb_per_s_peak_estimate", "inf_per_s", "bytes_per_inference",
                    "dram_read_busy_pct", "dram_write_busy_pct"])
        for name, r, w_, gb, rate, per, rb, wb in rows:
            w.writerow([name, "{:.2f}".format(r), "{:.2f}".format(w_), "{:.2f}".format(gb),
                        "" if not rate else "{:.0f}".format(rate),
                        "" if not rate else "{:.0f}".format(per),
                        "{:.2f}".format(rb), "{:.2f}".format(wb)])
    print("\n  wrote {}".format(dest))


if __name__ == "__main__":
    main()
