#!/usr/bin/env python3
"""Read the eviction test: does a co-runner make the target fetch more bytes per inference?

    python3 utility/eviction-verdict.py [/mnt/nvme/f.scarfato/scratch/eviction]

The question the memory subsection leaves open. A co-runner that asks the DRAM for little still costs
the target about ten points of median, and the two candidates predict different things:

    eviction      the co-runner's lines displace the target's, so the target refetches what the
                  cache used to hold: MORE BYTES PER INFERENCE, and the device's DRAM rate does not
                  fall in proportion as the target slows down
    arbitration   the lines stay and each access waits longer: THE SAME BYTES PER INFERENCE, fetched
                  more slowly

The quantity is therefore bytes per inference, which needs no window and no attribution:

    bytes per inference = (integral of the DRAM counters over the run) / (inferences in the run)

The integral comes from the trace, where the counters are sampled at 10 kHz and each direction is
converted against its own copy calibration. The inference count comes from the runner's own stats
file, so neither the fourteen-second clock offset nor the nsys startup enters the arithmetic.

One correction, and it is why the compute participant is the only usable co-runner here: the counters
are device-wide and see the participant's traffic too. Its demand was measured at 13.9 GB/s by the
profiler, so the target's own bytes per inference lies between the total and the total minus a
participant running at full rate for the whole run. Both bounds are printed; the verdict is read from
the bound that still holds.
"""
import csv
import glob
import os
import re
import sqlite3
import subprocess
import sys

NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")
DATA = os.environ.get("DATA", "/mnt/nvme/f.scarfato")
COPY_READ_PCT, COPY_WRITE_PCT, COPY_GB_S = 48.0, 83.0, 91.5
COMPUTE_GB_S = 13.9          # the compute participant's own DRAM demand, from the profiler
FREQ_HZ = 10000.0

series_dir = os.path.join(DATA, "scratch", "soc-series")
sqlite_dir = os.path.join(DATA, "scratch", "soc-sqlite")


def load_series(rep):
    """The read and write counter series of one trace, from the cache or a fresh export."""
    os.makedirs(series_dir, exist_ok=True)
    cache = os.path.join(series_dir, os.path.basename(rep)[:-len(".nsys-rep")] + ".csv")
    if os.path.exists(cache):
        rows = csv.reader(open(cache))
        next(rows, None)
        r, w = [], []
        for row in rows:
            if len(row) >= 2:
                r.append(float(row[0]))
                w.append(float(row[1]))
        return r, w
    os.makedirs(sqlite_dir, exist_ok=True)
    db = os.path.join(sqlite_dir, os.path.basename(rep)[:-len(".nsys-rep")] + ".sqlite")
    subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                    "--output", db, rep], check=True, capture_output=True)
    con = sqlite3.connect(db)
    names = {}
    for row in con.execute("SELECT * FROM TARGET_INFO_SOC_METRICS"):
        for i, v in enumerate(row):
            if isinstance(v, str) and "DRAM" in v:
                for x in row:
                    if isinstance(x, int):
                        names[x] = v
                        break
    r, w = [], []
    for row in con.execute("SELECT * FROM SOC_METRICS"):
        if len(row) < 5:
            continue
        idx, val = row[3], row[4]
        nm = names.get(idx)
        if nm is None:
            nm = {6: "DRAM Read Throughput", 7: "DRAM Write Throughput"}.get(idx)
        if nm is None:
            continue
        (r if "Read" in nm else w).append(val)
    con.close()
    os.remove(db)
    with open(cache, "w") as fh:
        fh.write("read_pct,write_pct\n")
        for a, b in zip(r, w):
            fh.write("%.4f,%.4f\n" % (a, b))
    return r, w


def count_of(stats_path):
    """Inferences and seconds from the runner's own statistics file."""
    for row in csv.DictReader(open(stats_path)):
        if row.get("metric") == "gpu_ms":
            return int(float(row["count"])), float(row["duration_s"])
    return None, None


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "scratch", "eviction")
    reps = sorted(glob.glob(os.path.join(d, "*.nsys-rep")))
    if not reps:
        sys.exit("  no traces in " + d)

    print("\n  %-18s %8s %9s %12s %14s %14s" % (
        "condition", "samples", "count", "DRAM GB", "B per inference", "less the co-runner"))
    print("  " + "-" * 84)
    out = {}
    for rep in reps:
        label = os.path.basename(rep)[:-len(".nsys-rep")]
        r, w = load_series(rep)
        stats = [p for p in glob.glob(os.path.join(d, label + "-*_stats.csv"))
                 or glob.glob(os.path.join(d, label + "_stats.csv"))]
        if not stats:
            print("  %-18s %8d  no stats file" % (label, len(r)))
            continue
        n, dur = count_of(sorted(stats)[0])
        # the integral of the counters: each direction against its own calibration, in GB
        gb = (sum(r) / COPY_READ_PCT + sum(w) / COPY_WRITE_PCT) * COPY_GB_S / FREQ_HZ / 100.0
        per_inf = gb * 1e9 / n
        # the co-runner's contribution at full rate for the whole run, if any
        co = COMPUTE_GB_S * dur if "icompute" in label else 0.0
        per_inf_less = (gb - co) * 1e9 / n
        out[label] = (per_inf, per_inf_less)
        print("  %-18s %8d %9d %12.2f %14.0f %14.0f"
              % (label, len(r), n, gb, per_inf, per_inf_less))

    alone = next((v for k, v in out.items() if k.endswith("alone")), None)
    pair = next((v for k, v in out.items() if "icompute" in k), None)
    if not alone or not pair:
        sys.exit("\n  need one 'alone' and one 'icompute' condition to compare")

    hi, lo = pair[0] / alone[0], pair[1] / alone[1]
    print("\n  the pair over the alone condition, bytes per inference:")
    print("    upper bound (the co-runner left in)  %.3f" % hi)
    print("    lower bound (its full rate removed)  %.3f" % lo)
    if lo > 1.05:
        print("    -> EVICTION: even with the co-runner's own traffic removed the target fetches"
              "\n       more bytes per inference than it does alone")
    elif hi < 1.05:
        print("    -> ARBITRATION: the target fetches the same bytes per inference, more slowly"
              "\n       (even the upper bound is within 5 per cent of the isolated run)")
    else:
        print("    -> not separated by these two runs: the bounds straddle the question")


if __name__ == "__main__":
    main()
