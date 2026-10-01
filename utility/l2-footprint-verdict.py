#!/usr/bin/env python3
"""Read the footprint sweep: does the co-runner's cache footprint, and not its traffic, cost?

    python3 utility/l2-footprint-verdict.py [/mnt/nvme/f.scarfato/scratch/l2-footprint]

Per arm the target's own DRAM per inference is the integral of the device counters over the run,
minus the co-runner's contribution over the time its kernels ran, divided by the target's
inferences. The same arithmetic as the displacement test, applied to seven footprints.
"""
import csv, glob, os, re, sqlite3, subprocess, sys

NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")
DATA = os.environ.get("DATA", "/mnt/nvme/f.scarfato")
COPY_READ_PCT, COPY_WRITE_PCT, COPY_GB_S = 48.0, 83.0, 91.5
FREQ_HZ = 10000.0

def series(rep):
    cache = os.path.join(DATA, "scratch", "soc-series",
                         os.path.basename(rep)[:-len(".nsys-rep")] + ".csv")
    if os.path.exists(cache):
        rows = csv.reader(open(cache)); next(rows, None)
        r, w = [], []
        for row in rows:
            if len(row) >= 2:
                r.append(float(row[0])); w.append(float(row[1]))
        return r, w
    return None, None

def kernels(rep, pattern="sgemm"):
    """The co-runner's kernel seconds and its own rate, from a fresh export."""
    os.makedirs(os.path.join(DATA, "scratch", "soc-sqlite"), exist_ok=True)
    db = os.path.join(DATA, "scratch", "soc-sqlite",
                      os.path.basename(rep)[:-len(".nsys-rep")] + "-k.sqlite")
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                        "--output", db, rep], check=True, capture_output=True)
    con = sqlite3.connect(db)
    ids = dict(con.execute("SELECT id, value FROM StringIds"))
    secs = 0
    for nm, a, b in con.execute("SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if pattern in ids.get(nm, str(nm)):
            secs += (b - a)
    con.close()
    return secs / 1e9

def stats(path):
    for row in csv.DictReader(open(path)):
        if row.get("metric") == "gpu_ms":
            return float(row["median"]), int(float(row["count"]))
    return None, None

def footprint(log):
    txt = open(log, errors="replace").read()
    m = re.search(r"working set:\s+([0-9.]+) MB", txt)
    return float(m.group(1)) if m else float("nan")

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "scratch", "l2-footprint")
    print("\n  %-10s %9s %9s %9s %9s %9s %9s" % (
        "arm", "footprint", "median", "vs A0", "MB/Inf", "co GB/s", "L2 hit"))
    print("  " + "-" * 72)
    base = None
    rows = []
    for rep in sorted(glob.glob(os.path.join(d, "*.nsys-rep"))):
        label = os.path.basename(rep)[:-len(".nsys-rep")]
        st = (glob.glob(os.path.join(d, label + "_stats.csv"))
              or glob.glob(os.path.join(d, label + "_*_stats.csv")))
        if not st:
            continue
        med, n = stats(st[0])
        r, w = series(rep)
        if r is None:
            continue
        gb = (sum(r) / COPY_READ_PCT + sum(w) / COPY_WRITE_PCT) * COPY_GB_S / FREQ_HZ
        secs = kernels(rep)
        co_gb_s = 0.0
        # the co-runner's own rate, from its report in the log
        lg = open(os.path.join(d, label + ".log"), errors="replace").read()
        m = re.search(r"bytes moved:\s+[0-9.]+ GB \(([0-9.]+) GB/s\)", lg)
        if m:
            co_gb_s = float(m.group(1))
        else:
            m = re.search(r"arithmetic:\s+[0-9.]+ GFLOP \(([0-9.]+) TFLOP/s\)", lg)
            if m:                       # the compute kind: 86 FLOP per byte of L2 traffic
                co_gb_s = float(m.group(1)) * 1e12 / 86.0 / 1e9
        own = gb - co_gb_s * secs          # GB/s times seconds is GB
        per_inf = own * 1e9 / n
        if base is None:
            base = med
        hit = ""
        ncu = os.path.join(d, label + "-ncu.csv")
        if os.path.exists(ncu):
            for line in open(ncu, errors="replace"):
                if "lts__t_sector_hit_rate" in line:
                    hit = line.rsplit('","', 1)[-1].strip().strip('"'); break
        rows.append((label, footprint(os.path.join(d, label + ".log")), med, med / base - 1,
                     per_inf, co_gb_s, hit))
    for label, fp, med, dlt, per_inf, co, hit in rows:
        print("  %-10s %8.2f %9.4f %8.1f%% %9.1f %9.1f %9s"
              % (label, fp, med, dlt * 100, per_inf, co, hit))


if __name__ == "__main__":
    main()
