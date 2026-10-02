"""The SoC counters: one place for the export, the series and the calibration.

Three tools had grown their own copy of this: the sqlite export, the read of SOC_METRICS by
position because the export names no columns, the counter indices resolved by name, the per-trace
series cache, and the conversion that turns a percentage into GB/s. The copy calibration is one
measurement - a D2D copy that ran at 91.5 GB/s while the counters read 48 % of peak read and 83 %
of peak write - so each direction is converted against its own point.
"""
import csv
import os
import sqlite3
import subprocess

NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")
DATA = os.environ.get("DATA", "/mnt/nvme/f.scarfato")
FREQ_HZ = 10000.0
COPY_READ_PCT, COPY_WRITE_PCT, COPY_GB_S = 48.0, 83.0, 91.5

SERIES_DIR = os.path.join(DATA, "scratch", "soc-series")
SQLITE_DIR = os.path.join(DATA, "scratch", "soc-sqlite")


def _export(rep, suffix=""):
    """Export a trace to sqlite, into a directory this user owns, cached on disk.

    Not beside the trace: those come from root-owned runs, so the directory is root's and reading
    the report is possible while creating anything next to it is not.
    """
    os.makedirs(SQLITE_DIR, exist_ok=True)
    stem = os.path.basename(rep)[:-len(".nsys-rep")]
    db = os.path.join(SQLITE_DIR, stem + suffix + ".sqlite")
    # An export costs minutes and the database holds everything the trace has, so a file already
    # there under the other suffix is moved into place rather than exported again. Two suffixes
    # existed because two functions each exported the same trace; this makes the cost disappear
    # without pretending the files are different.
    other = os.path.join(SQLITE_DIR, stem + ("-k" if not suffix else "") + ".sqlite")
    if not os.path.exists(db) and os.path.exists(other):
        os.rename(other, db)
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                        "--output", db, rep], check=True, capture_output=True)
    return db


def _counter_names(con):
    """index -> counter name, from TARGET_INFO_SOC_METRICS rather than assumed."""
    names = {}
    try:
        for row in con.execute("SELECT * FROM TARGET_INFO_SOC_METRICS"):
            for v in row:
                if isinstance(v, str) and "DRAM" in v:
                    for x in row:
                        if isinstance(x, int):
                            names[x] = v
                            break
    except sqlite3.Error:
        pass
    return names


def series(rep):
    """The DRAM read and write percentage series of a trace, from the cache or a fresh export."""
    os.makedirs(SERIES_DIR, exist_ok=True)
    cache = os.path.join(SERIES_DIR, os.path.basename(rep)[:-len(".nsys-rep")] + ".csv")
    if os.path.exists(cache):
        rows = csv.reader(open(cache))
        next(rows, None)                      # the header
        r, w = [], []
        for row in rows:
            if len(row) >= 2:
                r.append(float(row[0]))
                w.append(float(row[1]))
        return r, w
    db = _export(rep)
    con = sqlite3.connect(db)
    names = _counter_names(con)
    r, w = [], []
    for row in con.execute("SELECT * FROM SOC_METRICS"):
        if len(row) < 5:
            continue
        idx, val = row[3], row[4]
        nm = names.get(idx) or {6: "DRAM Read Throughput", 7: "DRAM Write Throughput"}.get(idx)
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


def gb(read_pct_sum, write_pct_sum):
    """Samples of percent, summed at FREQ_HZ, to GB. Each direction against its own point.

    sum(p)/100 is the fraction of peak, times 1/10000 s is peak-seconds, times the peak rate. The
    peak rate is the copy's 91.5 GB/s divided by the fraction that copy registered.
    """
    return (read_pct_sum / COPY_READ_PCT + write_pct_sum / COPY_WRITE_PCT) * COPY_GB_S / FREQ_HZ


def busy_mean(values, threshold=1.0):
    """The mean over the samples above a threshold, which drops the idle gaps between inferences."""
    v = [x for x in values if x > threshold]
    return sum(v) / len(v) if v else 0.0


def with_kernels(rep, pattern="sgemm"):
    """The counter series and the kernel seconds from ONE export.

    Two functions each exporting the same trace was fourteen exports for seven arms, and the second
    one looked like a hang: several minutes with no output. The counters and the kernel table live in
    the same database, so they are read together.
    """
    r, w = series(rep)
    db = _export(rep)
    con = sqlite3.connect(db)
    ids = dict(con.execute("SELECT id, value FROM StringIds"))
    total = 0
    for nm, a, b in con.execute("SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if pattern in ids.get(nm, str(nm)):
            total += (b - a)
    con.close()
    return r, w, total / 1e9


def kernel_seconds(rep, pattern):
    """How long the kernels whose name matches spent executing, in seconds."""
    db = _export(rep)
    con = sqlite3.connect(db)
    ids = dict(con.execute("SELECT id, value FROM StringIds"))
    total = 0
    for nm, a, b in con.execute("SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        if pattern in ids.get(nm, str(nm)):
            total += (b - a)
    con.close()
    return total / 1e9
