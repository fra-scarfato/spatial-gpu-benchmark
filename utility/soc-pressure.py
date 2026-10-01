import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import soc

NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")



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
        r, w = soc.series(rep)
        s = {"read": r, "write": w}
        rate = rate_from_log(rep[:-len(".nsys-rep")] + ".log")
        r_peak, w_peak = max(s["read"] or [float("nan")]), max(s["write"] or [float("nan")])
        # the mean over the samples that are doing something, which needs no window boundary and
        # is a fairer proxy for sustained pressure than the peak
        busy = lambda v: statistics.mean([x for x in v if x > 1.0]) if [x for x in v if x > 1.0] else 0.0
        r_busy, w_busy = busy(s["read"]), busy(s["write"])
        # each counter against its own calibration point, since the two scales differ
        read_gb = r_peak / soc.COPY_READ_PCT * soc.COPY_GB_S
        write_gb = w_peak / soc.COPY_WRITE_PCT * soc.COPY_GB_S
        total_gb = read_gb + write_gb
        # the same conversion applied to the sustained statistic, which is the one the
        # demand table uses: each direction against its own calibration point
        busy_gb = r_busy / soc.COPY_READ_PCT * soc.COPY_GB_S + w_busy / soc.COPY_WRITE_PCT * soc.COPY_GB_S
        per_inf = (total_gb * 1e9 / rate) if rate else float("nan")
        rows.append((name, r_peak, w_peak, total_gb, rate, per_inf, r_busy, w_busy, busy_gb))
        print("  %-34s %-8.2f %-8.2f %-10.1f %-10s %-9s %d   busy %.1f/%.1f" % (
            name, r_peak, w_peak, total_gb, "{:.0f}".format(rate) if rate else "?",
            "{:.2e}".format(per_inf) if rate else "?", len(s["read"]), r_busy, w_busy))

    dest = os.path.join(out_dir, "soc-pressure.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "dram_read_peak_pct", "dram_write_peak_pct",
                    "gb_per_s_peak_estimate", "inf_per_s", "bytes_per_inference",
                    "dram_read_busy_pct", "dram_write_busy_pct", "gb_per_s_sustained"])
        for name, r, w_, gb, rate, per, rb, wb, bg in rows:
            w.writerow([name, "{:.2f}".format(r), "{:.2f}".format(w_), "{:.2f}".format(gb),
                        "" if not rate else "{:.0f}".format(rate),
                        "" if not rate else "{:.0f}".format(per),
                        "{:.2f}".format(rb), "{:.2f}".format(wb), "{:.2f}".format(bg)])
    print("\n  wrote {}".format(dest))


if __name__ == "__main__":
    main()
