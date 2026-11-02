#!/usr/bin/env python3
"""models.py - p99 and median error (points) per pair of the 8 + 8 stress models, side by side.

    python3 analysis/explore/models.py

Reads $DATA/bench/aggregate/stress-comparison-<model>.csv, written by stress.py compare <model>,
for ON/OFF (onoff88), uniform (uniform88), smooth (smooth88) and smooth topped by full speed
(smoothx88), the counter models (counters-dram, -l2, -ref, counter-model.py) and the inference-window
model (infwin88, infwin88d, infwin88p, infwin88pd, steadywin88p, steadywin88pd, infwin-model.py) and the max of Smooth, Temporal and one counter variant, raw
(max-dram, -ref, -l2) and bias-corrected (maxc-*, ensemble.py); missing models are
skipped. The reporter models predict from the reporter p99 for both statistics,
the thesis model (err_p99_p99, err_median_p99)."""
import csv, os
D = os.path.join(os.environ.get("DATA", "/mnt/nvme/f.scarfato"), "bench/aggregate/stress-comparison-%s.csv")
M = [m for m in ("onoff88", "uniform88", "smooth88", "smoothx88", "counters-dram", "counters-l2", "counters-ref", "infwin88", "infwin88d", "infwin88p", "infwin88pd", "steadywin88p", "steadywin88pd", "max-dram", "max-ref", "max-l2", "maxc-dram", "maxc-ref", "maxc-l2") if os.path.exists(D % m)]
for q in ("p99", "median"):
    err = {m: {(r["target"], r["corunner"]): float(r["err_%s_p99_pts" % q]) for r in csv.DictReader(open(D % m))}
           for m in M}
    print("\n%s error (predicted - measured, points)" % q)
    print("%-10s %-10s" % ("target", "co-runner") + "".join("%14s" % m for m in M))
    for k in sorted(err[M[0]]):
        print("%-10s %-10s" % k + "".join("%14s" % ("%+.1f" % err[m][k] if k in err[m] else "-") for m in M))
    for name, f in (("mean |e|", lambda v: sum(map(abs, v)) / len(v)), ("max |e|", lambda v: max(map(abs, v))),
                    ("bias", lambda v: sum(v) / len(v)), ("under", lambda v: sum(x < 0 for x in v))):
        print("%-21s" % name + "".join("%14.1f" % f(list(err[m].values())) for m in M))
