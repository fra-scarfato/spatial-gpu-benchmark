#!/usr/bin/env python3
"""peak-test.py - the peak-length test (2026-10-12): SAXPY at duty 10 %, ON parts of 0.01-3 ms.

    python3 analysis/explore/peak-test.py

Same mean bandwidth (10-12 GB/s) and height, only the length of a peak changes (peak-T0010..T3000 in
stress-onoff and calibration-onoff). Prints (1) the reporter's view of each configuration, (2) each
target's p99 / median slowdown against peak length, (3) today's model (reporter p99) and the
average-over-inference model (avg-model.py) against the measured slowdowns.
Results: the p99 grows with the peak until a peak outlasts an inference (MobileNet 11 -> 52 %), the
median stays flat; today's model 12.2 / 28.8 (p99), 19.6 / 34.1 (median); H 6.3 / 16.8, 2.5 / 4.7.
"""
import csv, os, statistics as st, sys
import numpy as np
from importlib import import_module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # analysis/
M = import_module("avg-model")                  # same directory; its windows() and the stress.py setup
S, A = M.S, M.A
AGG = M.AGG
CFG = ["peak-T0010", "peak-T0100", "peak-T0300", "peak-T1000", "peak-T3000"]

cal = sorted(csv.DictReader(open(os.path.join(A.FROZEN_DIR, "stress-calibration-smooth88.csv"))), key=lambda r: float(r["level"]))
lv = [float(r["level"]) for r in cal]
ns_med = [float(r["median"]) for r in cal]
ns_p99 = [float(r["p99"]) for r in cal]
C = M.curves("smooth88")
data = {k: v for k, v in A.load(S.BENCH).items() if k[0] == "exp23"}
T = {t: A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "median")[0] for t in S.NETS}

print("1. reporter (ns): median p99 p99.9 | mean GB/s on the window ruler | windows above smooth-100 median")
W = {c: M.windows(c) for c in CFG}
for c in CFG:
    a = np.concatenate(W[c])
    print("  %s  %6.1f %6.1f %6.1f | %5.1f | %4.1f %%" % (c, st.median(np.median(v) for v in W[c]),
          st.median(np.percentile(v, 99) for v in W[c]), st.median(np.percentile(v, 99.9) for v in W[c]),
          np.interp(a, ns_med, lv).mean(), 100 * (a > ns_med[-1]).mean()))

print("\n2. measured slowdown, p99 / median (%), against the target alone")
meas = {}
for t in S.NETS:
    al99 = A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "p99")[0]
    row = []
    for c in ["spin"] + CFG + ["smooth-D025"]:
        p = A.aggr(data, "exp23", t + "-" + c, t, "gpu_ms", "p99")[0] / al99
        m = A.aggr(data, "exp23", t + "-" + c, t, "gpu_ms", "median")[0] / T[t]
        meas[(t, c)] = (p, m)
        row.append("%5.1f/%4.1f" % ((p - 1) * 100, (m - 1) * 100))
    print("  %-9s (T %5.2f ms)  spin %s | %s | smooth25 %s" % (t, T[t], row[0], "  ".join(row[1:-1]), row[-1]))

print("\n3. predictions: error in points, p99 / median (pred - measured)")
for t in S.NETS:
    pts = C[t]
    L, s99, smd = [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]
    out_now, out_h = [], []
    for c in CFG:
        m99, mm = meas[(t, c)]
        leq = S.invert(st.median(np.percentile(v, 99) for v in W[c]), lv, ns_p99)[0]
        now = (np.interp(leq, L, s99) - m99, np.interp(leq, L, smd) - mm)
        f99, fmd, _ = M.felt(W[c], lv, ns_med, T[t])
        h = (np.interp(f99, L, s99) - m99, np.interp(fmd, L, smd) - mm)
        out_now.append("%+5.1f/%+5.1f" % (now[0] * 100, now[1] * 100))
        out_h.append("%+5.1f/%+5.1f (felt %5.1f)" % (h[0] * 100, h[1] * 100, f99))
    print("  %-9s today: %s" % (t, "  ".join(out_now)))
    print("  %-9s H:     %s" % ("", "  ".join(out_h)))
