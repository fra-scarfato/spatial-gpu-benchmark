#!/usr/bin/env python3
"""avg-model.py - the average-over-inference stress model (hypothesis H), offline, on the 8 + 8 pairs.

    python3 utility/analysis/avg-model.py

H: a target inference feels the co-runner traffic averaged over its own length. Each reporter window
beside a network (networks-8sm) is mapped to GB/s on a per-window ruler (the median window beside spin
and the smooth stressor at each level), the GB/s are averaged, time-weighted, over a sliding stretch as
long as one target inference alone, and the p99 / median of those averages are read on the target's
smooth sensitivity curves. Reads only; writes nothing.
  variant A: smooth ruler and curves (clipped at the smooth top, ~115 GB/s)
  variant B: the same topped by the plain kernel back to back (ON/OFF 100 %, ~135 GB/s)
Results 2026-10-12: p99 4.3 / 7.4 (A), 4.1 / 7.4 (B); median 7.8 / 14.5, all 12 under.
ponytail: T is the target's median latency alone; beside a co-runner the inference is longer.
"""
import csv, glob, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.argv = ["stress.py", "-", "smooth88"]       # stress.py picks its geometry from argv at import
import stress as S
import aggregate as A

AGG = os.path.join(S.BENCH, "aggregate")
NETS = S.NETS


def windows(cfg, root=None):
    """The reporter's window latencies (ns), one array per run of one configuration, from root
    (default: the smooth88 reporter directory)."""
    out = []
    for p in sorted(glob.glob(os.path.join(root or S.REPORTER, cfg, "r*_latency.txt"))):
        v = [float(l.split()[3]) for l in open(p)
             if not l.startswith("#") and len(l.split()) == 4 and l.split()[0] == "65536"]
        out.append(np.array(v))
    return out


def ruler(top):
    """Per-window ruler: (GB/s, median window ns) for spin (0) and the smooth levels, plus full speed."""
    pts = [(0.0, S.reporter_config("spin")["median"])]
    for c in ["smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100"] + (["onoff-D100"] if top else []):
        r = S.reporter_config(c)
        pts.append((r["level"], r["median"]))
    pts.sort()
    assert all(b[1] > a[1] for a, b in zip(pts, pts[1:])), pts
    return [p[0] for p in pts], [p[1] for p in pts]


def curves(name):
    """Each target's sensitivity points (level, p99 slowdown, median slowdown) from stress.py's CSV."""
    c = {}
    for r in csv.DictReader(open(os.path.join(AGG, "stress-sensitivity-%s.csv" % name))):
        c.setdefault(r["target"], []).append((float(r["level"]), float(r["slowdown_p99"]), float(r["slowdown_median"])))
    return {t: sorted(v) for t, v in c.items()}


def felt(runs, lv, ns, T_ms):
    """Per run, the GB/s of each window averaged (time-weighted) over k consecutive windows spanning
    T; returns the medians over runs of the p99 and of the median, and k."""
    p99, med, ks = [], [], []
    for v in runs:
        b = np.interp(v, ns, lv)                       # clamps at both ends
        d = 512 * v * 1e-6                             # window length, ms
        k = max(1, int(round(T_ms / d.mean())))
        cb, cd = np.concatenate([[0], np.cumsum(b * d)]), np.concatenate([[0], np.cumsum(d)])
        avg = (cb[k:] - cb[:-k]) / (cd[k:] - cd[:-k])
        p99.append(np.percentile(avg, 99)); med.append(np.median(avg)); ks.append(k)
    return st.median(p99), st.median(med), st.median(ks)


if __name__ == "__main__":
    data = {k: v for k, v in A.load(S.BENCH).items() if k[0] == "exp23"}
    T = {t: A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "median")[0] for t in NETS}
    meas = {(r["target"], r["corunner"]): (float(r["measured_p99"]), float(r["measured_median"]),
                                           float(r["pred_p99_p99"]), float(r["pred_median_p99"]))
            for r in csv.DictReader(open(os.path.join(AGG, "stress-comparison-smooth88.csv")))}
    W = {n: windows(n + "-8sm") for n in NETS}
    print("T alone (ms):", {t: round(v, 2) for t, v in T.items()})
    for var, top, cname in (("A smooth", False, "smooth88"), ("B smooth + full speed", True, "smoothx88")):
        lv, ns = ruler(top)
        C = curves(cname)
        print("\n== variant %s   ruler GB/s %s  ns %s" % (var, [round(x, 1) for x in lv], [round(x, 1) for x in ns]))
        for mode in ("1 window", "T", "whole run"):
            e99, emed, lines = [], [], []
            for (t, c), (m99, mmed, old99, oldmed) in meas.items():
                if mode == "whole run":
                    b = [np.interp(v, ns, lv) for v in W[c]]
                    f99 = fmed = st.median(float((x * v).sum() / v.sum()) for x, v in zip(b, W[c]))
                    k = "all"
                else:
                    f99, fmed, k = felt(W[c], lv, ns, T[t] if mode == "T" else 0.0)
                L = [p[0] for p in C[t]]
                p99 = np.interp(f99, L, [p[1] for p in C[t]]); pm = np.interp(fmed, L, [p[2] for p in C[t]])
                e99.append((p99 - m99) * 100); emed.append((pm - mmed) * 100)
                lines.append("  %-9s <- %-9s k=%-4s felt p99 %6.1f med %6.1f GB/s | p99 err %+5.1f (now %+5.1f) | median err %+5.1f (now %+5.1f)"
                             % (t, c, k, f99, fmed, e99[-1], (old99 - m99) * 100, emed[-1], (oldmed - mmed) * 100))
            print(" -- average over %s: p99 |err| mean %.1f max %.1f, %d under | median |err| mean %.1f max %.1f, %d under"
                  % (mode, np.mean(np.abs(e99)), np.max(np.abs(e99)), sum(x < 0 for x in e99),
                     np.mean(np.abs(emed)), np.max(np.abs(emed)), sum(x < 0 for x in emed)))
            if mode == "T":
                print("\n".join(lines))
