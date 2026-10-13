#!/usr/bin/env python3
"""infwin-model.py - the 8 + 8 stress model read over one target inference (Daniel, 2026-10-13).

    python3 utility/analysis/infwin-model.py            predict (frozen first), then compare
    python3 utility/analysis/infwin-model.py selftest   the aggregation

smooth88 reads the reporter's p99 over its 0.33 ms windows. A target inference lasts 0.67 ms
(MobileNet) to 15.6 ms (RetinaNet), so a short peak shows in the reporter's tail whether or not it
lasts long enough to slow an inference: ResNet and RetinaNet read the same p99, both above the top
of the ruler, yet slow the targets differently. Here, per target T:

  aggregate   the reporter's windows are averaged over k consecutive windows spanning one inference
              of T alone (k = T's median latency over the run's mean window length; sliding). The
              windows hold the same number of accesses, so the plain mean is the latency a reporter
              with a window of one inference would read.
  coordinate  a statistic of the aggregated windows, per run, then the median over the five runs:
              the median (Daniel's request), or the p99 (infwin88p: the inferences that meet the
              most stress, which is what the target's p99 is made of).
  ruler       the same aggregation and statistic on the stressor's reporter runs, giving R_T(L):
              smooth (spin as level 0, D025-D100, GB/s; curves smooth88) or Steady, the plain
              bandwidth-based SAXPY (saxpy-L000-L100 read on the target's SMs, stress-reporter-target,
              level as a fraction of full speed; curves 8smt, exp13). Steady reaches 871 ns, so the
              heavy co-runners stay inside it.
  predicted   S_T(L_eq), with S_T that stressor's sensitivity curves (p99 and median slowdown).

The dispatch (presence) cost is also modelled on its own: the "separated" prediction is
1 + d_T + (S_T(L_eq) - S_T(0)), with d_T = kernels per inference x 0.87 us over the latency alone
(tab:l2-presence) in place of the zero-traffic point the curve starts from (spin; Steady's resident
L000). check_dispatch() verifies that the curves already contain the presence, which is the double
count this must avoid: their loaded points, extended to zero traffic, should land on that zero point
and not on 1.

The predictions are written before the pairs are read (stress-predictions-<ruler>{,p}.csv), as
stress.py does; the comparison goes to stress-comparison-<ruler>{,d,p,pd}.csv for models.py, with
<ruler> infwin88 (smooth) or steadywin88 (Steady).
"""
import csv
import glob
import importlib
import os
import statistics as st
import sys

import numpy as np

CMD = sys.argv[1] if len(sys.argv) > 1 else ""   # read first: avg-model rewrites sys.argv for stress.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
H = importlib.import_module("avg-model")      # windows(), curves(); imports stress as smooth88
S, A = H.S, H.A
AGG = os.path.join(S.BENCH, "aggregate")
NETS = S.NETS
KPI = {"mobilenet": 55, "resnet": 76, "inception": 101, "retinanet": 172}
PRESENCE_US = 0.87
SMOOTH = ["smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100"]


def rulers():
    """Each stressor's ruler: reporter directory, configurations and their levels, curves, output name."""
    steady = [r for r in csv.DictReader(open(os.path.join(AGG, "stress-calibration-8smt.csv")))
              if r["config"].startswith("saxpy-L")]
    return {"infwin88": dict(rep=S.REPORTER, cfgs=["spin"] + SMOOTH, curves=H.curves("smooth88"),
                             levels=[0.0] + [S.reporter_config(c)["level"] for c in SMOOTH]),
            "steadywin88": dict(rep=os.path.join(S.BENCH, "stress-reporter-target"),
                                cfgs=[r["config"] for r in steady], curves=H.curves("8smt"),
                                levels=[float(r["level"]) for r in steady])}


STATS = {"": np.median, "p": lambda a: np.percentile(a, 99)}    # file suffix -> statistic


def aggregated(runs, T_ms, stat=np.median):
    """Median over runs of stat() of the k-window means, k windows spanning T_ms; and k."""
    meds, ks = [], []
    for v in runs:
        k = max(1, int(round(T_ms / (512 * v.mean() * 1e-6))))
        c = np.concatenate([[0], np.cumsum(v)])
        meds.append(float(stat((c[k:] - c[:-k]) / k)))
        ks.append(k)
    return st.median(meds), st.median(ks)


def check_dispatch(name, data, alone, curves):
    """Is the presence already in the curves? A line through the loaded points (for smooth with
    peak-T0010, the same kernel at duty 10 %, as the lowest) extended to zero traffic, against the
    zero point and the dispatch model. Prints only: it decides whether d_T may be added."""
    print("\ndispatch, %s: curves extended to zero traffic (p99 slowdown - 1, %%)" % name)
    print("  %-10s %10s %8s %10s %8s" % ("target", "intercept", "zero pt", "d_T model", "slope"))
    for t in NETS:
        pts = [(l, a) for l, a, _ in curves[t] if l > 0]
        if name == "infwin88":
            cfg = t + "-peak-T0010"
            p99 = A.aggr(data, "exp23", cfg, t, "gpu_ms", "p99")[0] / alone[t][1]
            lv = [S.achieved_level(f)[0] for f in sorted(glob.glob(os.path.join(S.BENCH, "stress-onoff", cfg, "r*_itf.log")))]
            pts.append((st.median(lv), p99))
        slope, icpt = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
        spin = [a for l, a, _ in curves[t] if l == 0][0]
        print("  %-10s %10.1f %8.1f %10.1f %8.3f" % (t, 100 * (icpt - 1), 100 * (spin - 1),
                                                    100 * dispatch(t, alone), 100 * slope))


def dispatch(t, alone):
    return KPI[t] * PRESENCE_US * 1e-3 / alone[t][0]


def predict(alone, ruler, stat):
    """Per target: the ruler, each co-runner's coordinate and L_eq, both predictions. Reads no pair.
    The networks' reporter runs are networks-8sm (stress-reporter-onoff) for both rulers."""
    W = {n: H.windows(n + "-8sm") for n in NETS}
    R = {c: H.windows(c, ruler["rep"]) for c in ruler["cfgs"]}
    lv, curves = ruler["levels"], ruler["curves"]
    rows = []
    print("\nruler and co-runners, %s of windows spanning one target inference (ns)"
          % ("median" if stat is np.median else "p99"))
    for t in NETS:
        T = alone[t][0]
        rv = [aggregated(R[c], T, stat)[0] for c in ruler["cfgs"]]
        print("  %-10s T %.2f ms, ruler %s" % (t, T, " ".join("%.1f" % y for y in rv)))
        L, s99, s50 = [p[0] for p in curves[t]], [p[1] for p in curves[t]], [p[2] for p in curves[t]]
        for c in NETS:
            if c == t:
                continue
            y, k = aggregated(W[c], T, stat)
            leq, inside = S.invert(y, lv, rv)
            b99, b50 = float(np.interp(leq, L, s99)), float(np.interp(leq, L, s50))
            d = dispatch(t, alone)
            rows.append(dict(target=t, corunner=c, k=k, coord_ns=round(y, 2), L_eq=round(leq, 3), inside=inside,
                             pred_p99=round(b99, 4), pred_median=round(b50, 4),
                             pred_p99_sep=round(1 + d + b99 - s99[0], 4), pred_median_sep=round(1 + d + b50 - s50[0], 4)))
            print("    beside %-10s k %3d  %.1f ns  L_eq %7.3f%s" % (c, k, y, leq, "" if inside else "  OUTSIDE"))
    return rows


def write(name, rows):
    path = os.path.join(AGG, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path)


def compare(preds, alone, sfx):
    data = A.load(S.BENCH)
    for suffix, key, label in (("", "", "dispatch inside the curve"), ("d", "_sep", "dispatch separated (d_T model)")):
        rows = []
        for p in preds:
            t, c = p["target"], p["corunner"]
            m99 = A.aggr(data, "exp24", t + "+" + c, t, "gpu_ms", "p99")[0] / alone[t][1]
            m50 = A.aggr(data, "exp24", t + "+" + c, t, "gpu_ms", "median")[0] / alone[t][0]
            # the column names of stress.py compare, so models.py reads every model alike
            rows.append(dict(target=t, corunner=c, measured_p99=round(m99, 4), measured_median=round(m50, 4),
                             pred_p99_p99=p["pred_p99" + key], err_p99_p99_pts=round((p["pred_p99" + key] - m99) * 100, 2),
                             pred_median_p99=p["pred_median" + key],
                             err_median_p99_pts=round((p["pred_median" + key] - m50) * 100, 2), L_eq=p["L_eq"]))
        write("stress-comparison-%s%s.csv" % (sfx, suffix), rows)
        print("  %s" % label)
        for q in ("p99", "median"):
            e = np.array([r["err_%s_p99_pts" % q] for r in rows])
            print("    %-6s mean |error| %.1f, max %.1f, %d under, largest under %.1f"
                  % (q, np.mean(np.abs(e)), np.max(np.abs(e)), np.sum(e < 0), min(e.min(), 0)))


def main():
    data = {k: v for k, v in A.load(S.BENCH).items() if k[0] == "exp23"}
    alone = {t: (A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "median")[0],
                 A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "p99")[0]) for t in NETS}
    for name, ruler in rulers().items():
        print("\n===== %s" % name)
        check_dispatch(name, data, alone, ruler["curves"])
        for sfx, stat in STATS.items():
            preds = predict(alone, ruler, stat)
            write("stress-predictions-%s%s.csv" % (name, sfx), preds)
            compare(preds, alone, name + sfx)


def selftest():
    v = np.array([600.0] * 6 + [900.0] * 6)          # windows of 0.3-0.46 ms
    y, k = aggregated([v], 512 * 750e-6 * 3)   # three windows of the mean length
    assert k == 3 and y == 750.0                      # 10 sliding means: 600 700 800 900 ...
    assert aggregated([v], 0.0) == (750.0, 1)  # k = 1 is the plain median of the windows
    assert aggregated([v], 0.0, STATS["p"])[0] == 900.0
    print("  selftest passed")


if __name__ == "__main__":
    {"selftest": selftest}.get(CMD, main)()
