#!/usr/bin/env python3
"""stress.py - the stress model: predict a target's slowdown from a co-runner's stress.

    python3 utility/stress.py predict     build the model, write the predictions (frozen)
    python3 utility/stress.py compare     the predictions against the measured pairs
    python3 utility/stress.py selftest    the interpolation and its inverse

predict reads only the reporter runs ($DATA/bench/stress-reporter) and the stress-sensitivity
family. It never reads stress-validation: those pairs are the ground truth and must not enter the
model. compare is the one place that reads them, and it refuses to run before predict has written
its file. Every quantity is defined in MEASUREMENTS.md, "The stress model".

  R(L)       the reporter's latency (L2 and DRAM buffer) beside SAXPY at achieved level L
  L_eq(net)  the level whose R equals the reporter's latency beside that network, per coordinate
  S(L)       a target's slowdown beside SAXPY at level L, against the target alone (p99 and median)
  predicted  S_target(L_eq(co-runner))
"""
import csv
import glob
import os
import re
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as A  # noqa: E402

BENCH = os.path.join(A.DATA_ROOT, "bench")
REPORTER = os.path.join(BENCH, "stress-reporter")
OUT = os.path.join(BENCH, "aggregate")
SIZES = {"2048": "l2", "65536": "dram"}
NETS = ("mobilenet", "resnet", "inception", "retinanet")


def achieved_level(log):
    """Median of the achieved levels itfproc printed once a second, or None."""
    vals = [float(m) for m in re.findall(r"achieved ([0-9.]+)", open(log, errors="replace").read())]
    return (st.median(vals), min(vals), max(vals)) if vals else None


def reporter_latency(path):
    """Per buffer coordinate, the median ns per access of one reporter run."""
    per = {}
    for line in open(path):
        f = line.split()
        if f and not line.startswith("#") and f[0] in SIZES:
            per.setdefault(SIZES[f[0]], []).append(float(f[3]))
    return {k: st.median(v) for k, v in per.items()}


def reporter_config(name):
    """The reporter beside one co-runner configuration: per coordinate, the median over runs of the
    per-run medians, plus the co-runner's achieved level or occupancy, also a median over runs."""
    d = os.path.join(REPORTER, name)
    runs = sorted(glob.glob(os.path.join(d, "r*_latency.txt")))
    if not runs:
        return None
    lat = [reporter_latency(r) for r in runs]
    row = {c: st.median(x[c] for x in lat) for c in SIZES.values()}
    row["runs"] = len(runs)
    levels = [achieved_level(f) for f in sorted(glob.glob(os.path.join(d, "r*_c0.log")))]
    levels = [lv for lv in levels if lv]
    if levels:
        row["level"] = st.median(lv[0] for lv in levels)
        row["level_spread"] = max(lv[2] for lv in levels) - min(lv[1] for lv in levels)
    occ, rate = [], []
    for f in sorted(glob.glob(os.path.join(d, "r*_c0_stats.csv"))):
        g = A.read_stats(f)["gpu_ms"]
        occ.append(g["mean"] * g["count"] / (g["duration_s"] * 1000.0))   # busy fraction
        rate.append(g["count"] / g["duration_s"])
    if occ:
        row["occupancy"], row["inf_per_s"] = st.median(occ), st.median(rate)
    return row


def invert(y, xs, ys):
    """x with R(x) = y on a monotonically rising curve, by linear interpolation; flags a y outside
    the calibrated range instead of extrapolating."""
    if not all(b > a for a, b in zip(ys, ys[1:])):
        raise SystemExit("stress.py: the calibration curve is not monotonic: %s" % ys)
    inside = ys[0] <= y <= ys[-1]
    return float(np.interp(y, ys, xs)), inside


def sensitivity(data, target):
    """S(L) for one target from stress-sensitivity only: x = achieved level per configuration, y =
    the target's p99 and median against its alone row, all medians over the five runs."""
    alone = A.aggr(data, "exp13", target + "-alone", target, "gpu_ms", "median")
    alone99 = A.aggr(data, "exp13", target + "-alone", target, "gpu_ms", "p99")
    if not alone or not alone99:
        return None
    pts = []
    for cfg in sorted({c for (e, c, _) in data if e == "exp13" and c.startswith(target + "-saxpy-")}):
        med = A.aggr(data, "exp13", cfg, target, "gpu_ms", "median")
        p99 = A.aggr(data, "exp13", cfg, target, "gpu_ms", "p99")
        logs = sorted(glob.glob(os.path.join(BENCH, "stress-sensitivity", cfg, "r*_itf.log")))
        lv = [achieved_level(f) for f in logs]
        lv = [x[0] for x in lv if x]
        if med and p99 and lv:
            pts.append((st.median(lv), p99[0] / alone99[0], med[0] / alone[0]))
    pts.sort()
    return pts


def write(name, rows):
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path)


def fmt(v):
    return round(v, 4) if isinstance(v, float) else v


def predict():
    # 1. calibration: R(L) per coordinate, at the achieved levels
    calib = []
    for name in sorted(os.listdir(REPORTER)) if os.path.isdir(REPORTER) else []:
        if name.startswith("saxpy-L"):
            r = reporter_config(name)
            if r and "level" in r:
                calib.append(dict(config=name, **{k: fmt(v) for k, v in r.items()}))
    calib.sort(key=lambda r: r["level"])
    if len(calib) < 3:
        raise SystemExit("stress.py: too few calibration levels measured")
    idle = reporter_config("idle")
    write("stress-calibration.csv", calib + ([dict(config="idle", **{k: fmt(v) for k, v in idle.items()})]
                                             if idle else []))
    xs = [r["level"] for r in calib]

    # 2. equivalent stress of each network as a co-runner, per coordinate
    eq = {}
    corunners = []
    for net in NETS:
        r = reporter_config(net)
        if not r:
            continue
        row = dict(corunner=net, runs=r["runs"])
        for c in SIZES.values():
            leq, inside = invert(r[c], xs, [k[c] for k in calib])
            row[c + "_ns"], row["L_eq_" + c], row[c + "_inside"] = fmt(r[c]), fmt(leq), inside
            eq[(net, c)] = leq
        row["occupancy"], row["inf_per_s"] = fmt(r.get("occupancy", "")), fmt(r.get("inf_per_s", ""))
        corunners.append(row)
    write("stress-corunners.csv", corunners)

    # 3. sensitivity curves, from stress-sensitivity ONLY
    data = {k: v for k, v in A.load(BENCH).items() if k[0] == "exp13"}
    curves, sens = {}, []
    for t in NETS:
        pts = sensitivity(data, t)
        if pts:
            curves[t] = pts
            sens += [dict(target=t, level=fmt(l), slowdown_p99=fmt(a), slowdown_median=fmt(b))
                     for l, a, b in pts]
    write("stress-sensitivity.csv", sens)

    # 4. the predictions, frozen before any pair is read
    preds = []
    for t, pts in curves.items():
        L = [p[0] for p in pts]
        for net in NETS:
            if net == t or (net, "dram") not in eq:
                continue
            row = dict(target=t, corunner=net)
            for c in SIZES.values():
                row["L_eq_" + c] = fmt(eq[(net, c)])
                row["pred_p99_" + c] = fmt(float(np.interp(eq[(net, c)], L, [p[1] for p in pts])))
                row["pred_median_" + c] = fmt(float(np.interp(eq[(net, c)], L, [p[2] for p in pts])))
            preds.append(row)
    write("stress-predictions.csv", preds)


def compare():
    path = os.path.join(OUT, "stress-predictions.csv")
    if not os.path.exists(path):
        raise SystemExit("stress.py: run `predict` first; the predictions must exist before the pairs are read")
    preds = list(csv.DictReader(open(path)))
    occ = {r["corunner"]: r for r in csv.DictReader(open(os.path.join(OUT, "stress-corunners.csv")))}
    data = A.load(BENCH)
    rows = []
    for p in preds:
        t, net = p["target"], p["corunner"]
        alone = A.aggr(data, "exp13", t + "-alone", t, "gpu_ms", "median")
        alone99 = A.aggr(data, "exp13", t + "-alone", t, "gpu_ms", "p99")
        med = A.aggr(data, "exp14", t + "+" + net, t, "gpu_ms", "median")
        p99 = A.aggr(data, "exp14", t + "+" + net, t, "gpu_ms", "p99")
        if not (alone and alone99 and med and p99):
            continue
        meas = {"p99": p99[0] / alone99[0], "median": med[0] / alone[0]}
        row = dict(target=t, corunner=net, measured_p99=fmt(meas["p99"]), measured_median=fmt(meas["median"]))
        for q in ("p99", "median"):
            for c in SIZES.values():
                pred = float(p["pred_%s_%s" % (q, c)])
                row["pred_%s_%s" % (q, c)] = fmt(pred)
                row["err_%s_%s_pts" % (q, c)] = fmt((pred - meas[q]) * 100)       # slowdown points
        row.update(L_eq_l2=p["L_eq_l2"], L_eq_dram=p["L_eq_dram"],
                   occupancy=occ.get(net, {}).get("occupancy", ""),
                   inf_per_s=occ.get(net, {}).get("inf_per_s", ""))
        rows.append(row)
    if rows:
        write("stress-comparison.csv", rows)
        for q in ("p99", "median"):
            for c in SIZES.values():
                e = [abs(r["err_%s_%s_pts" % (q, c)]) for r in rows]
                print("  %-6s from %-4s: mean |error| %.1f points, max %.1f, over %d pairs"
                      % (q, c, st.mean(e), max(e), len(e)))


def selftest():
    xs, ys = [0.0, 0.5, 1.0], [300.0, 400.0, 600.0]
    assert invert(350.0, xs, ys) == (0.25, True)
    assert invert(500.0, xs, ys) == (0.75, True)
    assert invert(700.0, xs, ys) == (1.0, False)          # outside: clipped and flagged
    try:
        invert(350.0, xs, [300.0, 300.0, 600.0])
        raise AssertionError("a flat curve must be refused")
    except SystemExit:
        pass
    assert float(np.interp(0.25, xs, [1.0, 1.2, 1.6])) == 1.1
    print("  selftest passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"predict": predict, "compare": compare, "selftest": selftest}.get(
        cmd, lambda: sys.exit(__doc__))()
