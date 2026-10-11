#!/usr/bin/env python3
"""stress.py - the stress model: predict a target's slowdown from a co-runner's stress.

    python3 utility/stress.py predict [geom]  build the model, write the predictions (frozen)
    python3 utility/stress.py compare [geom]  the predictions against the measured pairs
    python3 utility/stress.py selftest        the interpolation and its inverse

The stressor's geometry: 8sm (the default, SAXPY on 8 SMs, exp13) or 4sm (SAXPY on the 4 SMs the real
co-runners use, exp15). Each writes its own files, so the frozen 8 SM predictions are never touched.

predict reads only the reporter runs ($DATA/bench/stress-reporter) and the stress-sensitivity
family. It never reads stress-validation: those pairs are the ground truth and must not enter the
model. compare is the one place that reads them, and it refuses to run before predict has written
its file. Every quantity is defined in MEASUREMENTS.md, "The stress model".

  R(L)       the reporter's DRAM latency beside SAXPY at achieved level L: per run, the p99 (the
             model's coordinate) and the median of its 0.3 ms windows
  L_eq(net)  the level whose R equals the reporter's reading beside that network, per statistic
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
# The reporter's coordinates, all on the 64 MB (DRAM) chain: a statistic of the run's windows.
STATS = {"p99": lambda v: float(np.percentile(v, 99)), "median": lambda v: float(np.median(v)),
         "mean": lambda v: float(np.mean(v))}     # mean: the unpatched reporter's one number per run
NETS = ("mobilenet", "resnet", "inception", "retinanet")
# The stressor's geometry: calibration configs, sensitivity family and its directory, file suffix.
GEOMS = {"8sm": dict(cal="saxpy-L", exp="exp13", dir="stress-sensitivity", suffix="", rep="stress-reporter"),
         "4sm": dict(cal="saxpy4-L", exp="exp15", dir="stress-sensitivity-4sm", suffix="-4sm",
                     rep="stress-reporter"),
         # the baseline: time-based SAXPY, the unpatched reporter (WINDOW=0) on the target's SMs
         "tb": dict(cal="tb-D", exp="exp17", dir="stress-sensitivity-tb", suffix="-tb",
                    rep="stress-reporter-baseline"),
         # the same time-based SAXPY and sensitivity, read by the windowed reporter on the target's SMs
         "tbw": dict(cal="tb-D", exp="exp17", dir="stress-sensitivity-tb", suffix="-tbw",
                     rep="stress-reporter-tbw"),
         # time-based SAXPY scrambled within 2 MB windows, read by the windowed reporter; the
         # networks' reporter runs are tbw's (symlinked), so only the stressor differs from tbw
         "tbsw": dict(cal="tbs-D", exp="exp18", dir="stress-sensitivity-tbs", suffix="-tbsw",
                      rep="stress-reporter-tbsw"),
         # the bandwidth-based SAXPY (exp13) read by the windowed reporter on the target's SMs
         # (LAYOUT=target): the symmetric bandwidth-based model
         "8smt": dict(cal="saxpy-L", exp="exp13", dir="stress-sensitivity", suffix="-8smt",
                      rep="stress-reporter-target"),
         # Block 2 (2026-10-11): ON/OFF SAXPY over 400 ms periods, and its uniform counterpart (32 MB
         # units over ~3 ms periods), at matched mean bandwidth (exp23), read by the windowed reporter
         # on the target's SMs. The level is the stressor's mean GB/s, one axis for both, and level 0
         # is the target alone: with nothing resident in OFF, a vanishing duty is no co-runner at all.
         "onoff": dict(cal="onoff-D", exp="exp23", dir="stress-onoff", suffix="-onoff",
                       rep="stress-reporter-onoff", gbs=True),
         "uniform": dict(cal="uniform-D", exp="exp23", dir="stress-onoff", suffix="-uniform",
                         rep="stress-reporter-onoff", gbs=True)}
G = GEOMS[sys.argv[2] if len(sys.argv) > 2 else "8sm"]
REPORTER = os.path.join(BENCH, G["rep"])


def achieved_level(log):
    """The co-runner's achieved level, or None: the median of the levels itfproc printed once a
    second; for the gbs geometries, its mean GB/s over the run (the last of the cumulative figures
    it prints once a second, which span the start-up)."""
    txt = open(log, errors="replace").read()
    if G.get("gbs"):
        vals = [float(m) for m in re.findall(r"bytes moved:.*\(([0-9.]+) GB/s\)", txt)]
        return (vals[-1], min(vals), max(vals)) if vals else None
    vals = [float(m) for m in re.findall(r"achieved ([0-9.]+)", txt)]
    return (st.median(vals), min(vals), max(vals)) if vals else None


def reporter_latency(path):
    """Per coordinate, one statistic of the DRAM-chain window latencies (ns) of one reporter run."""
    v = []
    for line in open(path):
        f = line.split()
        # four fields: a reporter stopped mid-line (beside networks) leaves a partial last line
        if len(f) == 4 and not line.startswith("#") and f[0] == "65536":
            v.append(float(f[3]))
    return {k: fn(v) for k, fn in STATS.items()}


def reporter_config(name):
    """The reporter beside one co-runner configuration: per coordinate, the median over runs of the
    per-run medians, plus the co-runner's achieved level or occupancy, also a median over runs."""
    d = os.path.join(REPORTER, name)
    runs = sorted(glob.glob(os.path.join(d, "r*_latency.txt")))
    if not runs:
        return None
    lat = [reporter_latency(r) for r in runs]
    row = {c: st.median(x[c] for x in lat) for c in STATS}
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
    the calibrated range instead of extrapolating.

    The curve is used from its lowest point upward. At level 0 the stressor moves no data, the DRAM
    drifts toward its low-power state, and the reporter's p99 there is power-state noise (621-635 ns
    across five runs, against +-0.6 ns at 2 %), lying above the 2 % point. Points before the minimum
    are therefore not part of the scale; a reading below the minimum maps to its level, flagged. Any
    other non-monotonic point still refuses the curve."""
    lo = int(np.argmin(ys))
    xs, ys = list(xs)[lo:], list(ys)[lo:]
    if not all(b > a for a, b in zip(ys, ys[1:])):
        raise SystemExit("stress.py: the calibration curve is not monotonic: %s" % ys)
    inside = ys[0] <= y <= ys[-1]
    return float(np.interp(y, ys, xs)), inside


def sensitivity(data, target):
    """S(L) for one target from stress-sensitivity only: x = achieved level per configuration, y =
    the target's p99 and median against its alone row, all medians over the five runs."""
    alone = A.aggr(data, G["exp"], target + "-alone", target, "gpu_ms", "median")
    alone99 = A.aggr(data, G["exp"], target + "-alone", target, "gpu_ms", "p99")
    if not alone or not alone99:
        return None
    pts = []
    # exp23 holds both stressors of block 2, so a gbs geometry takes only its own configurations
    prefix = target + "-" + (G["cal"] if G.get("gbs") else "")
    for cfg in sorted({c for (e, c, _) in data if e == G["exp"] and c.startswith(prefix) and c != target + "-alone"}):
        med = A.aggr(data, G["exp"], cfg, target, "gpu_ms", "median")
        p99 = A.aggr(data, G["exp"], cfg, target, "gpu_ms", "p99")
        logs = sorted(glob.glob(os.path.join(BENCH, G["dir"], cfg, "r*_itf.log")))
        lv = [achieved_level(f) for f in logs]
        lv = [x[0] for x in lv if x]
        if med and p99 and lv:
            pts.append((st.median(lv), p99[0] / alone99[0], med[0] / alone[0]))
    if G.get("gbs"):
        pts.append((0.0, 1.0, 1.0))           # level 0 is the target alone
    pts.sort()
    return pts


def write(name, rows):
    if not rows:
        print("  %s: nothing to write yet" % name)
        return
    path = os.path.join(OUT, name.replace(".csv", G["suffix"] + ".csv"))
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
        if name.startswith(G["cal"]):
            r = reporter_config(name)
            if r and "level" in r:
                calib.append(dict(config=name, **{k: fmt(v) for k, v in r.items()}))
    idle = reporter_config("idle")
    if G.get("gbs") and idle:                 # level 0 of a gbs geometry: nothing resident
        calib.append(dict(config="idle", level=0.0, level_spread=0.0, **{k: fmt(v) for k, v in idle.items()}))
    calib.sort(key=lambda r: r["level"])
    if len(calib) < 3:
        raise SystemExit("stress.py: too few calibration levels measured")
    write("stress-calibration.csv", calib + ([dict(config="idle", **{k: fmt(v) for k, v in idle.items()})]
                                             if idle and not G.get("gbs") else []))
    xs = [r["level"] for r in calib]

    # 2. equivalent stress of each network as a co-runner, per coordinate
    eq = {}
    corunners = []
    for net in NETS:
        r = reporter_config(net)
        if not r:
            continue
        row = dict(corunner=net, runs=r["runs"])
        for c in STATS:
            leq, inside = invert(r[c], xs, [k[c] for k in calib])
            row[c + "_ns"], row["L_eq_" + c], row[c + "_inside"] = fmt(r[c]), fmt(leq), inside
            eq[(net, c)] = leq
        row["occupancy"], row["inf_per_s"] = fmt(r.get("occupancy", "")), fmt(r.get("inf_per_s", ""))
        corunners.append(row)
    write("stress-corunners.csv", corunners)

    # 3. sensitivity curves, from stress-sensitivity ONLY
    data = {k: v for k, v in A.load(BENCH).items() if k[0] == G["exp"]}
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
            if net == t or (net, "p99") not in eq:
                continue
            row = dict(target=t, corunner=net)
            for c in STATS:
                row["L_eq_" + c] = fmt(eq[(net, c)])
                row["pred_p99_" + c] = fmt(float(np.interp(eq[(net, c)], L, [p[1] for p in pts])))
                row["pred_median_" + c] = fmt(float(np.interp(eq[(net, c)], L, [p[2] for p in pts])))
            preds.append(row)
    write("stress-predictions.csv", preds)


def compare():
    path = os.path.join(OUT, "stress-predictions%s.csv" % G["suffix"])
    if not os.path.exists(path):
        raise SystemExit("stress.py: run `predict` first; the predictions must exist before the pairs are read")
    preds = list(csv.DictReader(open(path)))
    occ = {r["corunner"]: r for r in csv.DictReader(open(os.path.join(OUT, "stress-corunners%s.csv" % G["suffix"])))}
    data = A.load(BENCH)
    rows = []
    for p in preds:
        t, net = p["target"], p["corunner"]
        alone = A.aggr(data, G["exp"], t + "-alone", t, "gpu_ms", "median")
        alone99 = A.aggr(data, G["exp"], t + "-alone", t, "gpu_ms", "p99")
        med = A.aggr(data, "exp14", t + "+" + net, t, "gpu_ms", "median")
        p99 = A.aggr(data, "exp14", t + "+" + net, t, "gpu_ms", "p99")
        if not (alone and alone99 and med and p99):
            continue
        meas = {"p99": p99[0] / alone99[0], "median": med[0] / alone[0]}
        row = dict(target=t, corunner=net, measured_p99=fmt(meas["p99"]), measured_median=fmt(meas["median"]))
        for q in ("p99", "median"):
            for c in STATS:
                pred = float(p["pred_%s_%s" % (q, c)])
                row["pred_%s_%s" % (q, c)] = fmt(pred)
                row["err_%s_%s_pts" % (q, c)] = fmt((pred - meas[q]) * 100)       # slowdown points
        row.update(L_eq_p99=p["L_eq_p99"], L_eq_median=p["L_eq_median"],
                   occupancy=occ.get(net, {}).get("occupancy", ""),
                   inf_per_s=occ.get(net, {}).get("inf_per_s", ""))
        rows.append(row)
    if rows:
        write("stress-comparison.csv", rows)
        for q in ("p99", "median"):
            for c in STATS:
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
    # a level-0 point above the curve's minimum is dropped; a reading below the minimum maps to it
    xs2, ys2 = [0.0, 0.02, 0.1, 0.2], [629.0, 623.0, 638.0, 657.0]
    assert invert(612.0, xs2, ys2) == (0.02, False)
    assert abs(invert(638.0, xs2, ys2)[0] - 0.1) < 1e-12
    try:
        invert(640.0, xs2, [623.0, 640.0, 630.0, 657.0])
        raise AssertionError("a dip after the minimum must still be refused")
    except SystemExit:
        pass
    assert float(np.interp(0.25, xs, [1.0, 1.2, 1.6])) == 1.1
    print("  selftest passed")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"predict": predict, "compare": compare, "selftest": selftest}.get(
        cmd, lambda: sys.exit(__doc__))()
