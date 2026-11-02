#!/usr/bin/env python3
"""stress-tables.py - the thesis tables of the stress models: stressors, probe scales, the smooth
model, the peak test, the temporal model, the counter variants and the two-SAXPY composition.

    python3 analysis/tables/stress-tables.py     every table as LaTeX rows, then the prose numbers

A regenerator, not a record: it reads the CSVs that stress.py (onoff88, uniform88, smooth88,
smoothx88), infwin-model.py, counter-model.py and compose.py write in $DATA/bench/aggregate, and,
for the peak test only, the probe's raw windows and the exp23 runs. It writes nothing. Each block is
headed by its label in stress-model.tex:

  tab:stress-scale          the probe next to each stressor: mean bandwidth and p99 (ns)
  tab:stress-sens           each target's p99 and median slowdown next to each stressor
  tab:stress-corunners      the probe next to each network, and its equivalent stress on each scale
  tab:stress-accuracy       the ON/OFF and uniform models' errors over the twelve pairs
  tab:stress-pairs          measured and predicted slowdowns of the twelve pairs, both models
  tab:stress-smooth         the smooth stressor's scale and the targets' slowdown next to it
  tab:stress-smooth-pairs   the smooth model on the twelve pairs
  tab:stress-peak           duty 10 %, peaks of 0.01-3 ms: the probe and the targets
  tab:stress-temporal-stat  the temporal model with the median and the p99 of the averaged windows
  tab:counter-variants      the counter model's error with each input, twelve and thirty pairs
  tab:stress-compose        two smooth stressors on 4 SMs each: equivalent stress alone and together

Every prediction uses the probe's p99 (the CSV column pred_<target statistic>_p99). Errors are the
CSVs' own, predicted minus measured slowdown in points, so a rounded measured value never flips a
rounded error.
"""
import csv
import importlib
import os
import statistics as st
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # analysis/
M = importlib.import_module("avg-model")     # first: it sets the geometry stress.py reads at import
S, A = M.S, M.A
AGG = M.AGG
NETS = S.NETS
DUTY = {"D025": "25\\,\\%", "D050": "50\\,\\%", "D075": "75\\,\\%", "D100": "100\\,\\%"}


def read(name):
    return list(csv.DictReader(open(os.path.join(AGG, name))))


def pts(x):
    """Signed points for LaTeX: $+$1.6, $-$0.3, and 0.0 when it rounds to zero."""
    s = "%.1f" % abs(x)
    return "0.0" if s == "0.0" else ("$+$" if x > 0 else "$-$") + s


def pct(s):
    """A slowdown ratio as a percentage, signed only when negative ($-$0.1)."""
    return pts(100 * (float(s) - 1)).replace("$+$", "")


def f1(x):
    """One decimal, a tie rounded up on the shortest decimal form: 790.55 -> 790.6 ("%.1f" gives
    790.5, the binary float being 790.5499...)."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def row(*cells):
    print(" & ".join(str(c) for c in cells) + " \\\\")


def head(label):
    print("\n%% ---- %s" % label)


DATA = A.load(S.BENCH)


def scale_rows(geom):
    """The scale of a stressor, recomputed by stress.py from the raw runs, in level order (spin first):
    [(config, level GB/s, probe p99 ns, [(p99, median slowdown) per target])]. Not read from the
    calibration and sensitivity CSVs, which keep 2-4 decimals: too few to round 790.553 ns or a
    slowdown of 1.25449 as the exact values do."""
    S.G, S.REPORTER = S.GEOMS[geom], os.path.join(S.BENCH, S.GEOMS[geom]["rep"])
    cfgs = [r["config"] for r in sorted(read("stress-calibration-%s.csv" % geom), key=lambda r: float(r["level"]))]
    sens = {t: S.sensitivity(DATA, t) for t in NETS}
    assert all(len(sens[t]) == len(cfgs) for t in NETS), geom
    out = []
    for i, cfg in enumerate(cfgs):
        r = S.reporter_config(cfg)
        level = 0.0 if cfg == S.G.get("zero") else r["level"]
        out.append((cfg, level, r["p99"], [sens[t][i][1:] for t in NETS]))
    return out


def comparison(name):
    return read("stress-comparison-%s.csv" % name)


def summary4(e):
    """mean |e|, max |e|, mean e, under: the four summary rows of a per-pair table."""
    e = np.array(e)
    return ["%.1f" % np.abs(e).mean(), "%.1f" % np.abs(e).max(), pts(e.mean()), "%d" % (e < 0).sum()]


# ------------------------------------------------------------------ the two gapped stressors

STRESSORS = (("ON/OFF", "onoff88"), ("uniform", "uniform88"))
rows = {g: scale_rows(g) for _, g in STRESSORS}

head("tab:stress-scale")
row("\\texttt{spin}", "---", f1(rows["onoff88"][0][1]), f1(rows["onoff88"][0][2]))
for name, g in STRESSORS:
    print("\\midrule")
    for i, (cfg, level, p99, _) in enumerate(rows[g][1:]):
        row(name if i == 0 else "", DUTY[cfg.split("-")[1]], f1(level), f1(p99))
for name, g in STRESSORS:
    ns = [x[2] for x in rows[g]]
    print("%% %s: spin to the lowest level %+.0f ns, lowest to highest %+.0f ns" % (name, ns[1] - ns[0], ns[-1] - ns[1]))

head("tab:stress-sens (p99 slowdown of the four targets, then the median)")
spin = rows["onoff88"][0][3]
row("\\texttt{spin}", "---", *[pct(x[0]) for x in spin], *[pct(x[1]) for x in spin])
for name, g in STRESSORS:
    print("\\midrule")
    for i, (cfg, _, _, s) in enumerate(rows[g][1:]):
        row(name if i == 0 else "", DUTY[cfg.split("-")[1]], *[pct(x[0]) for x in s], *[pct(x[1]) for x in s])

head("tab:stress-corunners (probe p99, then L_eq on the ON/OFF and uniform scales)")
co = {g: {r["corunner"]: r for r in read("stress-corunners-%s.csv" % g)} for _, g in STRESSORS}
for c in sorted(co["onoff88"], key=lambda c: float(co["onoff88"][c]["p99_ns"])):
    row(c, f1(co["onoff88"][c]["p99_ns"]), *[f1(co[g][c]["L_eq_p99"]) for _, g in STRESSORS])

head("tab:stress-accuracy (p99 error, then median error, both from the probe's p99)")
cmp = {g: comparison(g) for _, g in STRESSORS}
for name, g in STRESSORS:
    e99 = np.array([float(r["err_p99_p99_pts"]) for r in cmp[g]])
    emd = np.array([float(r["err_median_p99_pts"]) for r in cmp[g]])
    row(name, *["%.1f & %.1f & %s" % (np.abs(e).mean(), np.abs(e).max(), pts(e.mean())) for e in (e99, emd)])
    print("%%   %s: under p99 %d, under median %d" % (name, (e99 < 0).sum(), (emd < 0).sum()))
pm = [100 * (float(r["pred_median_p99"]) - 1) for r in cmp["onoff88"]]
mm = [100 * (float(r["measured_median"]) - 1) for r in cmp["onoff88"]]
print("%% ON/OFF median predicted at most %.1f %%, measured %.1f-%.1f %%" % (max(pm), min(mm), max(mm)))

head("tab:stress-pairs (p99: measured, ON/OFF, uniform; median: the same)")
last = None
for a, b in zip(*[cmp[g] for _, g in STRESSORS]):
    t, c = a["target"], a["corunner"]
    if last is not None and t != last:
        print("\\midrule")
    row(t if t != last else "", c, pct(a["measured_p99"]), pct(a["pred_p99_p99"]), pct(b["pred_p99_p99"]),
        pct(a["measured_median"]), pct(a["pred_median_p99"]), pct(b["pred_median_p99"]))
    last = t

# ------------------------------------------------------------------ the smooth stressor

head("tab:stress-smooth (duty, GB/s, probe p99; p99 slowdowns; median slowdowns)")
for cfg, level, p99, s in scale_rows("smooth88"):
    duty = "\\texttt{spin}" if cfg == "spin" else DUTY[cfg.split("-")[1]]
    row(duty, f1(level), f1(p99), *[pct(x[0]) for x in s], *[pct(x[1]) for x in s])
sm = read("stress-corunners-smooth88.csv")
print("% on the smooth scale: " + ", ".join("%s %.0f GB/s (%s)" % (r["corunner"], float(r["L_eq_p99"]),
                                                                 "inside" if r["p99_inside"] == "True" else "clamped")
                                         for r in sm))

head("tab:stress-smooth-pairs (p99: measured, predicted, error; median: the same)")
last = None
for r in comparison("smooth88"):
    t = r["target"]
    if last is not None and t != last:
        print("\\midrule")
    row(t if t != last else "", r["corunner"], pct(r["measured_p99"]), pct(r["pred_p99_p99"]), pts(float(r["err_p99_p99_pts"])),
        pct(r["measured_median"]), pct(r["pred_median_p99"]), pts(float(r["err_median_p99_pts"])))
    last = t
for g in ("smooth88", "smoothx88"):
    e = np.array([float(r["err_p99_p99_pts"]) for r in comparison(g)])
    em = np.array([float(r["err_median_p99_pts"]) for r in comparison(g)])
    print("%% %s: p99 mean |e| %.1f, max %.1f, under %d; median mean |e| %.1f, over %d, max %+.1f" % (
        g, np.abs(e).mean(), np.abs(e).max(), (e < 0).sum(), np.abs(em).mean(), (em > 0).sum(), em.max()))

# ------------------------------------------------------------------ the peak test

head("tab:stress-peak (peak and period in ms, probe p99; p99 slowdowns; median slowdowns)")
data = {k: v for k, v in DATA.items() if k[0] == "exp23"}


def slowdown(t, cfg, q):
    return A.aggr(data, "exp23", t + "-" + cfg, t, "gpu_ms", q)[0] / A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", q)[0]


row("\\texttt{spin}", "---", f1(rows["onoff88"][0][2]),
    *["%.1f" % (100 * (slowdown(t, "spin", q) - 1)) for q in ("p99", "median") for t in NETS])
for cfg in ("peak-T0010", "peak-T0100", "peak-T0300", "peak-T1000", "peak-T3000"):
    peak = int(cfg[-4:]) / 1000                  # the ON part in ms; the period is ten times it
    probe = st.median(np.percentile(v, 99) for v in M.windows(cfg, os.path.join(S.BENCH, S.GEOMS["smooth88"]["rep"])))
    row("%g" % peak, "%g" % (10 * peak), "%.1f" % probe,
        *["%.1f" % (100 * (slowdown(t, cfg, q) - 1)) for q in ("p99", "median") for t in NETS])
print("\\midrule")
print("\\multicolumn{3}{l}{Inference alone (ms)} & " + " & ".join(
    "%.2f" % A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "median")[0] for t in NETS) + " & & & & \\\\")

# ------------------------------------------------------------------ the temporal model

head("tab:stress-temporal-stat (measured; median of the averages: L_eq, pred, error; p99 of the averages: the same)")
tm = {v: comparison(v) for v in ("infwin88", "infwin88p")}
last = None
for a, b in zip(tm["infwin88"], tm["infwin88p"]):
    t = a["target"]
    if last is not None and t != last:
        print("\\midrule")
    row(t if t != last else "", a["corunner"], pct(a["measured_p99"]),
        *["%s & %s & %s" % (f1(r["L_eq"]), pct(r["pred_p99_p99"]), pts(float(r["err_p99_p99_pts"]))) for r in (a, b)])
    last = t
print("\\midrule")
e = [[float(r["err_p99_p99_pts"]) for r in tm[v]] for v in ("infwin88", "infwin88p")]
for lab, i in (("mean $|e|$", 0), ("max $|e|$", 1), ("mean $e$", 2), ("under-predicted", 3)):
    print("\\multicolumn{3}{l}{%s} & & & %s & & & %s \\\\" % (lab, summary4(e[0])[i], summary4(e[1])[i]))

# ------------------------------------------------------------------ the counter model

head("tab:counter-variants (twelve pairs leave one network out; thirty pairs frozen)")
for lab, v in (("DRAM", "dram"), ("L2", "l2"), ("DRAM and L2", "ref")):
    cells = []
    for f in ("counters-%s" % v, "counters-new-%s" % v):
        e = np.array([float(r["err_p99_p99_pts"]) for r in comparison(f)])
        cells.append("%.1f & %s & %d & %s" % (np.abs(e).mean(), pts(e.mean()), (e < 0).sum(), pts(min(e.min(), 0))))
    row(lab, *cells)

# ------------------------------------------------------------------ composition

head("tab:stress-compose (a, b, a alone, b alone, sum, together, ratio, composition)")
cp = read("compose-4sm.csv")
alone = {r["a"]: float(r["p99_sum"]) / 2 for r in cp if r["a"] == r["b"]}   # the diagonal: twice one stressor
last = None
for r in cp:
    if last is not None and r["a"] != last:
        print("\\midrule")
    x = float(r["p99_ratio"])
    kind = "additive" if 0.95 <= round(x, 2) <= 1.05 else ("saturating$^{*}$" if x < 1 else "amplified")
    row("%d\\,\\%%" % int(r["a"]), "%d\\,\\%%" % int(r["b"]), "%.1f" % alone[r["a"]], "%.1f" % alone[r["b"]],
        f1(r["p99_sum"]), f1(r["p99_leq"]), "%.2f" % x, kind)
    last = r["a"]
leq = {(r["a"], r["b"]): float(r["p99_leq"]) for r in cp}
order = [abs(leq[(a, b)] - leq[(b, a)]) for a, b in leq if a < b]
print("% median ratio {:.2f}-{:.2f}; traffic together / alone {:.0f}-{:.0f} %; (a,b) against (b,a) {:.1f}-{:.1f} GB/s".format(
    min(float(r["median_ratio"]) for r in cp), max(float(r["median_ratio"]) for r in cp),
    100 * min(float(r["traffic_ratio"]) for r in cp), 100 * max(float(r["traffic_ratio"]) for r in cp), min(order), max(order)))
