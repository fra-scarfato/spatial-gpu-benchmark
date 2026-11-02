#!/usr/bin/env python3
"""geometry-tables.py - the thesis tables of the subsections on partition sizes, two co-runners and energy.

    python3 utility/analysis/geometry-tables.py     every table as LaTeX rows, then the prose numbers

A regenerator, not a record: it reads the current model (counter-model.py's presence cost and
counter-model-coeffs.csv, through geometry.py) and geometry.py's CSVs in $DATA/bench/aggregate
(geometry-splits.csv, geometry-triple-compare-current.csv, geometry-power.csv), so after a change of
model or data, re-run geometry.py (splits, triple-current, power) and then this, and paste. It writes
nothing. Each block is headed by its label in stress-model.tex:

  tab:stress-sizes            error by the SMs of target and co-runner (the designated targets only)
  tab:stress-sizes-sens       predicted (b0 + b1 L_t) and measured ((s - 1 - p_t) / L_c) sensitivity
  tab:stress-triple           8+4+4: the error with a factor f on the summed co-runner load
  tab:stress-triple-where     per triple: measured, predicted, error next to each co-runner and both
  tab:stress-energy           energy per inference of each network alone, by size
  tab:stress-energy-together  power and energy of the networks running together

The error is predicted minus measured p99 slowdown, in points. Development rows contain only the four
development networks (geometry.DEV). The f values are the two-SAXPY ratios with both duties at 75 % or
more (compose-4sm.csv, tab:stress-compose): their lowest, highest and mean, read from the CSV, so they
follow a re-run of compose.py.
"""
import csv
import importlib
import os
import statistics as st
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
G = importlib.import_module("geometry")
AGG = os.path.join(G.P.BENCH, "aggregate")
SPLIT = {"412": (4, 12), "610": (6, 10), "124": (12, 4)}


def read(name):
    return list(csv.DictReader(open(os.path.join(AGG, name))))


def pts(x):
    """Signed points for LaTeX: $+$1.6, $-$0.3, and 0.0 when it rounds to zero."""
    s = "%.1f" % abs(x)
    return "0.0" if s == "0.0" else ("$+$" if x > 0 else "$-$") + s


def summary(e):
    """mean e & mean |e| & under & worst, the four columns of every error table."""
    e = np.array(e)
    return "%s & %.1f & %d & %s" % (pts(e.mean()), np.abs(e).mean(), (e < 0).sum(), pts(min(e.min(), 0)))


def row(*cells):
    print(" & ".join(str(c) for c in cells) + " \\\\")


def head(label):
    print("\n%% ---- %s" % label)


# ------------------------------------------------------------------ the model and the measured rows

data = G.A.load(G.P.BENCH)
inp, b, g = G.inputs(data), G.coefficients(), G.dispatch_growth(data)


def sens(t, n):
    """The model's sensitivity of target t on n SMs, per GB/s of co-runner L2 bandwidth."""
    return b[0] + b[1] * inp[t][n]["L_scaled"]


def pred(t, n, others, gg=1.0, f=1.0):
    """Predicted p99 slowdown of t on n SMs next to others, a list of (network, SMs)."""
    return 1 + gg * inp[t][n]["p"] + sens(t, n) * f * sum(inp[o][m]["L_scaled"] for o, m in others)


def is_dev(cfg):
    return all(x in G.DEV for x in cfg.split("+"))


# Designated targets next to one co-runner: the three splits and 8+4. (t, n, others, s, dev).
splits, tri = read("geometry-splits.csv"), read("geometry-triple-compare-current.csv")
pairs, spins = [], defaultdict(list)
for r, (nt, nc) in [(r, SPLIT[r["split"]]) for r in splits] + [(r, (8, 4)) for r in tri if r["geometry"] == "8+4"]:
    if r["position"] != "target":
        continue
    t, s = r["victim"], float(r["measured"])
    if r["config"].endswith("-spin"):
        spins[nt].append((t, s))
        continue
    others = [(r["config"].split("+")[1], nc)]
    assert abs(pred(t, nt, others) - float(r["pred_scaled"])) < 2e-4, r    # the CSV is the current model's
    pairs.append((t, nt, others, s, is_dev(r["config"])))
for t in G.NETS:                                       # the spin on 4 next to a target on 8
    spins[8].append((t, G.slowdown(data, G.TRIPLE, t + "-spin4", t, t + "-alone")))

# Designated targets (on 8) next to two co-runners (on 4 each). (t, others, s, dev, config).
triples = []
for r in tri:
    if r["geometry"] == "8+4+4" and r["position"] == "target":
        t, a, c = r["config"].split("+")
        triples.append((t, [(a, 4), (c, 4)], float(r["measured"]), is_dev(r["config"]), r["config"]))
meas8 = {r["config"]: float(r["measured"]) for r in tri if r["geometry"] == "8+4" and r["position"] == "target"}

heavy = [float(r["p99_ratio"]) for r in read("compose-4sm.csv") if int(r["a"]) >= 75 and int(r["b"]) >= 75]
F = tuple(round(x, 2) for x in (1.0, min(heavy), np.mean(heavy), max(heavy)))   # the value printed is the value used

# ------------------------------------------------------------------ partition sizes

head("tab:stress-sizes (rows: target + co-runner SMs; development, then all)")
for nt, nc in ((4, 12), (6, 10), (8, 4), (12, 4)):
    sel = [x for x in pairs if x[1] == nt]
    e = [(pred(t, n, o) - s) * 100 for t, n, o, s, d in sel]
    ed = [(pred(t, n, o) - s) * 100 for t, n, o, s, d in sel if d]
    row(nt, nc, summary(ed), summary(e))
e = [(pred(t, n, o) - s) * 100 for t, n, o, s, d in pairs]
row("\\multicolumn{2}{l}{all}", summary([x for x, p in zip(e, pairs) if p[4]]), summary(e))

head("tab:stress-sizes-sens (points per 100 GB/s; predicted, then measured median over the pairs)")
for t in G.NETS:
    row(t, "predicted", *("%.1f" % (1e4 * sens(t, n)) for n in (4, 6, 8, 12)))
    row("", "measured", *("%.1f" % (1e4 * st.median((s - 1 - inp[t][n]["p"]) / sum(inp[o][m]["L_scaled"] for o, m in oo)
                                                    for tt, nn, oo, s, d in pairs if tt == t and nn == n))
                          for n in (4, 6, 8, 12)))
    print("\\midrule")
print("% dispatch: 1 + p_t minus the measured slowdown next to spin, points, by the target's SMs")
for n in sorted(spins):
    e = [(1 + inp[t][n]["p"] - s) * 100 for t, s in spins[n]]
    print("%%   %2d SMs: %s  (max |e| %.1f)" % (n, " ".join("%+.1f" % x for x in e), max(map(abs, e))))

# ------------------------------------------------------------------ two co-runners

head("tab:stress-triple (f from compose-4sm.csv, both duties >= 75 %%: %s)" % " ".join("%.3f" % x for x in heavy))
for f in F:
    e = [(pred(t, 8, o, g, f) - s) * 100 for t, o, s, d, c in triples]
    ed = [(pred(t, 8, o, g, f) - s) * 100 for t, o, s, d, c in triples if d]
    row("1 (plain sum)" if f == 1 else "%.2f" % f, summary(ed), summary(e))
print("%% g = %.2f" % g)

head("tab:stress-triple-where (measured, predicted, error: next to the first, the second, both)")
ep, et, above, ratio = [], [], 0, defaultdict(list)
for t0 in G.NETS:
    first = True
    for t, o, s, d, cfg in triples:
        if t != t0:
            continue
        (a, _), (c, _) = o
        ma, mc = meas8[t + "+" + a], meas8[t + "+" + c]
        pa, pc, pt = pred(t, 8, [(a, 4)]), pred(t, 8, [(c, 4)]), pred(t, 8, o, g)
        ep += [(pa - ma) * 100, (pc - mc) * 100]
        et.append((pt - s) * 100)
        above += s - 1 > (ma - 1) + (mc - 1)
        p = inp[t][8]["p"]
        ratio[t].append((s - 1 - g * p) / ((ma - 1 - p) + (mc - 1 - p)))
        row(t if first else "", "%s + %s" % (a, c),
            *("%.1f & %.1f & %s" % (100 * (m - 1), 100 * (q - 1), pts((q - m) * 100)) for m, q in ((ma, pa), (mc, pc), (s, pt))))
        first = False
    print("\\midrule")
allr = [x for v in ratio.values() for x in v]
print("% next to one co-runner: mean |e| {:.1f}, worst {:+.1f}; next to both: mean |e| {:.1f}, worst {:+.1f}".format(
    np.abs(ep).mean(), min(ep), np.abs(et).mean(), min(et)))
print("%% measured next to both above the sum of the two pairs: %d of %d" % (above, len(triples)))
print("%% measured ratio without the dispatch, both / (first + second): median %.2f (%.2f-%.2f)" % (
    st.median(allr), min(allr), max(allr)))
for t in G.NETS:
    print("%%   %-12s %.2f-%.2f" % (t, min(ratio[t]), max(ratio[t])))
grow = [(G.slowdown(data, G.TRIPLE, t + "-spin4x2", t, t + "-alone") - 1) /
        (G.slowdown(data, G.TRIPLE, t + "-spin4", t, t + "-alone") - 1) for t in G.NETS]
print("%% dispatch growth, two spins over one, per network: %s" % " ".join("%.2f" % x for x in grow))

# ------------------------------------------------------------------ power and energy

power = defaultdict(list)
for r in read("geometry-power.csv"):
    power[(r["family"], r["config"])].append(r)


def med(rs, k):
    return st.median(float(r[k]) for r in rs)


# Each size's alone rows: the family whose target has that size, the session of those runs; 10 SMs
# only in stress-sizes. (8 SMs is stress-triple's, the session of the 8+4 and 8+4+4 runs.)
ALONE = {4: ("stress-split-412", "-alone"), 6: ("stress-split-610", "-alone"), 8: ("stress-triple", "-alone"),
         10: ("stress-sizes", "-alone10"), 12: ("stress-split-124", "-alone")}
alone = {(t, n): power[(f, t + suf)] for t in G.NETS for n, (f, suf) in ALONE.items()}

head("tab:stress-energy (mJ per inference alone; 4, 6, 8, 10, 12 SMs)")
for t in G.NETS:
    row(t, *("%.1f" % med(alone[(t, n)], "mJ_per_inference") for n in sorted(ALONE)))
m4, m12 = alone[("mobilenet", 4)], alone[("mobilenet", 12)]
print("%% mobilenet 4 -> 12 SMs: %.1f -> %.1f W, throughput x%.1f" % (
    med(m4, "rails_mW") / 1e3, med(m12, "rails_mW") / 1e3, med(m12, "all_inf_s") / med(m4, "all_inf_s")))
for f, suf, nt, nspin in (("stress-split-124", "-spin", 12, "4"), ("stress-triple", "-spin4", 8, "4"),
                          ("stress-triple", "-spin4x2", 8, "4+4"), ("stress-split-610", "-spin", 6, "10"),
                          ("stress-split-412", "-spin", 4, "12")):
    d = [med(power[(f, t + suf)], "rails_mW") - med(alone[(t, nt)], "rails_mW") for t in G.NETS]
    print("%% spin on %-3s SMs next to a target on %2d adds %.1f W (median over the targets)" % (nspin, nt, st.median(d) / 1e3))

head("tab:stress-energy-together (power W; energy per inference and target throughput, together / alone)")
for label, fam, nt, nc, plus in (("4+12", "stress-split-412", 4, 12, 1), ("6+10", "stress-split-610", 6, 10, 1),
                                 ("8+4", "stress-triple", 8, 4, 1), ("12+4", "stress-split-124", 12, 4, 1),
                                 ("8+4+4", "stress-triple", 8, 4, 2)):
    W, energy, thr = [], [], []
    for (f, cfg), rs in power.items():
        if f != fam or cfg.count("+") != plus or any(int(r["corunner_missing"]) for r in rs):
            continue
        nets = cfg.split("+")
        t_s = med(rs, "target_inf_s")
        c_s = [st.median(float(r["corunner_inf_s"].split(";")[i]) for r in rs) for i in range(plus)]
        e_alone = t_s * med(alone[(nets[0], nt)], "mJ_per_inference") + \
            sum(c * med(alone[(n, nc)], "mJ_per_inference") for c, n in zip(c_s, nets[1:]))
        energy.append(med(rs, "mJ_per_inference") / (e_alone / (t_s + sum(c_s))))
        thr.append(t_s / med(alone[(nets[0], nt)], "target_inf_s"))
        W.append(med(rs, "rails_mW") / 1e3)
    row(label, len(W), "%.1f (%.1f--%.1f)" % (st.median(W), min(W), max(W)),
        *("%.2f (%.2f--%.2f)" % (st.median(x), min(x), max(x)) for x in (energy, thr)))
