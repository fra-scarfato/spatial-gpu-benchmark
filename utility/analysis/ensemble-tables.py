#!/usr/bin/env python3
"""ensemble-tables.py - the thesis tables of the three p99 models, their correction, maximum and margin.

    python3 utility/analysis/ensemble-tables.py     every table as LaTeX rows, then the prose numbers

A regenerator, not a record: it reads the current model and the CSVs the analysis has written in
$DATA/bench/aggregate, so after a change, re-run the chain that writes them (stress.py compare
smooth88 / smoothnew, infwin-model.py, counter-model.py, its new-predict / new-compare, ensemble.py)
and then this, and paste. It writes nothing. Each block is headed by its label in stress-model.tex:

  tab:models-dev         the twelve 8+8 pairs: measured, each model's raw prediction and error
  tab:models-val         the thirty pairs with the new networks, the same
  tab:bound-dev          the twelve pairs: each model corrected, and their maximum
  tab:bound-val          the thirty pairs: the same, with the corrections frozen on the twelve
  tab:stress-margin      the corrected maximum of the three plus a margin of 0-5 points
  tab:stress-pessimism   each corrected model and two maxima at the margin that covers the twelve pairs

The models are ensemble.py's: Smooth (smooth88 / smoothnew), Temporal (infwin88p / infwinnewp) and the
counter model with the L2 input, leave one network out on the twelve pairs, frozen on the thirty. The
corrections are ensemble.py's (correct, counter, frozen_shift). Errors are predicted minus measured
p99 slowdown, in points. Rows follow the comparison CSVs, which follow the thesis's order.
"""
import csv
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = importlib.import_module("ensemble")
C, A, P = E.C, E.A, E.P
QI = E.STAT["p99"]
THREE = ["Smooth", "Temporal", "L2"]


def pts(x):
    """Signed points for LaTeX: $+$1.6, $-$0.3, and 0.0 when it rounds to zero."""
    s = "%.1f" % abs(x)
    return "0.0" if s == "0.0" else ("$+$" if x > 0 else "$-$") + s


def worst(e):
    """The largest under-prediction: $-$0.0 when a pair is under by less than 0.05, 0.0 when none is."""
    w = min(e.min(), 0)
    return "$-$%.1f" % -w if w < 0 else "0.0"


def name(n):
    return n.replace("int8", "-int8")


def head(label):
    print("\n%% ---- %s" % label)


# ------------------------------------------------------------------ predictions

data = A.load(P.BENCH)
prof, rows = C.profile(data), C.measured(data)
raw_l2, cor_l2 = E.counter(rows, prof, C.VARIANTS["l2"], QI)
raw = {"dev": {"Smooth": E.stored("smooth88", "p99"), "Temporal": E.stored("infwin88p", "p99"), "L2": raw_l2}}
cor = {"dev": {m: E.correct(E.stored(n, "p99"), rows, QI) for m, n in E.STORED.items()}}
cor["dev"]["L2"] = cor_l2
shift = E.frozen_shift(rows, prof)
raw["new"] = {m: E.stored(E.NEW_FILES[m], "p99") for m in THREE}
cor["new"] = {m: {k: p + shift[m] for k, p in raw["new"][m].items()} for m in THREE}


def measured(name):
    """[(target, co-runner, measured p99 slowdown)] in the order of a comparison CSV."""
    path = os.path.join(C.AGG, "stress-comparison-%s.csv" % name)
    return [(r["target"], r["corunner"], float(r["measured_p99"])) for r in csv.DictReader(open(path))]


# The twelve pairs take counter-model's exact measurement, in the order of the smooth88 CSV, whose
# copy is rounded to four decimals (enough to flip a rounded error); the thirty pairs, the CSV's.
meas_dev = {r[:2]: r[QI] for r in rows}
assert all(abs(m - meas_dev[(t, c)]) < 1e-3 for t, c, m in measured("smooth88"))
pairs = {"dev": [(t, c, meas_dev[(t, c)]) for t, c, m in measured("smooth88")], "new": measured("counters-new-l2")}


def with_max(pred, members):
    return E.ensemble(pred, members)[0] if len(members) > 1 else pred[members[0]]


def errs(p, s):
    return np.array([(p[(t, c)] - m) * 100 for t, c, m in pairs[s]])


def table(label, s, pred, cols):
    """The per-pair rows of one table and its four summary rows."""
    head(label)
    last = None
    for t, c, m in pairs[s]:
        if last is not None and t != last:
            print("\\midrule")
        cells = [name(t) if t != last else "", name(c), "%.1f" % (100 * (m - 1))]
        for p in cols:
            cells += ["%.1f" % (100 * (p[(t, c)] - 1)), pts((p[(t, c)] - m) * 100)]
        print(" & ".join(cells) + " \\\\")
        last = t
    print("\\midrule")
    e = [errs(p, s) for p in cols]
    for lab, f in (("mean $|e|$", lambda x: "%.1f" % np.abs(x).mean()), ("max $|e|$", lambda x: "%.1f" % np.abs(x).max()),
                   ("mean $e$", lambda x: pts(x.mean())), ("under-predicted", lambda x: "%d" % (x < 0).sum())):
        print("\\multicolumn{3}{l}{%s} & " % lab + " & ".join("& " + f(x) for x in e) + " \\\\")


table("tab:models-dev", "dev", raw["dev"], [raw["dev"][m] for m in THREE])
table("tab:models-val", "new", raw["new"], [raw["new"][m] for m in THREE])
table("tab:bound-dev", "dev", cor["dev"], [cor["dev"][m] for m in THREE] + [with_max(cor["dev"], THREE)])
table("tab:bound-val", "new", cor["new"], [cor["new"][m] for m in THREE] + [with_max(cor["new"], THREE)])

head("tab:stress-margin (corrected max of the three: twelve pairs, then thirty new pairs)")
e0 = {s: errs(with_max(cor[s], THREE), s) for s in ("dev", "new")}
for m in range(6):
    print("%d & " % m + " & ".join("%d & %s & %s" % ((e0[s] + m < 0).sum(), worst(e0[s] + m), pts((e0[s] + m).mean()))
                                   for s in ("dev", "new")) + " \\\\")

head("tab:stress-pessimism (thirty new pairs, at the smallest margin with no development pair under)")
for lab, ms in (("Counters (L2)", ["L2"]), ("Temporal", ["Temporal"]), ("Smooth", ["Smooth"]), (None, None),
                ("Max of the three", THREE), ("Max of temporal and counters", ["Temporal", "L2"]),
                ("Max of the two stress models", ["Smooth", "Temporal"])):
    if lab is None:
        print("\\midrule")
        continue
    m = next(m for m in range(100) if (errs(with_max(cor["dev"], ms), "dev") + m >= 0).all())
    p = with_max(cor["new"], ms)
    e = errs(p, "new") + m
    excess = np.array([(p[(t, c)] + m / 100) / s - 1 for t, c, s in pairs["new"]]) * 100
    cover = next(mm for mm in range(100) if (errs(p, "new") + mm >= 0).all())
    line = "%s & %d & %d & %s & %s & %s & %.1f \\\\" % (lab, m, (e < 0).sum(), worst(e), pts(e.mean()), pts(e.max()), excess.mean())
    print(("% (prose only) " if "stress models" in lab else "") + line +
          "   %% excess max %.1f %%, all thirty covered at %d" % (excess.max(), cover))

# ------------------------------------------------------------------ prose numbers

print("\n% corrections (points): " + ", ".join("%s %+.1f" % (m, 100 * shift[m]) for m in THREE))
for s in ("dev", "new"):
    who = E.ensemble(cor[s], THREE)[1]
    print("%% %s: the corrected max is set by %s" % (s, ", ".join("%s on %d" % (m, sum(w == m for w in who.values())) for m in THREE)))
    e2 = errs(with_max(cor[s], ["Smooth", "Temporal"]), s)
    print("%%   max of the two stress models: %d under, worst %s" % ((e2 < 0).sum(), worst(e2)))
m3 = next(m for m in range(100) if (e0["dev"] + m >= 0).all())
print("%% under on the new pairs at the development margin %d: %s" % (m3, ", ".join(
    "%s+%s %.2f" % (t, c, x) for (t, c, _), x in zip(pairs["new"], e0["new"] + m3) if x < 0)))
who = E.ensemble(cor["new"], THREE)[1]
print("% largest over-predictions of the max on the new pairs (margin 0), and the other models' errors:")
for x, (t, c, s) in sorted(zip(e0["new"], pairs["new"]), reverse=True)[:3]:
    print("%%   %s+%s %+.1f by %s; %s" % (t, c, x, who[(t, c)], ", ".join(
        "%s %+.1f" % (m, 100 * (cor["new"][m][(t, c)] - s)) for m in THREE if m != who[(t, c)])))
