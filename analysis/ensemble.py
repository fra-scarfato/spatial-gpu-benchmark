#!/usr/bin/env python3
"""ensemble.py - the p99 models side by side, their max, and the margin that makes the max a bound.

    python3 analysis/ensemble.py            tables, ensembles, margin curve; writes the CSVs
    python3 analysis/ensemble.py new        the frozen ensembles on the validation pairs
    python3 analysis/ensemble.py selftest   the max, the correction and its folds

The question (notes/p99-models.md, "Comparison and max"): do the models fail on the same pairs, or
does one cover another's under-predictions? If the max of several models is a bound with little
extra pessimism, the admission controller can use it; a model that never sets the max is redundant.

The models, all on the 12 pairs of 8 + 8 SMs (exp24), p99 slowdown:
  Smooth     today's stress-sensitivity model (stress.py compare smooth88)
  Temporal   Smooth with the stress read as the p99 of inference-long windows (infwin88p,
             temporal-model.py, the dispatch inside the curve)
  DRAM, DRAM+L2, L2   the counter model's variants (counter-model.py: dram, ref, l2)
Each ensemble is max(Smooth, Temporal, one counter variant).

Two versions of every ensemble:
  raw        the predictions as each model gives them;
  corrected  each model shifted by minus its mean error (bias removal, no safety margin), learnt
             leaving one network out with the counter model's folds: fold k predicts the 3 pairs
             with k as target and learns its shift from the 6 pairs without k. Smooth and Temporal
             fit nothing on the pairs, so their shift is the mean of their errors on those 6 pairs.
             The counters are refitted on the same 6 pairs (as counter-model.loo does) and their
             shift is the mean in-sample residual of that fit, so no pair ever corrects itself.
The margin that turns the corrected max into a bound is kept apart and only described: for
0..8 points added, how many pairs stay under and how much pessimism it costs. Which margin a
criticality level deserves is the user's choice (an input of the admission controller), and the
correction is not a margin, so the two are never stacked (agreed with the user, 2026-10-13).

Errors are predicted - measured slowdown, in points: negative means under-predicted. The median is
carried through only so that the written CSVs have models.py's columns; the goal is the p99.

The validation set (new, 2026-10-13): the 30 pairs of stress-validation-new, on networks no model
saw (the four in INT8, EfficientNet-B0, the ViT). Every member is frozen: its predictions come from
stress.py smoothnew, temporal-model.py new-predict and counter-model.py new-predict, written before the
pairs were measured; its correction is minus its mean error on all 12 development pairs (for the
counters, the mean in-sample residual of the fit on the 12, whose coefficients they use). The margin
curve is printed again on these pairs, by block, with the margin the development curve gave (the
smallest that left no development pair under): the test of the margin, which the 12 pairs cannot be.

ponytail: 12 pairs, 4 networks. The margin curve is the development set's, not a coverage
guarantee; the real check is new networks after freezing (CLAUDE.md).
"""
import csv
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("counter-model")   # profile(), measured(), fit(), predict(), the folds
A, P = C.A, C.P
STORED = {"Smooth": "smooth88", "Temporal": "infwin88p"}
COUNTERS = {"DRAM": "dram", "DRAM+L2": "ref", "L2": "l2"}
MODELS = list(STORED) + list(COUNTERS)
STAT = {"p99": 2, "median": 3}                 # the column of C.measured() rows
MARGINS = range(9)           # past the worst corrected under-prediction only the bias grows


def stored(name, q):
    """The predicted slowdown of a model written by stress.py compare: {(t, c): slowdown}."""
    path = os.path.join(C.AGG, "stress-comparison-%s.csv" % name)
    return {(r["target"], r["corunner"]): float(r["pred_%s_p99" % q]) for r in csv.DictReader(open(path))}


def correct(raw, rows, qi):
    """Bias removal of a model with no fit on the pairs: each pair shifted by minus the mean error of
    the pairs that don't contain its target (the fold of counter-model.loo)."""
    return {k: p - np.mean([raw[r[:2]] - r[qi] for r in rows if k[0] not in r[:2]]) for k, p in raw.items()}


def counter(rows, prof, X, qi):
    """The counter model leaving one network out, raw and corrected. Fold k fits on the 6 pairs without
    k, predicts the pairs with k as target, and shifts them by minus the fit's mean in-sample residual
    (least squares without an intercept need not have a zero-mean residual)."""
    raw, cor = {}, {}
    for k in C.NETS:
        tr = [r for r in rows if k not in r[:2]]
        b = C.fit(tr, prof, X, qi)
        bias = np.mean([C.predict(r[0], r[1], prof, X, b) - r[qi] for r in tr])
        for r in rows:
            if r[0] == k:
                raw[r[:2]] = C.predict(r[0], r[1], prof, X, b)
                cor[r[:2]] = raw[r[:2]] - bias
    return raw, cor


def ensemble(pred, members):
    """Max over the members, per pair: ({(t, c): slowdown}, {(t, c): the member that sets it})."""
    keys = pred[members[0]]
    who = {k: max(members, key=lambda m: pred[m][k]) for k in keys}
    return {k: pred[who[k]][k] for k in keys}, who


def errors(p, rows, qi):
    return np.array([(p[r[:2]] - r[qi]) * 100 for r in rows])


def score(e):
    """mean |e|, max |e|, pairs under, worst under (0 if none), bias, mean over-prediction."""
    return np.abs(e).mean(), np.abs(e).max(), int((e < 0).sum()), min(e.min(), 0), e.mean(), np.maximum(e, 0).mean()


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def report(title, pred, rows):
    """Everything for one version (raw or corrected) of the p99 predictions; returns the ensembles."""
    qi = STAT["p99"]
    err = {m: errors(pred[m], rows, qi) for m in MODELS}
    ens = {v: ensemble(pred, ["Smooth", "Temporal", v]) for v in COUNTERS}
    eerr = {v: errors(ens[v][0], rows, qi) for v in COUNTERS}
    print("\n===== %s" % title)

    print("\np99 slowdown %: measured, predicted (error, points)")
    print("  %-10s %-10s %6s" % ("target", "co-runner", "meas") + "".join("%16s" % m for m in MODELS))
    for i, r in enumerate(rows):
        print("  %-10s %-10s %6.1f" % (r[0], r[1], (r[qi] - 1) * 100) +
              "".join("   %5.1f (%+5.1f)" % ((pred[m][r[:2]] - 1) * 100, err[m][i]) for m in MODELS))

    print("\nmax(Smooth, Temporal, X): error of the max (points), the model that sets it")
    print("  %-10s %-10s" % ("target", "co-runner") + "".join("%22s" % ("X = " + v) for v in COUNTERS))
    for i, r in enumerate(rows):
        print("  %-10s %-10s" % r[:2] + "".join("%15s %+6.1f" % (ens[v][1][r[:2]], eerr[v][i])
                                               for v in COUNTERS))

    print("\nscores (points): each model alone, then each max")
    print("  %-22s %13s %6s %7s %7s %7s" % ("model", "mean/max |e|", "under", "worst", "bias", "over"))
    for name, e in [(m, err[m]) for m in MODELS] + [("max(S, T, %s)" % v, eerr[v]) for v in COUNTERS]:
        s = score(e)
        print("  %-22s %6.1f / %4.1f %6d %7.1f %+7.1f %7.1f" % ((name,) + s))

    print("\ndominance: pairs on which each member sets the max")
    for v in COUNTERS:
        print("  max(S, T, %-8s" % (v + ")") + "".join("  %s %2d" % (m, list(ens[v][1].values()).count(m))
                                                       for m in ("Smooth", "Temporal", v)))

    print("\nagreement: pairs both under-predict (upper), Spearman of the signed errors (lower)")
    print("  %-9s" % "" + "".join("%9s" % m for m in MODELS))
    for a in MODELS:
        cells = []
        for b in MODELS:
            if a == b:
                cells.append("%9d" % (err[a] < 0).sum())
            elif MODELS.index(b) > MODELS.index(a):
                cells.append("%9d" % ((err[a] < 0) & (err[b] < 0)).sum())
            else:
                cells.append("%+9.2f" % spearman(err[a], err[b]))
        print("  %-9s" % a + "".join(cells))
    return ens


def write(name, rows, ens):
    """stress.py compare's columns (models.py reads them) plus the member that sets the max."""
    path = os.path.join(C.AGG, "stress-comparison-%s.csv" % name)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "corunner", "measured_p99", "measured_median", "pred_p99_p99", "err_p99_p99_pts",
                    "pred_median_p99", "err_median_p99_pts", "sets_max_p99"])
        for r in rows:
            p, m = ens["p99"][0][r[:2]], ens["median"][0][r[:2]]
            w.writerow([r[0], r[1], round(r[2], 4), round(r[3], 4), round(p, 4), round((p - r[2]) * 100, 4),
                        round(m, 4), round((m - r[3]) * 100, 4), ens["p99"][1][r[:2]]])
    print("  wrote", path)


def main():
    data = A.load(P.BENCH)
    prof = C.profile(data)
    rows = C.measured(data)
    raw, cor = {q: {} for q in STAT}, {q: {} for q in STAT}
    for q, qi in STAT.items():
        for m, name in STORED.items():
            raw[q][m] = stored(name, q)
            cor[q][m] = correct(raw[q][m], rows, qi)
        for m, v in COUNTERS.items():
            raw[q][m], cor[q][m] = counter(rows, prof, C.VARIANTS[v], qi)

    # The stored CSVs and counter-model.py must agree with this script's measured pairs and folds.
    for m, name in list(STORED.items()) + [(m, "counters-" + v) for m, v in COUNTERS.items()]:
        f = {(r["target"], r["corunner"]): r for r in csv.DictReader(open(os.path.join(C.AGG,
             "stress-comparison-%s.csv" % name)))}
        for r in rows:
            assert abs(float(f[r[:2]]["measured_p99"]) - r[2]) < 1e-3, (name, r)
            assert abs(float(f[r[:2]]["pred_p99_p99"]) - raw["p99"][m][r[:2]]) < 1e-3, (name, r)

    ens = {}
    for version, pred in (("raw", raw), ("corrected", cor)):
        ens[version] = report("%s predictions (p99)" % version, pred["p99"], rows)
        for v, short in COUNTERS.items():
            write("max%s-%s" % ("" if version == "raw" else "c", short), rows,
                  {q: ensemble(pred[q], ["Smooth", "Temporal", v]) for q in STAT})

    print("\nmargin curve on the corrected max: margin m points added to every prediction")
    print("  %4s" % "m" + "".join("%26s" % ("max(S, T, %s)" % v) for v in COUNTERS))
    print("  %4s" % "" + "".join("%26s" % "under  worst   bias" for v in COUNTERS))
    e0 = {v: errors(ens["corrected"][v][0], rows, STAT["p99"]) for v in COUNTERS}
    for m in MARGINS:
        print("  %4d" % m + "".join("%11d %7.1f %+6.1f" % score(e0[v] + m)[2:5] for v in COUNTERS))


NEW_FILES = {"Smooth": "smoothnew", "Temporal": "infwinnewp", "DRAM": "counters-new-dram",
             "DRAM+L2": "counters-new-ref", "L2": "counters-new-l2"}


def block(t, c):
    """The validation block of a pair: INT8, EfficientNet, ViT, or EfficientNet with the ViT."""
    s = {t, c}
    if all(n.endswith("int8") for n in s):
        return "INT8"
    return {frozenset(["efficientnet"]): "EfficientNet", frozenset(["vit"]): "ViT"}.get(
        frozenset(s & {"efficientnet", "vit"}), "EfficientNet+ViT")


def frozen_shift(rows, prof):
    """Each member's correction learnt on all 12 development pairs (p99): minus its mean error."""
    qi = STAT["p99"]
    shift = {m: -np.mean([stored(n, "p99")[r[:2]] - r[qi] for r in rows]) for m, n in STORED.items()}
    for m, v in COUNTERS.items():
        b = C.fit(rows, prof, C.VARIANTS[v], qi)
        shift[m] = -np.mean([C.predict(r[0], r[1], prof, C.VARIANTS[v], b) - r[qi] for r in rows])
    return shift


def dev_margin(rows, prof):
    """Per counter variant, the smallest margin that left no development pair under the corrected max."""
    cor = {}
    for m, name in STORED.items():
        cor[m] = correct(stored(name, "p99"), rows, STAT["p99"])
    for m, v in COUNTERS.items():
        cor[m] = counter(rows, prof, C.VARIANTS[v], STAT["p99"])[1]
    out = {}
    for v in COUNTERS:
        e = errors(ensemble(cor, ["Smooth", "Temporal", v])[0], rows, STAT["p99"])
        out[v] = next(m for m in range(100) if (e + m >= 0).all())
    return out


def new():
    data = A.load(P.BENCH)
    prof = C.profile(data)
    rows = C.measured(data)
    shift, margin = frozen_shift(rows, prof), dev_margin(rows, prof)
    meas, raw = {}, {}
    for m, name in NEW_FILES.items():
        path = os.path.join(C.AGG, "stress-comparison-%s.csv" % name)
        if not os.path.exists(path):
            raise SystemExit("ensemble.py: %s missing; run its new-compare (or stress.py compare smoothnew)" % path)
        f = {(r["target"], r["corunner"]): r for r in csv.DictReader(open(path))}
        raw[m] = {k: float(r["pred_p99_p99"]) for k, r in f.items()}
        for k, r in f.items():
            meas.setdefault(k, float(r["measured_p99"]))
            assert abs(meas[k] - float(r["measured_p99"])) < 1e-3, (m, k)
    keys = sorted(meas, key=lambda k: (block(*k), k))
    assert all(set(raw[m]) == set(keys) for m in MODELS), "the members cover different pairs"
    rows_new = [(t, c, meas[(t, c)]) for t, c in keys]
    cor = {m: {k: p + shift[m] for k, p in raw[m].items()} for m in MODELS}

    print("frozen corrections (points, from the 12 development pairs):",
          ", ".join("%s %+.1f" % (m, 100 * shift[m]) for m in MODELS))
    print("development margins (smallest with no development pair under):",
          ", ".join("max(S, T, %s) %d" % (v, m) for v, m in margin.items()))
    for title, pred in (("raw", raw), ("corrected", cor)):
        ens = report("validation pairs, %s predictions (p99)" % title, pred, rows_new)
        if title == "corrected":
            print("\nscores by block, corrected max, with the development margin (points)")
            print("  %-17s %-14s %5s %13s %6s %7s %7s" % ("block", "ensemble", "pairs", "mean/max |e|", "under", "worst", "bias"))
            for b in ("INT8", "EfficientNet", "ViT", "EfficientNet+ViT", "all"):
                sel = [i for i, r in enumerate(rows_new) if b == "all" or block(*r[:2]) == b]
                for v in COUNTERS:
                    e = errors(ens[v][0], rows_new, 2)[sel] + margin[v]
                    s = score(e)
                    print("  %-17s %-14s %5d %6.1f / %4.1f %6d %7.1f %+7.1f" % ((b, "max(S,T,%s)" % v, len(sel)) + s[:5]))
            print("\nmargin curve on the validation pairs, corrected max (under / worst / bias)")
            print("  %4s" % "m" + "".join("%26s" % ("max(S, T, %s)" % v) for v in COUNTERS))
            e0 = {v: errors(ens[v][0], rows_new, 2) for v in COUNTERS}
            for m in MARGINS:
                print("  %4d" % m + "".join("%11d %7.1f %+6.1f" % score(e0[v] + m)[2:5] for v in COUNTERS))


def selftest():
    nets = ["a", "b", "c", "d"]
    rows = [(t, c, 1.2, 1.1) for t in nets for c in nets if t != c]
    biased = {r[:2]: r[2] + 0.03 for r in rows}                    # +3 points everywhere
    assert all(abs(v - 1.2) < 1e-12 for v in correct(biased, rows, 2).values())
    spike = {r[:2]: r[2] + (1.0 if r[:2] == ("a", "b") else 0) for r in rows}
    cs = correct(spike, rows, 2)
    assert abs(cs[("a", "b")] - 2.2) < 1e-12                       # its own fold never saw it
    assert cs[("c", "d")] < 1.2                                     # fold c did
    pred = {"x": {k: 1.0 for k in biased}, "y": {k: 1.5 for k in biased}}
    pred["x"][("a", "b")] = 2.0
    p, who = ensemble(pred, ["x", "y"])
    assert who[("a", "b")] == "x" and p[("a", "b")] == 2.0 and who[("b", "a")] == "y" and p[("b", "a")] == 1.5
    assert [block(*k) for k in (("resnetint8", "mobilenetint8"), ("vit", "resnet"), ("mobilenet", "efficientnet"),
                                ("efficientnet", "vit"))] == ["INT8", "ViT", "EfficientNet", "EfficientNet+ViT"]
    print("  selftest passed")


if __name__ == "__main__":
    {"selftest": selftest, "new": new}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
