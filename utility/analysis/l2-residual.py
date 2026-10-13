#!/usr/bin/env python3
"""l2-residual.py - does the temporal stress model leave errors that point to the L2? (prints only)

    python3 utility/analysis/l2-residual.py            tests A and B on both rulers, then C
    python3 utility/analysis/l2-residual.py selftest   the additive split

The question comes before building a second, L2-resident reporter (model 3 in notes/p99-models.md):
only if the temporal model (window p99, infwin-model.py) leaves errors that the L2 explains is a
second coordinate worth its probe, calibration and runs.

Test A, the decisive one. The L2 chapter ran MobileNet beside co-runners that move the same DRAM
traffic but use the L2 differently (exp22, l2-vs-dram: compute_gemm against SAXPY L0.243, 22.3 GB/s;
the 288-cubed SGEMM over 8 sets against SAXPY L0.087, 8.0 GB/s), and the DRAM reporter beside each
(stress-reporter-target/l2d-*). Each arm goes through the temporal model: MobileNet's T, window p99,
the ruler and MobileNet's curve. The residual is predicted - measured p99 slowdown, against exp22's
own alone row. The co-runners run on 4 SMs (50:25) while the rulers and curves are 8 + 8, so only
the difference between an L2 co-runner and its DRAM-matched SAXPY is read:

  delta = residual(L2 co-runner) - residual(matched SAXPY)

Criterion, fixed before the run: the error is L2-attributable if delta < 0 for both pairs and
|delta| exceeds the summed run-to-run spread (max - min over the runs) of the two arms' measured
p99 slowdown.

Test B, descriptive (4 networks). The 12 pairs' errors of the same model are split additively,
error = a_target + b_corunner (the b centred on zero), and the co-runner effect is set against the
L2 traffic that never reaches the DRAM, L*h, which a DRAM-latency reporter cannot see; its DRAM GB/s
D is the control. The target effect is set against the target's hit rate h. The indicators are model
1's isolated profiles (counter-model.py). Spearman over 4 points carries no significance.

Test C, the cheapest stand-in for an L2 coordinate: a measured L2 term added to a stress model,
pred + g * L_corunner (L2 GB/s alone, counters), with g fitted leave-one-network-out on the 8 + 8
pairs as in counter-model.py (the user's OK, 2026-10-13). If even this does not lower the errors,
an L2 reporter, an indirect reading of the same pressure, is not expected to. The same term in the
co-runner's DRAM GB/s is the control. Bases: Smooth (smooth88) and Smooth + window p99 (infwin88p).
"""
import csv
import importlib
import os
import sys

import numpy as np

CMD = sys.argv[1] if len(sys.argv) > 1 else ""   # read first: avg-model rewrites sys.argv for stress.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = importlib.import_module("infwin-model")    # aggregated(), rulers(), and through it avg-model, stress
C = importlib.import_module("counter-model")   # profile(): the isolated DRAM and L2 indicators
H, S, A = W.H, W.S, W.A
P99 = W.STATS["p"]
L2D = os.path.join(S.BENCH, "stress-reporter-target")
ARMS = ["spin-K100", "saxpy-L0087", "gemm-N8", "saxpy-L0243", "computegemm"]
MATCHED = [("computegemm", "saxpy-L0243"), ("gemm-N8", "saxpy-L0087")]


def test_a(data, name, ruler, T):
    """Residual and spread (points) of each l2-vs-dram arm under one ruler; the deltas and verdict."""
    rv = [W.aggregated(H.windows(c, ruler["rep"]), T, P99)[0] for c in ruler["cfgs"]]
    curve = ruler["curves"]["mobilenet"]
    alone = A.aggr(data, "exp22", "mobilenet-alone", "mobilenet", "gpu_ms", "p99")
    res = {}
    print("\nTest A, %s ruler (MobileNet, T %.2f ms, window p99)" % (name, T))
    print("  %-12s %9s %8s %9s %9s %10s %8s" % ("co-runner", "coord ns", "L_eq", "pred", "measured", "residual", "spread"))
    for arm in ARMS:
        y, _ = W.aggregated(H.windows("l2d-" + arm, L2D), T, P99)
        leq, inside = S.invert(y, ruler["levels"], rv)
        pred = float(np.interp(leq, [p[0] for p in curve], [p[1] for p in curve]))
        m, lo, hi, _ = A.aggr(data, "exp22", "mobilenet-" + arm, "mobilenet", "gpu_ms", "p99")
        meas = m / alone[0]
        res[arm] = ((pred - meas) * 100, (hi - lo) / alone[0] * 100)
        print("  %-12s %9.1f %8.3f%s %8.3f %9.3f %+10.1f %8.1f" % (arm, y, leq, " " if inside else "*", pred, meas, *res[arm]))
    ok = True
    for l2, sax in MATCHED:
        d = res[l2][0] - res[sax][0]
        spread = res[l2][1] + res[sax][1]
        ok &= d < 0 and -d > spread
        print("  delta %-11s - %-11s %+6.1f points (spread %.1f)" % (l2, sax, d, spread))
    print("  verdict: %s" % ("L2-attributable" if ok else "not L2-attributable by the criterion"))


def split(err):
    """err {(t, c): e} -> (a_t, b_c): least squares of e = a_t + b_c, with the b centred on zero."""
    ts, cs = sorted({k[0] for k in err}), sorted({k[1] for k in err})
    X = np.array([[t == k[0] for t in ts] + [c == k[1] for c in cs] for k in err], float)
    x = np.linalg.lstsq(X, np.array(list(err.values())), rcond=None)[0]
    a, b = x[:len(ts)], x[len(ts):]
    a, b = a + b.mean(), b - b.mean()     # the split is defined up to this constant
    return dict(zip(ts, a)), dict(zip(cs, b))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def test_b(name, prof):
    path = os.path.join(W.AGG, "stress-comparison-%sp.csv" % name)
    err = {(r["target"], r["corunner"]): float(r["err_p99_p99_pts"]) for r in csv.DictReader(open(path))}
    a, b = split(err)
    nets = sorted(b)
    print("\nTest B, %s (12 pairs, p99 error = a_target + b_corunner, points)" % name)
    print("  %-10s %8s %8s %10s %7s %8s" % ("network", "a_target", "b_co", "L*h GB/s", "D GB/s", "hit %"))
    for n in nets:
        x = prof[n]
        print("  %-10s %+8.1f %+8.1f %10.1f %7.1f %8.1f" % (n, a[n], b[n], x["L"] * x["h"], x["D"], 100 * x["h"]))
    print("  Spearman: b_co with L*h %+.2f, with D %+.2f; a_target with hit rate %+.2f (n = 4, no significance)"
          % (spearman([b[n] for n in nets], [prof[n]["L"] * prof[n]["h"] for n in nets]),
             spearman([b[n] for n in nets], [prof[n]["D"] for n in nets]),
             spearman([a[n] for n in nets], [prof[n]["h"] for n in nets])))
    print("  (RetinaNet's ncu L2 is suspect: L2 x miss 477 against 819 MB/inf on the DRAM counters)")


def test_c(prof):
    print("\nTest C: base model + g x co-runner load, g fitted leaving one network out (p99, points)")
    print("  %-11s %-10s %13s %6s %7s %7s %8s" % ("base", "term", "mean/max", "under", "worst", "bias", "g full"))
    for base in ("smooth88", "infwin88p"):
        rows = list(csv.DictReader(open(os.path.join(W.AGG, "stress-comparison-%s.csv" % base))))
        err = np.array([float(r["err_p99_p99_pts"]) for r in rows])
        for term, key in (("none", None), ("L2 GB/s", "L"), ("DRAM GB/s", "D")):
            new = err.copy()
            if key:
                x = np.array([prof[r["corunner"]][key] for r in rows])
                for k in C.NETS:
                    tr = np.array([k not in (r["target"], r["corunner"]) for r in rows])
                    g = -np.dot(x[tr], err[tr]) / np.dot(x[tr], x[tr])
                    te = np.array([r["target"] == k for r in rows])
                    new[te] = err[te] + g * x[te]
                gfull = -np.dot(x, err) / np.dot(x, x)
            print("  %-11s %-10s %6.1f / %4.1f %6d %7.1f %+7.1f %8s" % (base, term, np.abs(new).mean(), np.abs(new).max(),
                  (new < 0).sum(), min(new.min(), 0), new.mean(), "%.4f" % gfull if key else ""))


def main():
    data = A.load(S.BENCH)
    T = A.aggr(data, "exp23", "mobilenet-alone", "mobilenet", "gpu_ms", "median")[0]
    prof = C.profile(data)
    for name, ruler in W.rulers().items():
        print("\n===== %s" % name)
        test_a(data, name, ruler, T)
        test_b(name, prof)
    test_c(prof)


def selftest():
    a = {"x": 1.0, "y": -2.0, "z": 0.5}
    b = {"x": 3.0, "y": -1.0, "z": -2.0}            # centred: sums to zero
    a2, b2 = split({(t, c): a[t] + b[c] for t in a for c in b if t != c})
    assert all(abs(a2[k] - a[k]) < 1e-9 and abs(b2[k] - b[k]) < 1e-9 for k in a)
    assert abs(spearman([1, 2, 3, 4], [10, 20, 40, 30]) - 0.8) < 1e-12
    print("  selftest passed")


if __name__ == "__main__":
    {"selftest": selftest}.get(CMD, main)()
