#!/usr/bin/env python3
"""counter-model.py - model 1: the target's slowdown from counters of each network alone (no reporter).

    python3 utility/analysis/counter-model.py            profiles, LOO predictions, scores, checks
    python3 utility/analysis/counter-model.py selftest   the fit and the hit-rate aggregation

Each network is characterised once, alone on 8 SMs:
  D    DRAM GB/s, run average of the GPU's own counters (profiles/derived/dram.csv, green context 50 %)
  L    L2 GB/s: ncu L2 bytes per inference (profiles/ncu/nets, 4 SMs; bytes per inference don't
       depend on the SM budget, MEASUREMENTS.md) times the inferences per second at 8 SMs
  h    L2 hit rate, weighted by L2 bytes
  p    presence: kernels per inference x 0.87 us per kernel boundary over the latency alone (exp23),
       the dispatch cost no memory counter sees (tab:l2-presence: 0.83-0.91 us per boundary)

The model, with s = slowdown - 1 (p99 and median fitted separately) and target t, co-runner c:
  dram   s - p_t = (b0 + b1 D_t) D_c
  l2     s - p_t = (b0 + b1 L_t) L_c
  ref    s - p_t = (b0 + b1 D_t) D_c + (b2 + b3 L_t) L_c
No free intercept: at zero load only presence is left. b1 and b3 are the target's sensitivity, read
from its own counters, so a new network needs only its isolated profile (gpulet, ATC 2022, fits the
same linear form on the alone L2 and DRAM utilisations of both models).

The coefficients are fitted on the 8 + 8 pairs (exp24), leaving one network out: in fold k the fit
sees the 6 pairs without k, and predicts the 3 pairs with k as target. This is an exception to the
stress model's rule that the pairs never enter the model, agreed with the user on 2026-10-13; the
real test is new networks after freezing. The fit on all 12 pairs is the frozen model
(counter-model-coeffs.csv).

Several co-runners: aggregate() sums D and L and takes the combined hit rate from the sums, never as
an average of rates. Presence is not summed: two co-runners cost less than twice one (fact 1 in
notes/p99-models.md). The checks print the measured traffic of two networks together against the sum
of their alone runs, which the sum cannot see.

ponytail: the isolated profiles are green-context runs while the pairs are MPS static; the check
against MobileNet alone under MPS static (soc-l2) bounds the difference for one network.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import aggregate as A  # noqa: E402
import profiles as P   # noqa: E402

AGG = os.path.join(P.BENCH, "aggregate")
# engine name of each network and its kernels per inference (the period of its kernel sequence)
ENG = {"mobilenet": ("mobilenetv2-12", 55), "resnet": ("resnet50-v2-7", 76),
       "inception": ("inception_v3", 101), "retinanet": ("retinanet-9", 172)}
NETS = list(ENG)
PRESENCE_US = 0.87
SATURATION_GB_S = 135.0      # the highest GPU DRAM rate measured: SAXPY back to back on 8 SMs
VARIANTS = {"dram": lambda t, c: [c["D"], t["D"] * c["D"]],
            "l2": lambda t, c: [c["L"], t["L"] * c["L"]],
            "ref": lambda t, c: [c["D"], t["D"] * c["D"], c["L"], t["L"] * c["L"]]}


def ncu_l2(eng, kpi):
    """L2 MB per inference and byte-weighted L2 hit rate of one network from its ncu nets profile.

    ncu recorded 600 kernels whatever the network, so only whole inferences are kept: a partial one
    (RetinaNet has 3.5) would weight its layers wrongly."""
    k = {}
    path = os.path.join(P.PROF, "ncu", "nets", "nets-%s.csv" % eng)
    for f in csv.reader(l for l in open(path, errors="replace") if l.startswith('"')):
        if len(f) >= 15 and f[12] != "Metric Name":
            try:
                k.setdefault(int(f[0]), {})[f[12]] = float(f[14].replace(",", ""))
            except ValueError:
                pass
    n = len(k) // kpi
    ks = [k[i] for i in sorted(k)][:n * kpi]
    lb = sum(x["lts__t_bytes.sum"] for x in ks)
    hit = sum(x["lts__t_sector_hit_rate.pct"] * x["lts__t_bytes.sum"] for x in ks) / lb / 100
    return lb / n / 1e6, hit, n


def profile(data):
    """Each network alone on 8 SMs: D, L, h, p, plus the MB per inference behind them."""
    dram = {r["arm"]: r for r in csv.DictReader(open(os.path.join(P.DERIVED, "dram.csv")))}
    out = {}
    for net, (eng, kpi) in ENG.items():
        r = dram["soc-greenctx-%s-50" % eng]
        l2_mb, hit, n = ncu_l2(eng, kpi)
        alone_ms = A.aggr(data, "exp23", net + "-alone", net, "gpu_ms", "median")[0]
        out[net] = dict(D=float(r["gb_s_run_average"]), dram_mb=float(r["mb_per_inference"]),
                        L=l2_mb * float(r["inf_per_s"]) / 1e3, l2_mb=l2_mb, h=hit, ncu_inf=n,
                        p=kpi * PRESENCE_US * 1e-3 / alone_ms)
    return out


def aggregate(profs):
    """One co-runner load out of several: rates add, a hit rate is rebuilt from the misses."""
    L = sum(x["L"] for x in profs)
    return dict(D=sum(x["D"] for x in profs), L=L,
                h=1 - sum(x["L"] * (1 - x["h"]) for x in profs) / L)


def measured(data):
    """The 12 pairs: (target, co-runner, p99 slowdown, median slowdown) against exp23 alone."""
    rows = []
    for t in NETS:
        a99 = A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "p99")[0]
        a50 = A.aggr(data, "exp23", t + "-alone", t, "gpu_ms", "median")[0]
        for c in NETS:
            if c != t:
                cfg = t + "+" + c
                rows.append((t, c, A.aggr(data, "exp24", cfg, t, "gpu_ms", "p99")[0] / a99,
                             A.aggr(data, "exp24", cfg, t, "gpu_ms", "median")[0] / a50))
    return rows


def fit(rows, prof, X, q):
    """Least squares of s - p_t on the variant's features; q is the column (2 p99, 3 median)."""
    A_ = np.array([X(prof[r[0]], aggregate([prof[r[1]]])) for r in rows])
    y = np.array([r[q] - 1 - prof[r[0]]["p"] for r in rows])
    return np.linalg.lstsq(A_, y, rcond=None)[0]


def predict(t, c, prof, X, b):
    return 1 + prof[t]["p"] + float(np.dot(X(prof[t], aggregate([prof[c]])), b))


def loo(rows, prof, X, q):
    """Leave one network out: each pair predicted by a fit that never saw its target."""
    pred = {}
    for k in NETS:
        b = fit([r for r in rows if k not in r[:2]], prof, X, q)
        for r in rows:
            if r[0] == k:
                pred[r[:2]] = predict(r[0], r[1], prof, X, b)
    return pred


def checks(prof, rows):
    print("\nchecks")
    print("  saturation: target + co-runner D, largest pair %.1f GB/s against %.0f measured"
          % (max(prof[t]["D"] + prof[c]["D"] for t, c, *_ in rows), SATURATION_GB_S))
    for r in csv.DictReader(open(os.path.join(P.DERIVED, "pair-traffic.csv"))):
        print("  two networks together (green context) %-40s measured / sum of alone %s"
              % (r["arm"].replace("soc-greenctx-", ""), r["ratio"]))
    # MobileNet alone under MPS static (soc-l2, 50 % partition) against the green-context profile.
    # The GB/s scale is profiles.py's calibration, taken from its derived file.
    cal = next(csv.DictReader(open(os.path.join(P.DERIVED, "calibration.csv"))))
    P.COPY_GB_S, P.COPY_READ_PCT, P.COPY_WRITE_PCT = (float(cal[k]) for k in ("copy_gb_s", "read_pct", "write_pct"))
    stem = os.path.join(P.PROF, "soc-l2", "mobilenet-alone")
    nets, _ = P.report(stem + ".log")
    n, rate = nets["mobilenet"]
    mb = P.dram_gb(stem)[0] * 1e3 / (n + P.WARMUP)
    m = prof["mobilenet"]
    print("  MobileNet alone, MPS static: %.2f MB/inf, %.1f GB/s, %.0f inf/s; green context: %.2f MB/inf, %.1f GB/s"
          % (mb, mb * rate / 1e3, rate, m["dram_mb"], m["D"]))


def write(name, rows):
    path = os.path.join(AGG, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path)


def main():
    data = A.load(P.BENCH)
    prof = profile(data)
    print("each network alone on 8 SMs")
    print("  %-10s %7s %8s %8s %6s %8s %6s %4s %12s" % ("network", "D GB/s", "DRAM MB", "L2 MB", "hit %",
                                                       "L GB/s", "p %", "inf", "L2*miss MB"))
    for k, x in prof.items():
        print("  %-10s %7.1f %8.1f %8.1f %6.1f %8.1f %6.2f %4d %12.1f" % (k, x["D"], x["dram_mb"], x["l2_mb"],
              100 * x["h"], x["L"], 100 * x["p"], x["ncu_inf"], x["l2_mb"] * (1 - x["h"])))
    rows = measured(data)
    pred = {(v, q): loo(rows, prof, X, qi) for v, X in VARIANTS.items() for q, qi in (("p99", 2), ("median", 3))}

    for q, qi in (("p99", 2), ("median", 3)):
        print("\n%s slowdown: measured, leave-one-network-out prediction (error, points)" % q)
        print("  %-10s %-10s %6s" % ("target", "co-runner", "meas") + "".join("%16s" % v for v in VARIANTS))
        err = {v: [] for v in VARIANTS}
        for r in rows:
            line = "  %-10s %-10s %6.3f" % (r[0], r[1], r[qi])
            for v in VARIANTS:
                p = pred[(v, q)][r[:2]]
                err[v].append((p - r[qi]) * 100)
                line += "   %6.3f (%+5.1f)" % (p, err[v][-1])
            print(line)
        for name, f in (("mean |error|", lambda e: np.mean(np.abs(e))), ("max |error|", lambda e: np.max(np.abs(e))),
                        ("bias", np.mean), ("under-predicted", lambda e: np.sum(e < 0)),
                        ("largest under", np.min)):
            print("  %-27s" % name + "".join("%16.1f" % f(np.array(err[v])) for v in VARIANTS))

    coeffs = []
    for v, X in VARIANTS.items():
        write("stress-comparison-counters-%s.csv" % v, [dict(
            target=r[0], corunner=r[1], measured_p99=round(r[2], 4), measured_median=round(r[3], 4),
            # the column names of stress.py compare, so models.py reads every model alike
            pred_p99_p99=round(pred[(v, "p99")][r[:2]], 4),
            err_p99_p99_pts=round((pred[(v, "p99")][r[:2]] - r[2]) * 100, 4),
            pred_median_p99=round(pred[(v, "median")][r[:2]], 4),
            err_median_p99_pts=round((pred[(v, "median")][r[:2]] - r[3]) * 100, 4)) for r in rows])
        for q, qi in (("p99", 2), ("median", 3)):
            b = list(fit(rows, prof, X, qi)) + [None] * 2
            coeffs.append(dict(variant=v, statistic=q,
                               **{"b%d" % i: "" if x is None else float("%.6g" % x) for i, x in enumerate(b[:4])}))
    write("counter-model-coeffs.csv", coeffs)
    checks(prof, rows)


def selftest():
    prof = {n: dict(D=10.0 + 7 * i, L=40.0 + 30 * i * i, h=0.7, p=0.01 * (i + 1)) for i, n in enumerate(NETS)}
    b = np.array([0.003, -2e-5, 0.001, -3e-6])
    rows = [(t, c, 1 + prof[t]["p"] + float(np.dot(VARIANTS["ref"](prof[t], prof[c]), b)), 0.0)
            for t in NETS for c in NETS if t != c]
    for k, v in loo(rows, prof, VARIANTS["ref"], 2).items():
        assert abs(v - next(r[2] for r in rows if r[:2] == k)) < 1e-9, k
    g = aggregate([dict(D=10, L=100, h=0.9), dict(D=30, L=50, h=0.4)])
    assert (g["D"], g["L"]) == (40, 150) and abs(g["h"] - (1 - 40 / 150)) < 1e-12   # misses 10 + 30
    print("  selftest passed")


if __name__ == "__main__":
    {"selftest": selftest}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
