#!/usr/bin/env python3
"""catalog.py - what an admission controller knows about the networks, exported for the simulator.

    python3 analysis/catalog.py [outdir]    write the four catalog-*.csv (default bench/aggregate)
    python3 analysis/catalog.py selftest

The admission-controller simulator lives in its own repository and reads only these files. The line
between measuring and fitting (this repository) and deciding (the simulator) is four CSVs, whose sha256
the simulator records with this repository's commit. Nothing is fitted here: every number is a model's
input or output, or a measurement, already in this repository, read through the functions that made
the thesis tables.

  catalog-networks.csv   one row per network and partition size (2, 4, ..., 16 SMs):
                           p99_ms   its p99 alone at that size, the base every predicted slowdown
                                    multiplies
                           p, L     presence and L2 GB/s, the normalized L2 model's inputs (empty on 16:
                                    a network on 16 SMs has no co-runner)
                           rate, W  inferences per second and power (the three rails) alone, the energy
                                    model's inputs
  catalog-pairs88.csv    one row per ordered pair on 8 + 8: the raw p99 slowdowns of Smooth, Temporal
                         and L2, the members of the conservative ensemble
  catalog-constants.csv  the L2 coefficients b0 b1, the dispatch growth g, the ensemble's frozen
                         corrections, the constant baseline's slowdown, the energy model's B and u
  catalog-measured.csv   every tenant of every measured configuration with its p99 (ms): the ground
                         truth the simulator scores its decisions against

Sources. Per size: the geometry campaign's alone rows at 4-12 SMs (geometry.ALONE), the final check's
at 2 and 14 (geometry.CHECK_ALONE), energy-alone-16 on 16. L is scaled within its own session
(geometry.inputs), so the 2- and 14-SM rows are relative to the check's own 8 SMs. Power and rate come
from the power rows of the same families (geometry.ENERGY_ALONE, plus the check's at 2 and 14).

The 8 + 8 members are the thesis's, frozen before the pairs were measured: Smooth and Temporal from
the development comparisons for pairs of the four development networks and from the validation ones
otherwise (as ensemble.py does); L2 from the frozen coefficients on each network's 8-SM profile
(counter-model.profile: exp23 for the four, exp26 for EfficientNet and the ViT). The corrections are
ensemble.frozen_shift, learnt on the 12 development pairs.

Measured configurations. A target's p99 comes straight from its stats. A co-runner's is its measured
slowdown in the geometry-*.csv files (cut to the target's window) times the alone p99 it was set
against, which gives back the p99 it was computed from. A configuration measured more than once (4+12
from either side; an 8 + 8 pair, whose two tenants are the targets of two runs) has a row per
measurement, and the simulator decides how to combine them.

ponytail: FP16 networks only, since INT8 was profiled at 8 SMs only. energy-tenants is left out of the
measured set: its p99 needs the raw rows of the common window (geometry.p99_together), and the
simulator's first version does without it.
"""
import csv
import importlib
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G   # noqa: E402
E = importlib.import_module("ensemble")
C, A, P = G.C, G.A, G.P

SIZES = (2, 4, 6, 8, 10, 12, 14, 16)
CHECK_SIZES = (2, 14)                   # the only sizes taken from the final check's session
POWER_ALONE = G.ENERGY_ALONE | {n: ("stress-sizes-check", "-alone%02d" % n) for n in CHECK_SIZES}
# The measured families in the geometry-*.csv files: file, the column naming the split, and per split
# value its family, the target's SMs, each co-runner's SMs, and the alone rows a co-runner was set against.
MEASURED = [("geometry-splits.csv", "split",
             {name: (exp, nt, nc, G.ALONE) for exp, (name, nt, nc) in G.SPLITS.items()}),
            ("geometry-check.csv", "split",
             {name: (exp, nt, nc, G.CHECK_ALONE) for exp, (name, nt, nc) in G.CHECK.items()}),
            ("geometry-triple-compare-current.csv", "geometry",
             {"8+4": (G.TRIPLE, 8, 4, G.ALONE), "8+4+4": (G.TRIPLE, 8, 4, G.ALONE)})]


def key(tenants):
    """The name of a configuration, whatever the order of its tenants: [(net, SMs)] -> "a@4+b@12"."""
    return "+".join(sorted("%s@%d" % t for t in tenants))


def networks(data, by):
    med = lambda rs, k: st.median(float(r[k]) for r in rs)
    main, check = G.inputs(data), G.inputs(data, G.CHECK_ALONE)
    rows = []
    for net in G.NETS:
        for n in SIZES:
            if n == 16:
                p99, x = A.aggr(data, "exp39", net + "-alone16", net, "gpu_ms", "p99")[0], {}
            elif n in CHECK_SIZES:
                p99, x = G.alone_p99(data, net, n, G.CHECK_ALONE), check[net][n]
            else:
                p99, x = G.alone_p99(data, net, n), main[net][n]
            pr = by[(POWER_ALONE[n][0], net + POWER_ALONE[n][1])]
            rows.append(dict(net=net, sms=n, p99_ms=p99, p=x.get("p", ""), L=x.get("L_scaled", ""),
                             rate=med(pr, "target_inf_s"), W=med(pr, "rails_mW") / 1e3))
    return rows


def pairs88(data):
    prof = dict(C.profile(data), **C.profile(data, {n: G.ENG[n] for n in G.NEW}, "exp26"))
    b = G.coefficients()
    dev = {m: E.stored(name, "p99") for m, name in E.STORED.items()}
    new = {m: E.stored(E.NEW_FILES[m], "p99") for m in E.STORED}
    rows = []
    for t in G.NETS:
        for c in G.NETS:
            if c != t:
                src = dev if t in G.DEV and c in G.DEV else new
                rows.append(dict(target=t, corunner=c, smooth=src["Smooth"][(t, c)],
                                 temporal=src["Temporal"][(t, c)], l2=C.predict(t, c, prof, C.VARIANTS["l2"], b)))
    return rows


def constants(data, by):
    pairs = C.measured(data)
    shift = E.frozen_shift(pairs, C.profile(data))
    b, m = G.coefficients(), G.energy_fit(by)
    return [dict(name=k, value=v) for k, v in (
        ("b0", b[0]), ("b1", b[1]), ("g", G.dispatch_growth(data)),
        ("shift_smooth", shift["Smooth"]), ("shift_temporal", shift["Temporal"]), ("shift_l2", shift["L2"]),
        ("constant", st.mean(r[2] for r in pairs)), ("B", m["B"]), ("u1", m["u"][1]), ("u2", m["u"][2]))]


def measured(data):
    rows = []
    for t in G.NETS:          # 8 + 8: each tenant is the target of its own run
        for c in G.NETS:
            fam = "exp24" if t in G.DEV and c in G.DEV else "exp27"
            v = A.aggr(data, fam, t + "+" + c, t, "gpu_ms", "p99") if c != t else None
            if v:
                rows.append(dict(family=fam, config=key([(t, 8), (c, 8)]), net=t, sms=8, p99_ms=v[0]))
    for f, col, splits in MEASURED:
        for r in csv.DictReader(open(os.path.join(C.AGG, f))):
            if "+" not in r["config"]:
                continue          # a spin row: no second network
            exp, nt, nc, alone = splits[r[col]]
            names = r["config"].split("+")
            v, n = r["victim"], int(r["victim_sms"])
            if r["position"] == "target":
                p99 = A.aggr(data, exp, r["config"], v, "gpu_ms", "p99")[0]
            else:
                p99 = float(r["measured"]) * G.alone_p99(data, v, n, alone)
            rows.append(dict(family=exp, config=key([(names[0], nt)] + [(c, nc) for c in names[1:]]),
                             net=v, sms=n, p99_ms=p99))
    return rows


def write(path, rows):
    """rows to a CSV, floats to 7 significant digits: more than any input is known to."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: "%.7g" % v if isinstance(v, float) else v for k, v in r.items()} for r in rows)
    print("  %s: %d rows" % (path, len(rows)))


def main(outdir):
    data, by = A.load(P.BENCH), G.power_rows()
    for name, rows in (("networks", networks(data, by)), ("pairs88", pairs88(data)),
                       ("constants", constants(data, by)), ("measured", measured(data))):
        write(os.path.join(outdir, "catalog-%s.csv" % name), rows)


def selftest():
    assert key([("b", 8), ("a", 12), ("a", 4)]) == "a@12+a@4+b@8"
    assert set(SIZES) - {16} <= set(POWER_ALONE) and set(POWER_ALONE) <= set(SIZES)
    print("  selftest passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else C.AGG)
