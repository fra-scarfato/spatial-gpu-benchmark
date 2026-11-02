#!/usr/bin/env python3
"""geometry.py - the L2 counter model, frozen on the 8 + 8 pairs, on other splits and with two co-runners.

    python3 analysis/geometry.py inputs          each network's inputs at every partition size
    python3 analysis/geometry.py splits          4+12, 6+10, 12+4: the frozen model against the pairs
    python3 analysis/geometry.py triple-predict  8+4+4 and 8+4: freeze the predictions (no pair read)
    python3 analysis/geometry.py triple-compare  score the frozen predictions on the measured runs
    python3 analysis/geometry.py triple-current  the same scoring, with the current model's predictions
    python3 analysis/geometry.py power           power, completed inferences, energy per inference
    python3 analysis/geometry.py selftest

The model is counter-model.py's L2 variant with its frozen p99 coefficients (counter-model-coeffs.csv):

    s - p_t = (b0 + b1 L_t) L_c,    s = p99 slowdown - 1

Nothing is refitted. What changes with the geometry are the inputs, each read at the partition size the
network actually has:
  L  L2 GB/s = ncu L2 MB per inference (a kernel property, MEASUREMENTS.md) x inferences per second at
     that size, from two sources, both reported:
       scaled  the frozen 8-SM L times throughput(n) / throughput(8), both from the sweep's alone rows of
               this campaign (stress-split-*, stress-sizes), so nsys's 1-4 % slowdown cancels;
       sudo    profile.sh soc-mps-sizes: the inference rate under nsys at that size, as soc-mps at 8.
  D  DRAM GB/s at that size, by the same two routes; used only to place a co-runner in the SAXPY table.
  p  presence: kernels per inference x counter-model's PRESENCE_US / the target's median latency alone
     at its size.

Two co-runners (8 + 4 + 4): the load is L_a + L_b (counter-model's aggregate), and three versions are
frozen side by side (the user, 2026-10-28):
  sum      the plain sum
  rho      the sum times rho(D_a, D_b), the two-SAXPY composition ratio (compose.py, p99 coordinate),
           read off the 4x4 table by each co-runner's DRAM GB/s alone on 4 SMs, bilinear, clamped to the
           table's ends; the table is symmetrised, since the partition order was within noise
  rho_max1 the sum times max(1, rho): only the amplification at high load
and the presence of two co-runners is g p_t, with g the growth from one spin to two on MobileNet
(exp25 against exp19: +13.7 % against +8.6 % at the p99, g = 1.60), recomputed here from the data.

Every tenant is a victim (the user, 2026-10-28: admission checks each admitted network's p99, not only
the newcomer's). The target is measured as always; each runner co-runner's per-inference rows are kept
by sweep.sh as r<k>_co<i>.csv, cut to the target's timed window, and its p99 is set against its own
network's alone row at its size. Its prediction is the same model with the roles swapped: its own p
and L at its size, the others' loads at theirs. So 6+10 also tests a victim on 10 SMs, 8+4 a victim on
4 next to 8, and 8+4+4 two victims on 4 next to 8 + 4. All of them are frozen with the targets'.

Sacrifices, chosen knowingly:
- rho was measured on SAXPY streams and is applied to networks by their DRAM GB/s alone. Networks on
  4 SMs sit at the light end of the table, where rho is within the noise of its zero. This is why the
  three versions are frozen together instead of one being picked before the triples are seen.
- g is MobileNet's; the spin rows of stress-triple check it per target, they do not feed it.
- A co-runner victim in 8+4+4 has its two co-runners on 8 and 4 SMs, while rho was measured with two
  on 4. It is read by their DRAM GB/s all the same (clamped); its error is reported apart.
- A co-runner's p99 comes from fewer rows than the target's when it is the slower network (it runs only
  inside the target's ~30 s window); runs with fewer than 100 rows there are dropped.
- The power is the three INA3221 rails of the module (GPU+SoC, CPU, 5 V system input) over the
  target's timed window, not the wall plug.
"""
import csv
import glob
import hashlib
import importlib
import os
import re
import statistics as st
import sys

import numpy as np

CMD = sys.argv[1] if len(sys.argv) > 1 else ""
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
C = importlib.import_module("counter-model")
A, P = C.A, C.P

DEV = C.NETS
NEW = ("efficientnet", "vit")
NETS = list(DEV) + list(NEW)
ENG = dict(C.ENG, **{n: C.ENG_NEW[n] for n in NEW})
SHARE = {4: 25, 6: 37, 8: 50, 10: 63, 12: 75}
SPLITS = {"exp28": ("412", 4, 12), "exp29": ("610", 6, 10), "exp30": ("124", 12, 4)}
# Where each size's alone row lives: (family, configuration suffix).
ALONE = {4: ("exp28", "-alone"), 6: ("exp29", "-alone"), 12: ("exp30", "-alone"),
         8: ("exp31", "-alone08"), 10: ("exp31", "-alone10")}
TRIPLE = "exp32"
FAMILY_DIR = {e: f for f, e in A.EXP_DIR.items() if e in ("exp28", "exp29", "exp30", "exp31", "exp32")}
FROZEN = os.path.join(C.AGG, "geometry-triple-predictions.csv")


# ------------------------------------------------------------------ inputs

def throughput(data, exp, cfg, net):
    """Median over runs of the timed inferences per second of one configuration, or None."""
    v = [data[k][net]["gpu_ms"]["count"] / data[k][net]["gpu_ms"]["duration_s"]
         for k in sorted(data) if k[0] == exp and k[1] == cfg and net in data[k]]
    return st.median(v) if v else None


def sudo_rows():
    """soc-mps-sizes and soc-mps rows of derived/dram.csv by (engine, share), if the root step ran."""
    out = {}
    for r in csv.DictReader(open(os.path.join(P.DERIVED, "dram.csv"))):
        m = re.match(r"soc-mps-(.+)-(\d+)$", r["arm"])
        if m:
            out[(m.group(1), int(m.group(2)))] = r
    return out


def inputs(data):
    """{net: {size: dict(thr, alone_ms, p, L_scaled, D_scaled, L_sudo, D_sudo)}} for every size whose
    alone row exists. L and D at 8 SMs are the frozen profile's, so scaled equals frozen there."""
    prof = dict(C.profile(data), **C.profile(data, {n: ENG[n] for n in NEW}, "exp26"))
    sudo = sudo_rows()
    out = {}
    for net in NETS:
        eng, kpi = ENG[net]
        f8, s8 = ALONE[8]
        thr8 = throughput(data, f8, net + s8, net)
        out[net] = {}
        for n, (exp, suf) in sorted(ALONE.items()):
            thr = throughput(data, exp, net + suf, net)
            if thr is None or thr8 is None:
                continue
            ms = A.aggr(data, exp, net + suf, net, "gpu_ms", "median")[0]
            x = dict(thr=thr, alone_ms=ms, p=kpi * C.PRESENCE_US * 1e-3 / ms,
                     L_scaled=prof[net]["L"] * thr / thr8, D_scaled=prof[net]["D"] * thr / thr8,
                     L_sudo=None, D_sudo=None)
            r = sudo.get((eng, SHARE[n]))
            if r:
                x["L_sudo"] = prof[net]["l2_mb"] * float(r["inf_per_s"]) / 1e3
                x["D_sudo"] = float(r["gb_s_run_average"])
            out[net][n] = x
    return out


def coefficients():
    """The frozen L2 p99 coefficients (b0, b1)."""
    for r in csv.DictReader(open(os.path.join(C.AGG, "counter-model-coeffs.csv"))):
        if (r["variant"], r["statistic"]) == ("l2", "p99"):
            return float(r["b0"]), float(r["b1"])
    raise SystemExit("geometry.py: no l2 p99 row in counter-model-coeffs.csv")


def predict(b, p_t, L_t, L_c, g=1.0, f=1.0):
    """p99 slowdown of a target with presence p_t and load L_t beside a co-runner load L_c; g scales
    the presence (two co-runners), f the load (the composition factor)."""
    return 1 + g * p_t + (b[0] + b[1] * L_t) * f * L_c


# ------------------------------------------------------------------ composition

def rho_table():
    """The two-SAXPY table: the axis (each duty's GB/s alone on 4 SMs, from compose.py's runs) and
    the symmetrised p99 ratio, as (x[4], R[4][4])."""
    sys.argv = sys.argv[:1]            # compose imports stress.py, which reads its geometry from argv[2]
    K = importlib.import_module("compose")
    x = [K.config("s4-D" + d)["gbs0"] for d in K.DUTIES]
    r = {(row["a"], row["b"]): float(row["p99_ratio"])
         for row in csv.DictReader(open(os.path.join(C.AGG, "compose-4sm.csv")))}
    R = [[(r[(a, b)] + r[(b, a)]) / 2 for b in K.DUTIES] for a in K.DUTIES]
    return x, R


def rho(x, R, da, db):
    """Bilinear interpolation of R at (da, db) on the axis x, clamped to the table's ends."""
    def locate(v):
        v = min(max(v, x[0]), x[-1])
        i = min(int(np.searchsorted(x, v, side="right")) - 1, len(x) - 2)
        return i, (v - x[i]) / (x[i + 1] - x[i])
    (i, u), (j, w) = locate(da), locate(db)
    return ((1 - u) * (1 - w) * R[i][j] + u * (1 - w) * R[i + 1][j] +
            (1 - u) * w * R[i][j + 1] + u * w * R[i + 1][j + 1])


def dispatch_growth(data):
    """g: MobileNet's p99 presence with two spins on 4 + 4 over one spin on 4 (exp25, exp19)."""
    a = A.aggr(data, "exp19", "mobilenet-alone", "mobilenet", "gpu_ms", "p99")[0]
    one = A.aggr(data, "exp19", "mobilenet-spin-K100", "mobilenet", "gpu_ms", "p99")[0] / a - 1
    two = A.aggr(data, "exp25", "mobilenet-spin4x2", "mobilenet", "gpu_ms", "p99")[0] / a - 1
    return two / one


# ------------------------------------------------------------------ scoring

def slowdown(data, exp, cfg, net, alone_cfg):
    """Measured p99 slowdown of a configuration against an alone row of the same family, or None."""
    m = A.aggr(data, exp, cfg, net, "gpu_ms", "p99")
    a = A.aggr(data, exp, alone_cfg, net, "gpu_ms", "p99")
    return m[0] / a[0] if m and a else None


def alone_p99(data, net, n):
    """The p99 of a network alone on n SMs (median over runs), from this campaign's alone rows."""
    exp, suf = ALONE[n]
    a = A.aggr(data, exp, net + suf, net, "gpu_ms", "p99")
    return a[0] if a else None


def corunner_p99(d, k):
    """Median over the runs in folder d of the k-th co-runner's p99, over the target's timed window only.

    The co-runner starts 3 s before the target and is stopped after it, so its rows outside the window
    were taken next to a target that was loading or gone. Runs without the target's stats file (not
    done) or with fewer than 100 rows in the window are skipped. np.percentile's linear rule is the
    runner's own (stats.cpp)."""
    v = []
    for lat in sorted(glob.glob(os.path.join(d, "r*_latencies.csv"))):
        run = os.path.basename(lat)[:-len("_latencies.csv")]
        co = os.path.join(d, "%s_co%d.csv" % (run, k))
        if not (os.path.exists(co) and os.path.exists(os.path.join(d, run + "_stats.csv"))):
            continue
        lo, hi = timed_window(lat)
        x = [float(r["gpu_ms"]) for r in csv.DictReader(open(co))
             if r["phase"] == "bench" and lo <= int(r["t_start_ns"]) / 1e9 and int(r["t_end_ns"]) / 1e9 <= hi]
        if len(x) >= 100:
            v.append(float(np.percentile(x, 99)))
    return st.median(v) if v else None


def victims(cfg, n_target, n_co):
    """The tenants of a configuration "t+a[+b]" as (victim, its SMs, position, [(other, its SMs)])."""
    t, cos = cfg.split("+")[0], cfg.split("+")[1:]
    out = [(t, n_target, "target", [(c, n_co) for c in cos])]
    for k, c in enumerate(cos, 1):
        out.append((c, n_co, "co%d" % k, [(t, n_target)] + [(o, n_co) for o in cos if o != c]))
    return out


def measured(data, exp, cfg, victim, n, pos):
    """p99 slowdown of one tenant: the target's from its stats, a co-runner's from its kept rows."""
    if pos == "target":
        return slowdown(data, exp, cfg, victim, victim + "-alone")
    m, a = corunner_p99(os.path.join(P.BENCH, FAMILY_DIR[exp], cfg), int(pos[2:])), alone_p99(data, victim, n)
    return m / a if m and a else None


def victim_preds(b, inp, victim, n, others, rho_xr=None, g=1.0):
    """The frozen model's p99 slowdown of one victim on n SMs beside others [(net, SMs)], per input
    source. One other: {"scaled": s, "sudo": s}. Two others: the three composition versions
    {"sum_scaled": ..., "rho_scaled": ..., "rho_max1_scaled": ...}, plus the rho used ("rhoval_*")."""
    out = {}
    for src in ("scaled", "sudo"):
        Lv = inp[victim][n]["L_" + src]
        Ls = [inp[o][m]["L_" + src] for o, m in others]
        Ds = [inp[o][m]["D_" + src] for o, m in others]
        if Lv is None or None in Ls:
            continue
        p = inp[victim][n]["p"]
        if len(others) == 1:
            out[src] = predict(b, p, Lv, Ls[0])
            continue
        r = rho(rho_xr[0], rho_xr[1], Ds[0], Ds[1])
        out["rhoval_" + src] = r
        for v, f in (("sum", 1.0), ("rho", r), ("rho_max1", max(1.0, r))):
            out["%s_%s" % (v, src)] = predict(b, p, Lv, sum(Ls), g, f)
    return out


def summary(label, errs):
    e = np.array(errs)
    if not len(e):
        return "  %-44s (no rows)" % label
    return "  %-44s n %2d  mean |e| %4.1f  max |e| %4.1f  under %2d  worst %+5.1f  bias %+5.1f" % (
        label, len(e), np.abs(e).mean(), np.abs(e).max(), (e < 0).sum(), min(e.min(), 0), e.mean())


def block(*nets):
    return "development" if all(n in DEV for n in nets) else "with EfficientNet/ViT"


def write(name, rows):
    """rows to bench/aggregate/<name>; the columns are the union over rows (spin and pair rows, 8+4 and
    8+4+4 rows carry different predictions), in first-seen order, empty where a row has none."""
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(C.AGG, name)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)
    print("  wrote", path)


# ------------------------------------------------------------------ commands

def cmd_inputs():
    data = A.load(P.BENCH)
    inp = inputs(data)
    print("each network alone: inf/s, median ms, p %, L GB/s scaled / sudo, D GB/s scaled / sudo")
    rows = []
    for net in NETS:
        for n, x in inp[net].items():
            f = lambda v: "%7.1f" % v if v is not None else "      -"
            print("  %-12s %2d SMs %8.1f %7.3f %5.2f   L %s %s   D %s %s" % (
                net, n, x["thr"], x["alone_ms"], 100 * x["p"], f(x["L_scaled"]), f(x["L_sudo"]),
                f(x["D_scaled"]), f(x["D_sudo"])))
            rows.append(dict(net=net, sms=n, **{k: (round(v, 4) if v is not None else "") for k, v in x.items()}))
    write("geometry-inputs.csv", rows)


def cmd_splits():
    """Every tenant of every split pair against the frozen model; the spin rows against p_t."""
    data = A.load(P.BENCH)
    inp, b = inputs(data), coefficients()
    rows, err = [], {}
    for exp, (name, nt, nc) in SPLITS.items():
        print("\nsplit %d + %d: p99 slowdown of each tenant, measured and frozen model (error, points)" % (nt, nc))
        for t in NETS:
            if nt not in inp[t]:
                continue
            s = slowdown(data, exp, t + "-spin", t, t + "-alone")
            if s:
                e = (1 + inp[t][nt]["p"] - s) * 100
                err.setdefault(("dispatch", name, "victim on %d SMs" % nt), []).append(e)
                print("  %-12s %2d SMs  beside spin       meas %6.3f  p_t %6.3f (%+5.1f)" % (
                    t, nt, s, 1 + inp[t][nt]["p"], e))
                rows.append(dict(split=name, config=t + "-spin", victim=t, victim_sms=nt, position="target",
                                 measured=round(s, 4), pred_scaled=round(1 + inp[t][nt]["p"], 4), err_scaled=round(e, 2)))
            for c in NETS:
                if c == t:
                    continue
                cfg = t + "+" + c
                for v, n, pos, others in victims(cfg, nt, nc):
                    if n not in inp[v] or any(m not in inp[o] for o, m in others):
                        continue
                    s = measured(data, exp, cfg, v, n, pos)
                    if not s:
                        continue
                    pr = victim_preds(b, inp, v, n, others)
                    row = dict(split=name, config=cfg, victim=v, victim_sms=n, position=pos, measured=round(s, 4))
                    line = "  %-12s %2d SMs  beside %-10s meas %6.3f" % (v, n, others[0][0], s)
                    for src, val in pr.items():
                        e = (val - s) * 100
                        row["pred_" + src], row["err_" + src] = round(val, 4), round(e, 2)
                        err.setdefault((src, name, "victim on %d SMs" % n, block(t, c)), []).append(e)
                        line += "  %s %6.3f (%+5.1f)" % (src, val, e)
                    print(line)
                    rows.append(row)
    print("\nsummary (points; dispatch = spin row against p_t)")
    for k in sorted(err):
        print(summary(" / ".join(k), err[k]))
    write("geometry-splits.csv", rows)


def planned_triples():
    """The 8+4 pairs and 8+4+4 triples stress-triple will run (lib/matrix.sh), so the predictions can
    be frozen before any of them exists."""
    out = [t + "+" + c for t in NETS for c in NETS if c != t]
    for t in DEV:
        o = [c for c in DEV if c != t]
        out += ["%s+%s+%s" % (t, o[0], o[1]), "%s+%s+%s" % (t, o[0], o[2]),
                "%s+%s+%s" % (t, o[1], o[2]), "%s+efficientnet+vit" % t]
    for c in DEV:
        out += ["efficientnet+vit+" + c, "vit+efficientnet+" + c]
    return out


def triple_rows(data):
    """Every tenant of every planned 8+4 pair and 8+4+4 triple, with its frozen predictions; reads no
    pair. The target is on 8 SMs, each co-runner on 4."""
    inp, b = inputs(data), coefficients()
    xr, g = rho_table(), dispatch_growth(data)
    rows = []
    for cfg in planned_triples():
        for v, n, pos, others in victims(cfg, 8, 4):
            if n not in inp[v] or any(m not in inp[o] for o, m in others):
                print("  WARNING: %s, victim %s: an alone row is missing, no prediction frozen" % (cfg, v))
                continue
            row = dict(config=cfg, victim=v, victim_sms=n, position=pos, g=round(g, 4))
            row.update({("pred_" + k if not k.startswith("rhoval_") else k): round(val, 4)
                        for k, val in victim_preds(b, inp, v, n, others, xr, g).items()})
            rows.append(row)
    return rows


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def cmd_triple_predict():
    data = A.load(P.BENCH)
    if any(k[0] == TRIPLE and "+" in k[1] for k in data):
        raise SystemExit("geometry.py: stress-triple pairs exist already; the predictions must be frozen before them")
    rows = triple_rows(data)
    head = ["config", "victim", "victim_sms", "position", "g"]
    keys = head + sorted({k for r in rows for k in r} - set(head))
    with open(FROZEN, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    open(FROZEN + ".sha256", "w").write("%s  %s\n" % (sha256(FROZEN), os.path.basename(FROZEN)))
    print("  frozen %d rows: %s, sha256 %s" % (len(rows), FROZEN, sha256(FROZEN)))


def cmd_triple_compare(current=False):
    """Score 8+4 and 8+4+4. By default the frozen predictions (the test as it was run); with current,
    the predictions of the model as it is now, e.g. after the presence constant changed (2026-10-29):
    a re-scoring after the fact, written to its own file so the frozen record stays."""
    data = A.load(P.BENCH)
    if current:
        preds = triple_rows(data)
        print("  current model's predictions (not frozen: presence %.2f us)" % C.PRESENCE_US)
    else:
        want = open(FROZEN + ".sha256").read().split()[0]
        if sha256(FROZEN) != want:
            raise SystemExit("geometry.py: %s changed after it was frozen" % FROZEN)
        print("  frozen predictions intact (sha256 %s...)" % want[:12])
        preds = list(csv.DictReader(open(FROZEN)))
    err, out = {}, []
    print("\n8 + 4 and 8 + 4 + 4: p99 slowdown of each tenant, measured and frozen predictions (error, points)")
    for p in preds:
        cfg, v, n, pos = p["config"], p["victim"], int(p["victim_sms"]), p["position"]
        s = measured(data, TRIPLE, cfg, v, n, pos)
        if not s:
            continue
        geo = "8+4" if cfg.count("+") == 1 else "8+4+4"
        line = "  %-5s %-28s %-12s %2d SMs  meas %6.3f" % (geo, cfg, v, n, s)
        row = dict(geometry=geo, config=cfg, victim=v, victim_sms=n, position=pos, measured=round(s, 4))
        for k, val in p.items():
            if k.startswith("pred_") and val:
                e = (float(val) - s) * 100
                row[k], row["err" + k[4:]] = float(val), round(e, 2)
                err.setdefault((geo, k[5:], "victim on %d SMs" % n, block(*cfg.split("+"))), []).append(e)
                line += "  %s %+5.1f" % (k[5:], e)
        print(line)
        out.append(row)
    print("\nsummary (points)")
    for k in sorted(err):
        print(summary(" / ".join(k), err[k]))

    print("\ndispatch: one spin on 4 SMs, two on 4 + 4, growth against g = %.2f" % dispatch_growth(data))
    inp = inputs(data)
    for t in NETS:
        one = slowdown(data, TRIPLE, t + "-spin4", t, t + "-alone")
        two = slowdown(data, TRIPLE, t + "-spin4x2", t, t + "-alone")
        if one and two:
            print("  %-12s p_t %+5.1f %%  one %+5.1f %%  two %+5.1f %%  growth %.2f" % (
                t, 100 * inp[t][8]["p"], 100 * (one - 1), 100 * (two - 1), (two - 1) / (one - 1)))

    print("\ncomposition on networks, target on 8 SMs: (s(t, a+b) - 1) / ((s(t, a) - 1) + (s(t, b) - 1))")
    for r in out:
        cos = r["config"].split("+")[1:]
        if r["position"] != "target" or len(cos) != 2:
            continue
        t = r["victim"]
        sa = slowdown(data, TRIPLE, t + "+" + cos[0], t, t + "-alone")
        sb = slowdown(data, TRIPLE, t + "+" + cos[1], t, t + "-alone")
        if sa and sb:
            print("  %-28s %.2f" % (r["config"], (r["measured"] - 1) / (sa + sb - 2)))
    write("geometry-triple-compare-current.csv" if current else "geometry-triple-compare.csv", out)


# ------------------------------------------------------------------ power

def power_log(path):
    """[(uptime_s, gpu_soc_mW, cpu_cv_mW, sys5v_mW)] from a sweep _power.log."""
    out = []
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) == 7 and f[0][0].isdigit():
            v = [float(x) for x in f]
            out.append((v[0], v[1] * v[2] / 1e3, v[3] * v[4] / 1e3, v[5] * v[6] / 1e3))
    return out


def timed_window(latencies):
    """(start, end) in seconds on CLOCK_MONOTONIC of the target's timed phase ("bench" rows)."""
    lo = hi = None
    for r in csv.DictReader(open(latencies)):
        if r["phase"] == "bench":
            s, e = int(r["t_start_ns"]) / 1e9, int(r["t_end_ns"]) / 1e9
            lo = s if lo is None else min(lo, s)
            hi = e if hi is None else max(hi, e)
    return lo, hi


def corunner_rate(log):
    """Inferences per second a stopped co-runner completed (the runner's last stderr line), or None."""
    m = re.findall(r"\[info\] (\d+) timed inferences in ([0-9.]+) s before the stop", open(log, errors="replace").read())
    return int(m[-1][0]) / float(m[-1][1]) if m else None


def cmd_power():
    rows = []
    for log in sorted(glob.glob(os.path.join(P.BENCH, "*", "*", "r*_power.log"))):
        d, run = os.path.dirname(log), os.path.basename(log)[:-len("_power.log")]
        lat = os.path.join(d, run + "_latencies.csv")
        stats = os.path.join(d, run + "_stats.csv")
        if not (os.path.exists(lat) and os.path.exists(stats)):
            continue
        lo, hi = timed_window(lat)
        smp = [s for s in power_log(log) if lo <= s[0] <= hi]
        if len(smp) < 5:
            continue
        g = next(r for r in csv.DictReader(open(stats)) if r["metric"] == "gpu_ms")
        tgt = int(g["count"]) / float(g["duration_s"])
        co = [corunner_rate(f) for f in sorted(glob.glob(os.path.join(d, run + "_itf*.log")))]
        mw = [st.mean(s[i] for s in smp) for i in (1, 2, 3)]
        total = tgt + sum(c for c in co if c)
        rows.append(dict(family=os.path.basename(os.path.dirname(d)), config=os.path.basename(d), run=run,
                         samples=len(smp), gpu_soc_mW=round(mw[0], 1), cpu_cv_mW=round(mw[1], 1),
                         sys5v_mW=round(mw[2], 1), rails_mW=round(sum(mw), 1),
                         target_inf_s=round(tgt, 2), corunner_inf_s=";".join("%.2f" % c if c else "-" for c in co),
                         all_inf_s=round(total, 2), mJ_per_inference=round(sum(mw) / total, 4),
                         corunner_missing=sum(c is None for c in co)))
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["config"]), []).append(r)
    print("median over runs: rails mW (GPU+SoC / CPU / 5 V / sum), inferences/s (target, all), mJ per inference")
    for (f, c), rs in sorted(by.items()):
        m = lambda k: st.median(r[k] for r in rs)
        print("  %-18s %-36s %6.0f %6.0f %6.0f %6.0f  %8.1f %8.1f  %7.3f%s" % (
            f, c, m("gpu_soc_mW"), m("cpu_cv_mW"), m("sys5v_mW"), m("rails_mW"), m("target_inf_s"),
            m("all_inf_s"), m("mJ_per_inference"), "  (co-runner count missing)" if m("corunner_missing") else ""))
    write("geometry-power.csv", rows)


# ------------------------------------------------------------------ selftest

def selftest():
    x = [10.0, 20.0, 40.0, 80.0]
    R = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.2, 1.2, 1.2], [1.0, 1.2, 1.4, 1.4], [1.0, 1.2, 1.4, 1.6]]
    assert abs(rho(x, R, 20, 20) - 1.2) < 1e-12 and abs(rho(x, R, 80, 80) - 1.6) < 1e-12   # grid points
    assert abs(rho(x, R, 15, 10) - 1.0) < 1e-12                                             # an edge
    assert abs(rho(x, R, 30, 30) - 1.25) < 1e-12   # bilinear mid-cell: corners 1.2, 1.2, 1.2, 1.4
    assert rho(x, R, 1, 1) == R[0][0] and rho(x, R, 500, 500) == R[3][3]                   # clamped
    b = (0.002, -1e-5)
    assert predict(b, 0.05, 100, 0) == 1.05                                                 # no load: presence
    assert abs(predict(b, 0.05, 100, 50, g=2, f=1.5) - (1 + 0.1 + (0.002 - 0.001) * 75)) < 1e-12
    assert len(planned_triples()) == 30 + 24 and len(set(planned_triples())) == 54
    assert victims("t+a+b", 8, 4) == [("t", 8, "target", [("a", 4), ("b", 4)]),
                                      ("a", 4, "co1", [("t", 8), ("b", 4)]),
                                      ("b", 4, "co2", [("t", 8), ("a", 4)])]
    assert victims("t+a", 6, 10) == [("t", 6, "target", [("a", 10)]), ("a", 10, "co1", [("t", 6)])]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r1_power.log")
        open(p, "w").write("uptime_s a b c d e f\n100.00 5000 1000 4000 500 5000 2000\n100.20 5000 1200 4000 500 5000 2000\n")
        assert power_log(p) == [(100.0, 5000.0, 2000.0, 10000.0), (100.2, 6000.0, 2000.0, 10000.0)]
        q = os.path.join(d, "r1_itf.log")
        open(q, "w").write("  ...2000 inferences, 3.8 s\n[info] 1500 timed inferences in 30.00 s before the stop\n")
        assert corunner_rate(q) == 50.0
        # the co-runner's p99 counts only rows inside the target's timed window [10 s, 20 s]
        hdr = "phase,index,image,prep_ms,wall_ms,gpu_ms,t_start_ns,t_end_ns\n"
        open(os.path.join(d, "r1_latencies.csv"), "w").write(
            hdr + "warmup,0,x,0,1,1,1000000000,2000000000\nbench,0,x,0,1,1,10000000000,10500000000\n"
            "bench,1,x,0,1,1,19500000000,20000000000\n")
        open(os.path.join(d, "r1_stats.csv"), "w").write("metric\n")
        rows = ["bench,%d,x,0,1,%g,%d,%d" % (i, 1.0 + i / 1000, 10_000_000_000 + i * 10_000_000,
                                             10_000_000_000 + i * 10_000_000 + 5_000_000) for i in range(500)]
        rows += ["bench,900,x,0,1,99,5000000000,5100000000", "bench,901,x,0,1,99,21000000000,21100000000"]
        open(os.path.join(d, "r1_co1.csv"), "w").write(hdr + "\n".join(rows) + "\n")
        inside = [1.0 + i / 1000 for i in range(500) if 10 + i * 0.01 + 0.005 <= 20]
        assert abs(corunner_p99(d, 1) - np.percentile(inside, 99)) < 1e-12
    print("  selftest passed")


if __name__ == "__main__":
    {"inputs": cmd_inputs, "splits": cmd_splits, "triple-predict": cmd_triple_predict,
     "triple-compare": cmd_triple_compare,
     "triple-current": lambda: cmd_triple_compare(True), "power": cmd_power, "selftest": selftest}.get(
        CMD, lambda: sys.exit(__doc__))()
