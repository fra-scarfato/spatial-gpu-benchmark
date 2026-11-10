#!/usr/bin/env python3
"""geometry.py - the L2 counter model, frozen on the 8 + 8 pairs, on other splits and with two co-runners.

    python3 analysis/geometry.py inputs          each network's inputs at every partition size
    python3 analysis/geometry.py splits          4+12, 6+10, 12+4: the frozen model against the pairs
    python3 analysis/geometry.py triple-predict  8+4+4 and 8+4: freeze the predictions (no pair read)
    python3 analysis/geometry.py triple-compare  score the frozen predictions on the measured runs
    python3 analysis/geometry.py triple-current  the same scoring, with the current model's predictions
    python3 analysis/geometry.py power           power, completed inferences, energy per inference
    python3 analysis/geometry.py check-predict   the final check: freeze the normalized model's predictions
    python3 analysis/geometry.py check-compare   score them on 2+14, 14+2 and 6+6
    python3 analysis/geometry.py energy          sec:stress-energy: alone by size, pairs at equal work,
                                                 and the direct check (energy-same-n)
    python3 analysis/geometry.py tenants [family [alone family]]
                                                 16 -> 8+8 -> 8+4+4 -> 4+4+4+4: time, energy and p99
                                                 at equal work (energy-tenants; the shared GPU:
                                                 energy-tenants-mpsdefault, energy-tenants-nomps)
    python3 analysis/geometry.py energy-model    the additive power model: errors on runs it never used,
                                                 and the allocation with the least energy it picks
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

The normalized model (the user, 2026-11-09) reads the target's sensitivity from its L2 GB/s per SM,
scaled to the 8 SMs on which the coefficients were fitted:

    s - p_t = (b0 + b1 (8/n) L_t) L_c,    n = the target's SMs

On 8 SMs it is the model above. The form was chosen on the 4+12, 6+10 and 12+4 runs, so the final check
tests it on splits never measured (2+14, 14+2, 6+6) in a new session: the inputs come only from that
session's alone rows (stress-sizes-check, at 2, 6, 8 and 14 SMs) and check-predict freezes the
predictions before any other row of the check exists. Generalization only: one co-runner, no f, no g.

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
# The final check: its splits, its own alone rows (measured first, in the same session) and its frozen file.
CHECK = {"exp36": ("214", 2, 14), "exp37": ("142", 14, 2), "exp38": ("66", 6, 6)}
CHECK_ALONE = {n: ("exp35", "-alone%02d" % n) for n in (2, 6, 8, 14)}
CHECK_FROZEN = os.path.join(A.FROZEN_DIR, "geometry-check-predictions.csv")
FAMILY_DIR = {e: f for f, e in A.EXP_DIR.items()
              if e in ("exp28", "exp29", "exp30", "exp31", "exp32", "exp35", "exp36", "exp37", "exp38")}
FROZEN = os.path.join(A.FROZEN_DIR, "geometry-triple-predictions.csv")


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


def inputs(data, alone=ALONE):
    """{net: {size: dict(thr, alone_ms, p, L_scaled, D_scaled, L_sudo, D_sudo)}} for every size whose
    alone row exists; alone maps a size to the (family, suffix) of its alone rows (the final check
    passes its own session's). L and D at 8 SMs are the frozen profile's, so scaled equals frozen there."""
    prof = dict(C.profile(data), **C.profile(data, {n: ENG[n] for n in NEW}, "exp26"))
    sudo = sudo_rows()
    out = {}
    for net in NETS:
        eng, kpi = ENG[net]
        f8, s8 = alone[8]
        thr8 = throughput(data, f8, net + s8, net)
        out[net] = {}
        for n, (exp, suf) in sorted(alone.items()):
            thr = throughput(data, exp, net + suf, net)
            if thr is None or thr8 is None:
                continue
            ms = A.aggr(data, exp, net + suf, net, "gpu_ms", "median")[0]
            x = dict(thr=thr, alone_ms=ms, p=kpi * C.PRESENCE_US * 1e-3 / ms,
                     L_scaled=prof[net]["L"] * thr / thr8, D_scaled=prof[net]["D"] * thr / thr8,
                     L_sudo=None, D_sudo=None)
            r = sudo.get((eng, SHARE.get(n)))     # no root profile at 2 or 14 SMs
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


def alone_p99(data, net, n, alone=ALONE):
    """The p99 of a network alone on n SMs (median over runs), from this campaign's alone rows."""
    exp, suf = alone[n]
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
    u = e[e < 0]
    return "  %-44s n %2d  mean |e| %4.1f  max |e| %4.1f  under %2d (mean %4.1f)  worst %+5.1f  bias %+5.1f" % (
        label, len(e), np.abs(e).mean(), np.abs(e).max(), len(u), -u.mean() if len(u) else 0.0,
        min(e.min(), 0), e.mean())


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


def freeze(path, rows, head):
    """rows to path (head first, then the other columns sorted) and its sha256 next to it."""
    keys = head + sorted({k for r in rows for k in r} - set(head))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    open(path + ".sha256", "w").write("%s  %s\n" % (sha256(path), os.path.basename(path)))
    print("  frozen %d rows: %s, sha256 %s" % (len(rows), path, sha256(path)))


def frozen_rows(path):
    """The rows of a frozen file, after checking it against the sha256 written when it was frozen."""
    want = open(path + ".sha256").read().split()[0]
    if sha256(path) != want:
        raise SystemExit("geometry.py: %s changed after it was frozen" % path)
    print("  frozen predictions intact (sha256 %s...)" % want[:12])
    return list(csv.DictReader(open(path)))


def cmd_triple_predict():
    data = A.load(P.BENCH)
    if any(k[0] == TRIPLE and "+" in k[1] for k in data):
        raise SystemExit("geometry.py: stress-triple pairs exist already; the predictions must be frozen before them")
    freeze(FROZEN, triple_rows(data), ["config", "victim", "victim_sms", "position", "g"])


def cmd_triple_compare(current=False):
    """Score 8+4 and 8+4+4. By default the frozen predictions (the test as it was run); with current,
    the predictions of the model as it is now, e.g. after the presence constant changed (2026-10-29):
    a re-scoring after the fact, written to its own file so the frozen record stays."""
    data = A.load(P.BENCH)
    if current:
        preds = triple_rows(data)
        print("  current model's predictions (not frozen: presence %.2f us)" % C.PRESENCE_US)
    else:
        preds = frozen_rows(FROZEN)
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


# ------------------------------------------------------------------ final check

def normalized(b, p_t, L_t, n, L_c):
    """The normalized model's p99 slowdown of a target on n SMs: its L2 GB/s scaled to 8 SMs (per SM,
    times 8) before the sensitivity is read. On 8 SMs it is predict() itself."""
    return predict(b, p_t, 8.0 * L_t / n, L_c)


def check_rows(data):
    """Every tenant of every pair of the three check splits, and every target beside spin (the presence
    term alone), with the normalized and the as-fitted prediction. Reads only the alone rows."""
    inp, b = inputs(data, CHECK_ALONE), coefficients()
    missing = [(t, n) for t in NETS for n in CHECK_ALONE if n not in inp[t]]
    if missing:
        raise SystemExit("geometry.py: alone rows missing for %s; run stress-sizes-check first" % missing)
    rows = []
    for exp, (name, nt, nc) in CHECK.items():
        for t in NETS:
            p = round(1 + inp[t][nt]["p"], 4)
            rows.append(dict(split=name, config=t + "-spin", victim=t, victim_sms=nt, position="target",
                             pred_normalized=p, pred_fitted=p))
            for c in NETS:
                if c == t:
                    continue
                for v, n, pos, ((o, m),) in victims(t + "+" + c, nt, nc):
                    x, L_c = inp[v][n], inp[o][m]["L_scaled"]
                    rows.append(dict(split=name, config=t + "+" + c, victim=v, victim_sms=n, position=pos,
                                     pred_normalized=round(normalized(b, x["p"], x["L_scaled"], n, L_c), 4),
                                     pred_fitted=round(predict(b, x["p"], x["L_scaled"], L_c), 4)))
    return rows


def cmd_check_predict():
    data = A.load(P.BENCH)
    if any(k[0] in CHECK for k in data):
        raise SystemExit("geometry.py: rows of the check splits exist already; the predictions must be frozen before them")
    freeze(CHECK_FROZEN, check_rows(data), ["split", "config", "victim", "victim_sms", "position"])


def cmd_check_compare():
    """Score the frozen predictions: each tenant's p99 against its network alone at its size, both from
    the check's session. The spin rows score the presence term alone; the pairs both models."""
    preds, data = frozen_rows(CHECK_FROZEN), A.load(P.BENCH)
    exp_of = {name: exp for exp, (name, nt, nc) in CHECK.items()}
    err, out = {}, []
    for p in preds:
        exp, cfg, v, n, pos = exp_of[p["split"]], p["config"], p["victim"], int(p["victim_sms"]), p["position"]
        if pos == "target":
            m = A.aggr(data, exp, cfg, v, "gpu_ms", "p99")
            m = m[0] if m else None
        else:
            m = corunner_p99(os.path.join(P.BENCH, FAMILY_DIR[exp], cfg), int(pos[2:]))
        a = alone_p99(data, v, n, CHECK_ALONE)
        if not (m and a):
            continue
        s = m / a
        split = "%d+%d" % CHECK[exp][1:]
        row = dict(p, measured=round(s, 4))
        for k in ("normalized", "fitted"):
            row["err_" + k] = round((float(p["pred_" + k]) - s) * 100, 2)
        if cfg.endswith("-spin"):
            err.setdefault((split, "dispatch (spin)"), []).append(row["err_normalized"])
        else:
            for k in ("normalized", "fitted"):
                err.setdefault((split, pos if pos == "target" else "co-runner", k,
                                block(*cfg.split("+"))), []).append(row["err_" + k])
        out.append(row)
    print("\nfinal check: predicted minus measured p99 slowdown (points)")
    for k in sorted(err):
        print(summary(" / ".join(k), err[k]))
    write("geometry-check.csv", out)


# ------------------------------------------------------------------ power

def power_log(path):
    """[(uptime_s, gpu_soc_mW, cpu_cv_mW, sys5v_mW)] from a sweep _power.log. Since 2026-11-10 a line
    also carries the GPU clock and temperature (nine columns, read by periodic.py clock_log)."""
    out = []
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) in (7, 9) and f[0][0].isdigit():
            v = [float(x) for x in f[:7]]
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


# ------------------------------------------------------------------ energy at equal work
#
# The professor (2026-11-09): keep mean power, throughput and energy per inference apart, and compare
# networks running together with the same networks alone at equal work, the same number of inferences of
# each network, by the total energy. From the 30 s runs this is computed: the pair side by side at its
# measured power and rates until the faster has done its share, then the slower alone on its partition at
# its alone rate and energy per inference; alone, one network after the other. energy-same-n measures it:
# both networks launched together, each stopping after the same N, the rails integrated until the last
# exits, against the same N run by each network alone, on its partition and on the whole GPU.

# Each network alone at each size: the family and configuration suffix of its runs. 8 SMs is
# stress-triple's own alone row, the session of the 8+4 pairs; 16 SMs is energy-alone-16.
ENERGY_ALONE = {4: ("stress-split-412", "-alone"), 6: ("stress-split-610", "-alone"),
                8: ("stress-triple", "-alone"), 10: ("stress-sizes", "-alone10"),
                12: ("stress-split-124", "-alone"), 16: ("energy-alone-16", "-alone16")}
ENERGY_PAIRS = {"stress-split-412": (4, 12), "stress-split-610": (6, 10), "stress-triple": (8, 4),
                "stress-split-124": (12, 4)}


def power_rows():
    """geometry-power.csv (cmd_power), grouped by (family, config)."""
    by = {}
    for r in csv.DictReader(open(os.path.join(C.AGG, "geometry-power.csv"))):
        by.setdefault((r["family"], r["config"]), []).append(r)
    return by


def same_n_computed(by, fam, a, b):
    """(t, e, t_alone, e_alone) per inference of each network, for the pair a (first partition of fam) +
    b, from the 30 s runs: side by side at the pair's median power and rates until the faster has done
    one inference's worth, then the slower alone at its alone rate and energy per inference; alone, one
    after the other. Every term scales with the number of inferences, so the ratios do not depend on it.
    None if a row is missing or a co-runner count is."""
    med = lambda rs, k: st.median(float(r[k]) for r in rs)
    nt, nc = ENERGY_PAIRS[fam]
    rs = by.get((fam, a + "+" + b))
    al = [by.get((ENERGY_ALONE[n][0], x + ENERGY_ALONE[n][1])) for x, n in ((a, nt), (b, nc))]
    if not rs or None in al or any(int(r["corunner_missing"]) for r in rs):
        return None
    P = med(rs, "rails_mW") / 1e3
    rate = [med(rs, "target_inf_s"), st.median(float(r["corunner_inf_s"]) for r in rs)]
    thr = [med(x, "target_inf_s") for x in al]
    E = [med(x, "mJ_per_inference") / 1e3 for x in al]
    fast = 0 if rate[0] >= rate[1] else 1
    slow = 1 - fast
    t1 = 1.0 / rate[fast]                       # the faster's inference, side by side
    left = 1.0 - rate[slow] * t1                # what the slower still has to do, alone
    return t1 + left / thr[slow], P * t1 + left * E[slow], sum(1.0 / x for x in thr), sum(E)


def run_energy(prefix):
    """(seconds, joules, inferences per network) of one run: the three rails summed and integrated with
    trapezoids from the first inference of any network in the run (warm-up included) to the last, the
    ends interpolated between the 200 ms samples. The rows are <prefix>_latencies.csv (solo) or
    <prefix>_<label>_latencies.csv (pair). None without rows or a power log."""
    lats = sorted(glob.glob(prefix + "_*latencies.csv"))
    if not lats or not os.path.exists(prefix + "_power.log"):
        return None
    spans, counts = [], []
    for f in lats:
        rows = [(int(r["t_start_ns"]) / 1e9, int(r["t_end_ns"]) / 1e9) for r in csv.DictReader(open(f))]
        spans += rows
        counts.append(len(rows))
    t0, t1 = min(s for s, e in spans), max(e for s, e in spans)
    return t1 - t0, rails_energy(power_log(prefix + "_power.log"), t0, t1), counts


def rails_energy(p, t0, t1):
    """Joules of the three rails from t0 to t1 (seconds, CLOCK_MONOTONIC): the power_log samples p
    integrated with trapezoids, the ends interpolated between the 200 ms samples."""
    t = np.array([x[0] for x in p])
    w = np.array([(x[1] + x[2] + x[3]) / 1e3 for x in p])
    g = np.concatenate(([t0], t[(t > t0) & (t < t1)], [t1]))
    v = np.interp(g, t, w)
    return float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(g)))


def cmd_energy():
    by = power_rows()
    med = lambda rs, k: st.median(float(r[k]) for r in rs)
    rng = lambda v: "%.2f (%.2f-%.2f)" % (st.median(v), min(v), max(v))
    print("each network alone: power W, throughput inf/s, energy per inference mJ (medians of the runs)")
    for net in NETS:
        cells = []
        for n, (f, suf) in sorted(ENERGY_ALONE.items()):
            rs = by.get((f, net + suf))
            cells.append("%2d: %4.1f W %5.0f/s %6.1f mJ" % (n, med(rs, "rails_mW") / 1e3, med(rs, "target_inf_s"),
                                                          med(rs, "mJ_per_inference")) if rs else "%2d: -" % n)
        print("  %-12s %s" % (net, " | ".join(cells)))

    print("\npairs, the same number of inferences of each network, computed from the 30 s runs:")
    print("mean power together and alone in sequence (W), time and energy together / alone (median, range)")
    for fam, (nt, nc) in ENERGY_PAIRS.items():
        res = [x for x in (same_n_computed(by, fam, *c.split("+")) for f, c in by
                           if f == fam and c.count("+") == 1) if x]
        if res:
            print("  %2d+%-2d pairs %2d  %4.1f W  %4.1f W  time %s  energy %s" % (
                nt, nc, len(res), st.median(e / t for t, e, ta, ea in res), st.median(ea / ta for t, e, ta, ea in res),
                rng([t / ta for t, e, ta, ea in res]), rng([e / ea for t, e, ta, ea in res])))

    d = os.path.join(P.BENCH, "energy-same-n")
    if not os.path.isdir(d):
        return

    def measured(cfg):
        """Median (seconds, joules) over the runs of one energy-same-n configuration, and the counts."""
        rs = [x for x in (run_energy(f[:-len("_power.log")]) for f in sorted(glob.glob(os.path.join(d, cfg, "r*_power.log")))) if x]
        return (st.median(x[0] for x in rs), st.median(x[1] for x in rs), sorted({c for x in rs for c in x[2]})) if rs else None

    print("\ndirect check (energy-same-n): the same N inferences of each network, measured;")
    print("together against alone in sequence on the same partitions and on the whole GPU, and the computed ratio")
    for cfg in sorted(os.listdir(d)):
        if "+" not in cfg:
            continue
        a, b = cfg.split("+")
        tog, pa, pb, wa, wb = (measured(c) for c in (cfg, a + "-alone08", b + "-alone04", a + "-alone16", b + "-alone16"))
        if None in (tog, pa, pb):
            print("  %-24s incomplete" % cfg)
            continue
        line = "  %-24s together %6.1f s %7.0f J (inferences %s)  same partitions: time %.2f energy %.2f" % (
            cfg, tog[0], tog[1], tog[2], tog[0] / (pa[0] + pb[0]), tog[1] / (pa[1] + pb[1]))
        if wa and wb:
            line += "  whole GPU: time %.2f energy %.2f" % (tog[0] / (wa[0] + wb[0]), tog[1] / (wa[1] + wb[1]))
        c = same_n_computed(by, "stress-triple", a, b)
        if c:
            line += "  computed: time %.2f energy %.2f" % (c[0] / c[2], c[1] / c[3])
        print(line)


# ------------------------------------------------------------------ tenants
#
# The professor (2026-11-09): split the 16 SMs hierarchically, 16 -> 8+8 -> 8+4+4 -> 4+4+4+4, and find how
# far adding tenants pays in time and energy. energy-tenants: k of the development networks launched
# together on the k-tenant layout (LAYOUT; in 8+4+4 the first network named is on 8 SMs), each doing the
# same N inferences and keeping its partition until it is done, so running tenants are never resized;
# against the same networks one after the other alone on the whole GPU, the family's own alone16 rows.
# Per run: the time from the first inference to the last and the rails' energy over it (run_energy), and
# each network's p99 while all k are running, against its p99 alone on the whole GPU. The networks
# finish at very different times (MobileNet does 10000 inferences in seconds, RetinaNet in minutes), so
# the window in which all of them run can be short: the number of inferences behind each p99 is printed.
# The professor (2026-11-10): the same sets on the shared GPU, with no static partition, under the MPS
# daemon (energy-tenants-mpsdefault) and without it (energy-tenants-nomps); a triple is then one set, not
# three layouts, and each set is also printed beside the static layouts of energy-tenants with the same
# networks. Those ratios come from another session, each against its own alone runs.
LAYOUT = {2: "8+8", 3: "8+4+4", 4: "4+4+4+4"}


def p99_together(prefix):
    """{network: (p99 gpu_ms, inferences)} of one group run, over the timed ("bench") rows lying wholly in
    the window where every network is in its timed phase: from the latest first start to the earliest
    last end. A network with no row in the window gets (None, 0); a run without rows gives {}."""
    rows = {}
    for f in glob.glob(prefix + "_*_latencies.csv"):
        rows[f[len(prefix) + 1:-len("_latencies.csv")]] = [
            (int(r["t_start_ns"]), int(r["t_end_ns"]), float(r["gpu_ms"]))
            for r in csv.DictReader(open(f)) if r["phase"] == "bench"]
    if not rows or not all(rows.values()):
        return {}
    lo = max(min(s for s, e, g in v) for v in rows.values())
    hi = min(max(e for s, e, g in v) for v in rows.values())
    out = {}
    for net, v in rows.items():
        x = [g for s, e, g in v if lo <= s and e <= hi]
        out[net] = (float(np.percentile(x, 99)), len(x)) if x else (None, 0)
    return out


def tenants_prefixes(d, cfg):
    """The run prefixes (<d>/<cfg>/r1, r2, ...) of one configuration of a tenants family, from its power logs."""
    return [f[:-len("_power.log")] for f in sorted(glob.glob(os.path.join(d, cfg, "r*_power.log")))]


def tenants_alone(fam):
    """{network: (seconds, joules, p99 gpu_ms)} of each network alone on the whole GPU in a tenants family
    (its -alone16 configurations): medians over the runs."""
    d = os.path.join(P.BENCH, fam)
    if not os.path.isdir(d):
        sys.exit("no %s runs in %s" % (fam, d))
    alone = {}
    for cfg in sorted(c for c in os.listdir(d) if c.endswith("-alone16")):
        ps = tenants_prefixes(d, cfg)
        es = [x for x in map(run_energy, ps) if x]
        p99 = [np.percentile([float(r["gpu_ms"]) for r in csv.DictReader(open(p + "_latencies.csv"))
                              if r["phase"] == "bench"], 99) for p in ps]
        if es:
            alone[cfg[:-len("-alone16")]] = (st.median(x[0] for x in es), st.median(x[1] for x in es), st.median(p99))
    return alone


def tenants_groups(fam, alone):
    """{configuration: (seconds, joules, time ratio, energy ratio, {network: (p99 ratio, inferences)})} of the
    group runs of a tenants family, medians over the runs. The ratios are together over the same networks one
    after the other alone on the whole GPU (alone, from tenants_alone); a network's p99 ratio is its p99 while
    every network runs (p99_together) over its p99 alone, and is missing when no run has a row in that
    window. A configuration with no complete run, or a network without alone runs, is left out."""
    d = os.path.join(P.BENCH, fam)
    res = {}
    for cfg in (c for c in os.listdir(d) if "+" in c):
        nets = cfg.split("+")
        got = [(x, p99_together(p)) for p in tenants_prefixes(d, cfg) for x in [run_energy(p)] if x]
        if not got or any(n not in alone for n in nets):
            continue
        t, e = st.median(x[0] for x, q in got), st.median(x[1] for x, q in got)
        p99 = {}
        for n in nets:
            v = [q[n] for x, q in got if q.get(n, (None, 0))[0] is not None]
            if v:
                p99[n] = (st.median(a for a, c in v) / alone[n][2], st.median(c for a, c in v))
        res[cfg] = (t, e, t / sum(alone[n][0] for n in nets), e / sum(alone[n][1] for n in nets), p99)
    return res


def cmd_tenants():
    """tenants [family [alone family]]: energy-tenants (the MPS static layouts) by default. The alone runs
    come from the family itself unless another is named: energy-tenants-nomps has none and takes those of
    energy-tenants-mpsdefault, measured in the same session. A shared-GPU family is also printed beside
    the static layouts of energy-tenants with the same networks."""
    fam = sys.argv[2] if len(sys.argv) > 2 else "energy-tenants"
    alone = tenants_alone(sys.argv[3] if len(sys.argv) > 3 else fam)
    shared = fam != "energy-tenants"
    static = tenants_groups("energy-tenants", tenants_alone("energy-tenants")) if shared else {}
    res = tenants_groups(fam, alone)
    rng = lambda v: "%.2f (%.2f-%.2f)" % (st.median(v), min(v), max(v))
    span = lambda v: "%.2f" % v[0] if len(v) == 1 else "%.2f-%.2f" % (min(v), max(v))
    where = lambda k: "shared" if shared else LAYOUT[k]

    print("each network alone on the whole GPU, the same N inferences (medians of the runs):")
    for net, (t, e, p) in alone.items():
        print("  %-12s %6.1f s %6.0f J  p99 %.3f ms" % (net, t, e, p))

    print("\n%s, together against the same networks one after the other alone on the whole GPU:" % fam)
    print("time and energy ratios (medians of the runs); each network's p99 while all run, against its p99")
    print("alone on the whole GPU, with the inferences behind it" +
          ("; then the static layouts of energy-tenants" if shared else ""))
    cfgs = sorted((c for c in os.listdir(os.path.join(P.BENCH, fam)) if "+" in c), key=lambda c: (c.count("+"), c))
    for cfg in cfgs:
        nets = cfg.split("+")
        if cfg not in res:
            print("  %-8s %-40s incomplete" % (where(len(nets)), cfg))
            continue
        t, e, rt, re, p99 = res[cfg]
        cells = "  ".join("%s %s" % (n, "%.2f (%d)" % p99[n] if n in p99 else "-") for n in nets)
        line = "  %-8s %-40s %6.1f s %6.0f J  time %.2f  energy %.2f  | %s" % (where(len(nets)), cfg, t, e, rt, re, cells)
        same = [v for c, v in static.items() if set(c.split("+")) == set(nets)]
        if same:
            line += "  | static time %s energy %s" % (span([v[2] for v in same]), span([v[3] for v in same]))
        print(line)

    print("\nby tenant count: median (range) of the time and energy ratios")
    for k in sorted({c.count("+") + 1 for c in res}):
        for label, keep in (("all", lambda c: True), ("with RetinaNet", lambda c: "retinanet" in c),
                            ("without RetinaNet", lambda c: "retinanet" not in c)):
            v = [x for c, x in res.items() if c.count("+") + 1 == k and keep(c)]
            if v:
                print("  %d tenants %-8s %-18s %2d %s  time %s  energy %s" % (
                    k, where(k), label, len(v), "sets" if shared else "layouts",
                    rng([x[2] for x in v]), rng([x[3] for x in v])))


# ------------------------------------------------------------------ energy model
#
# The professor (2026-11-10): start from the additive approximation, generalize it to three tenants, check
# how well it predicts the powers already measured and, above all, whether it is good enough to pick the
# allocation with the least energy despite its errors.
#
# Power while a set S of networks runs, each on its partition:
#
#     P(S) = sum over S of P_alone(network, its SMs) - (|S| - 1) B
#
# B is the power of the module that networks running together draw only once (the base the plain sum
# would count |S| times): the median over the 30 s pairs of P(a) + P(b) - P(a+b). Time: a network runs at
# u_m times its alone rate while m others run, u_1 from the 30 s pairs and u_2 from the 30 s triples; no
# 30 s quads exist, so u_3 = u_2 (a known gap: the quad is the one case it extrapolates). The inputs are
# the networks alone at each size (ENERGY_ALONE) and the 30 s pairs and triples. It is checked only on
# runs it never used: every phase (between two network ends) of the equal-work runs, energy-same-n and
# energy-tenants, and their totals.
#
# The decision: for each set of networks, the allocations measured in one session (energy-tenants: the
# layouts and one after the other alone on 16 SMs; energy-same-n: 8+4 either way round and one after the
# other on 16 SMs), the one with the least predicted energy against the one with the least measured
# energy. The regret is the measured energy of the model's choice over the best one, minus 1: what its
# errors cost. Allocations from different sessions are not compared, because the power drifts a few
# percent between sessions, as much as the differences being ranked. The shared-GPU families have no
# input of their own here, so the model does not predict them.
TENANT_SIZES = {"energy-same-n": {2: (8, 4)}, "energy-tenants": {2: (8, 8), 3: (8, 4, 4), 4: (4, 4, 4, 4)}}


def run_phases(prefix):
    """The phases of one equal-work run, cut at every network's last inference: ([(t0, t1, [networks
    running], mean W)], {network: (first start, last end, inferences)}). The first phase starts at the
    first inference of any network (warm-up included, as in run_energy); a phase under 50 ms, between two
    ends that nearly coincide, is dropped. A solo run's network is named after its configuration. ([], {})
    without rows or a power log."""
    lats = sorted(glob.glob(prefix + "_*latencies.csv"))
    if not lats or not os.path.exists(prefix + "_power.log"):
        return [], {}
    run, cfg = os.path.basename(prefix), os.path.basename(os.path.dirname(prefix))
    span = {}
    for f in lats:
        net = os.path.basename(f)[len(run) + 1:-len("_latencies.csv")] or cfg.split("-")[0]
        rows = [(int(r["t_start_ns"]) / 1e9, int(r["t_end_ns"]) / 1e9) for r in csv.DictReader(open(f))]
        span[net] = (min(s for s, e in rows), max(e for s, e in rows), len(rows))
    p = power_log(prefix + "_power.log")
    t = np.array([x[0] for x in p])
    w = np.array([(x[1] + x[2] + x[3]) / 1e3 for x in p])
    cuts = sorted({min(s[0] for s in span.values())} | {s[1] for s in span.values()})
    phases = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a < 0.05:
            continue
        g = np.concatenate(([a], t[(t > a) & (t < b)], [b]))
        v = np.interp(g, t, w)
        running = [n for n, s in span.items() if s[0] < b - 1e-3 and s[1] > a + 1e-3]
        phases.append((a, b, running, float(np.sum((v[1:] + v[:-1]) / 2 * np.diff(g))) / (b - a)))
    return phases, span


def energy_simulate(sizes, n_inf, rate, power, u):
    """(seconds, joules) of the networks in sizes ({network: SMs}) doing n_inf[network] inferences side by
    side from t = 0. While m others run, a network goes at u[m] times rate(network, SMs); when one
    finishes, its partition stays idle and the others go on at the rate of the smaller set. power(S) is
    the model's power of the set S, a list of (network, SMs)."""
    left = {n: float(n_inf[n]) for n in sizes}
    t = e = 0.0
    while left:
        r = {n: rate(n, sizes[n]) * u[len(left) - 1] for n in left}
        dt = min(left[n] / r[n] for n in left)
        e += dt * power([(n, sizes[n]) for n in left])
        t += dt
        for n in list(left):
            left[n] -= r[n] * dt
            if left[n] <= 1e-6:
                del left[n]
    return t, e


def cmd_energy_model():
    by = power_rows()
    med = lambda rs, k: st.median(float(r[k]) for r in rs)
    alone = lambda n, k: by[(ENERGY_ALONE[k][0], n + ENERGY_ALONE[k][1])]
    pw = lambda n, k: med(alone(n, k), "rails_mW") / 1e3
    rate = lambda n, k: med(alone(n, k), "target_inf_s")
    complete = lambda rs: not any(int(r["corunner_missing"]) for r in rs)
    summ = lambda e: "n %3d  mean %+5.1f %%  mean |e| %4.1f %%  max |e| %4.1f %%" % (
        len(e), st.mean(e), st.mean(abs(x) for x in e), max(abs(x) for x in e))

    # 1. The inputs: B and u_1 from the 30 s pairs, u_2 from the 30 s triples (8+4+4, in stress-triple).
    Bs, u1 = [], []
    for (f, c), rs in by.items():
        if f in ENERGY_PAIRS and c.count("+") == 1 and complete(rs):
            (a, b), (ka, kb) = c.split("+"), ENERGY_PAIRS[f]
            Bs.append(pw(a, ka) + pw(b, kb) - med(rs, "rails_mW") / 1e3)
            u1 += [med(rs, "target_inf_s") / rate(a, ka), med(rs, "corunner_inf_s") / rate(b, kb)]
    B = st.median(Bs)
    power = lambda S: sum(pw(n, k) for n, k in S) - (len(S) - 1) * B
    err, u2 = [], []
    for (f, c), rs in by.items():
        if f == "stress-triple" and c.count("+") == 2 and complete(rs):
            a, b, d = c.split("+")
            err.append(100 * (power([(a, 8), (b, 4), (d, 4)]) / (med(rs, "rails_mW") / 1e3) - 1))
            co = [[float(x) for x in r["corunner_inf_s"].split(";")] for r in rs]
            u2 += [med(rs, "target_inf_s") / rate(a, 8), st.median(x[0] for x in co) / rate(b, 4),
                   st.median(x[1] for x in co) / rate(d, 4)]
    u = {0: 1.0, 1: st.median(u1), 2: st.median(u2), 3: st.median(u2)}
    print("B = %.1f W (median of %d pairs, range %.1f-%.1f); u1 = %.3f, u2 = %.3f (u3 = u2)" % (
        B, len(Bs), min(Bs), max(Bs), u[1], u[2]))
    print("30 s triples, power predicted vs measured: " + summ(err))

    # 2. The equal-work runs, never used above: the power of every phase, and each run's totals.
    phase_err, runs = {}, {}
    for fam in TENANT_SIZES:
        for d in sorted(glob.glob(os.path.join(P.BENCH, fam, "*"))):
            cfg = os.path.basename(d)
            if "-alone" in cfg:
                n, k = cfg.split("-alone")
                sizes = {n: int(k)}
            else:
                nets = cfg.split("+")
                sizes = dict(zip(nets, TENANT_SIZES[fam][len(nets)]))
            if not all(k in ENERGY_ALONE for k in sizes.values()):
                continue
            for pl in sorted(glob.glob(os.path.join(d, "r*_power.log"))):
                phases, span = run_phases(pl[:-len("_power.log")])
                if not phases:
                    continue
                e_meas = e_pow = 0.0
                for a, b, S, w in phases:
                    p = power([(n, sizes[n]) for n in S])
                    phase_err.setdefault(len(S), []).append(100 * (p / w - 1))
                    e_meas += (b - a) * w
                    e_pow += (b - a) * p
                t_meas = max(s[1] for s in span.values()) - min(s[0] for s in span.values())
                t_pred, e_pred = energy_simulate(sizes, {n: s[2] for n, s in span.items()}, rate, power, u)
                runs.setdefault((fam, cfg), []).append((t_meas, e_meas, e_pow, t_pred, e_pred))
    print("\nphase power, predicted vs measured, by networks running:")
    for k in sorted(phase_err):
        print("  %d running  %s" % (k, summ(phase_err[k])))
    tot = {key: [st.median(x[i] for x in v) for i in range(5)] for key, v in runs.items()}
    print("\ntotals (medians of the runs): measured | energy with the measured phase times | full prediction")
    for (fam, cfg), m in sorted(tot.items()):
        print("  %-15s %-38s T %6.1f s E %6.0f J | E %+5.1f %% | T %+5.1f %% E %+5.1f %%" % (
            fam, cfg, m[0], m[1], 100 * (m[2] / m[1] - 1), 100 * (m[3] / m[0] - 1), 100 * (m[4] / m[1] - 1)))

    # 3. The decision, within each session: the allocation with the least energy, predicted against
    #    measured, and the regret of the predicted choice.
    print("\nleast energy per set of networks, in one session: allocation, measured J, predicted J")
    sets = {}
    for (fam, cfg), m in tot.items():
        if "+" in cfg:
            sets.setdefault((fam, frozenset(cfg.split("+"))), []).append((cfg, m[1], m[4]))
    regrets, right, lay_regrets, lay_right = [], 0, [], 0
    for (fam, nets), opts in sorted(sets.items(), key=lambda x: (x[0][0], len(x[0][1]), sorted(x[0][1]))):
        seq = [tot.get((fam, n + "-alone16")) for n in nets]
        if None in seq:
            continue
        opts = opts + [("one after the other on 16", sum(s[1] for s in seq), sum(s[4] for s in seq))]
        best = min(opts, key=lambda o: o[1])
        pick = min(opts, key=lambda o: o[2])
        regret = 100 * (pick[1] / best[1] - 1)
        regrets.append(regret)
        right += pick is best
        print("  %-15s %s" % (fam, "  ".join("%s %.0f/%.0f" % (o[0], o[1], o[2]) for o in opts)))
        print("  %15s picks %s, best %s: regret %.1f %%" % ("", pick[0], best[0], regret))
        lay = opts[:-1]                 # the layouts alone: which network gets the larger partition
        if len(lay) > 1:
            lb, lp = min(lay, key=lambda o: o[1]), min(lay, key=lambda o: o[2])
            lay_regrets.append(100 * (lp[1] / lb[1] - 1))
            lay_right += lp is lb
    print("  the model picks the least-energy allocation in %d of %d sets; regret median %.1f %%, max %.1f %%" % (
        right, len(regrets), st.median(regrets), max(regrets)))
    print("  among the layouts only (no one after the other): %d of %d sets; regret max %.1f %%" % (
        lay_right, len(lay_regrets), max(lay_regrets)))


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
    assert normalized(b, 0.05, 100, 8, 50) == predict(b, 0.05, 100, 50)                    # 8 SMs: unchanged
    assert abs(normalized(b, 0.05, 50, 4, 50) - predict(b, 0.05, 100, 50)) < 1e-12         # per SM, times 8
    assert sorted(CHECK_ALONE) == [2, 6, 8, 14] and CHECK_ALONE[2] == ("exp35", "-alone02")
    with tempfile.TemporaryDirectory() as d:
        # 10 W from 0 to 30 s; inferences from 10 to 20 s: 10 s and 100 J, the ends interpolated
        open(os.path.join(d, "r1_power.log"), "w").write("uptime_s a b c d e f\n" + "".join(
            "%.2f 5000 2000 0 0 0 0\n" % (0.2 * i) for i in range(151)))
        hdr = "phase,index,image,prep_ms,wall_ms,gpu_ms,t_start_ns,t_end_ns\n"
        open(os.path.join(d, "r1_a_latencies.csv"), "w").write(hdr + "warmup,0,x,0,1,1,10050000000,11000000000\n")
        open(os.path.join(d, "r1_b_latencies.csv"), "w").write(hdr + "bench,0,x,0,1,1,12000000000,20000000000\n")
        s, j, n = run_energy(os.path.join(d, "r1"))
        assert abs(s - 9.95) < 1e-9 and abs(j - 99.5) < 1e-6 and n == [1, 1]
    by = {("stress-triple", "a+b"): [dict(rails_mW="30000", target_inf_s="100", corunner_inf_s="50", corunner_missing="0")],
          ("stress-triple", "a-alone"): [dict(target_inf_s="200", mJ_per_inference="100")],
          ("stress-split-412", "b-alone"): [dict(target_inf_s="100", mJ_per_inference="200")]}
    t, e, ta, ea = same_n_computed(by, "stress-triple", "a", "b")
    # side by side 0.01 s at 30 W (b does 0.5), then b alone 0.005 s and 0.1 J; alone 0.015 s, 0.3 J
    assert abs(t - 0.015) < 1e-12 and abs(e - 0.4) < 1e-12 and abs(ta - 0.015) < 1e-12 and abs(ea - 0.3) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        # a's timed rows run from 10 to 20.5 s, b's from 15 to 29.5 s: both are timed from 15 to 20.5 s,
        # which holds six rows of each; the warm-up row is never counted
        hdr = "phase,index,image,prep_ms,wall_ms,gpu_ms,t_start_ns,t_end_ns\n"
        for net, first, last in (("a", 10, 20), ("b", 15, 29)):
            open(os.path.join(d, "r1_%s_latencies.csv" % net), "w").write(
                hdr + "warmup,0,x,0,1,99,16000000000,16100000000\n" + "".join(
                    "bench,%d,x,0,1,%d,%d,%d\n" % (i, i, i * 10**9, i * 10**9 + 5 * 10**8)
                    for i in range(first, last + 1)))
        q = p99_together(os.path.join(d, "r1"))
        assert q["a"][1] == 6 and q["b"][1] == 6
        assert abs(q["a"][0] - np.percentile(range(15, 21), 99)) < 1e-12
    t, e = energy_simulate({"a": 8, "b": 8}, {"a": 10, "b": 20}, lambda n, k: 10.0,
                           lambda S: 10.0 * len(S), {0: 1.0, 1: 0.5})
    assert abs(t - 3.0) < 1e-9 and abs(e - 50.0) < 1e-9
    print("  selftest passed")


if __name__ == "__main__":
    {"inputs": cmd_inputs, "splits": cmd_splits, "triple-predict": cmd_triple_predict,
     "triple-compare": cmd_triple_compare, "check-predict": cmd_check_predict,
     "check-compare": cmd_check_compare, "energy": cmd_energy, "tenants": cmd_tenants,
     "energy-model": cmd_energy_model,
     "triple-current": lambda: cmd_triple_compare(True), "power": cmd_power, "selftest": selftest}.get(
        CMD, lambda: sys.exit(__doc__))()
