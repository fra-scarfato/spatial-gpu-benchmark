#!/usr/bin/env python3
"""compose.py - does the stress of two co-runners add up? (Daniel's composition test, 2026-10-22)

    python3 analysis/compose.py            the tables, and compose-4sm.csv in bench/aggregate
    python3 analysis/compose.py selftest   the ruler inversion and the log parsing

The reporter sits on 8 SMs (LAYOUT=target). Beside it, reporter.sh compose-4sm runs two smooth SAXPY
on 4 SMs each at duties 25/50/75/100 %, every combination, each duty alone on the first 4-SM
partition, and calibration-smooth gives the 8-SM ruler of the same session (spin, smooth-D025..D100,
topped by top8, the plain kernel back to back). Every reading becomes an equivalent stress through
that ruler, L_eq = the ruler's GB/s with the same reporter reading (stress.py's invert, p99 and
median of the 0.3 ms windows), and each combination is compared with the sum of its two halves:

    ratio = L_eq(a + b) / (L_eq(a alone) + L_eq(b alone))

1 is additive, below 1 saturating, above 1 amplified. The same ratio on the stressors' own GB/s
(what itfproc moved, from its logs) separates two causes: the stressors slowing each other down
(they move less together) from the memory system answering the same traffic with more latency.

Sacrifices: the halves' alone readings are taken on the first partition only. Every combination is
run in both orders, so the second partition is checked against the first rather than measured
alone. A reading above top8 cannot be converted and is flagged, never extrapolated.
"""
import csv
import glob
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.argv = sys.argv[:2]           # stress.py reads its geometry from argv[2]; the default suits us
import stress as S  # noqa: E402

ROOT = os.path.join(S.BENCH, "stress-reporter-compose")
DUTIES = ("025", "050", "075", "100")
COORDS = ("p99", "median")


def gbs(log):
    """The mean GB/s itfproc moved over the run: the last of its cumulative figures, or None."""
    vals = re.findall(r"bytes moved:.*\(([0-9.]+) GB/s\)", open(log, errors="replace").read())
    return float(vals[-1]) if vals else None


def config(name):
    """One configuration over its runs: the reporter's p99 and median (median over runs), their
    spread (max - min over runs), and each co-runner's GB/s (median over runs), or None."""
    d = os.path.join(ROOT, name)
    runs = sorted(glob.glob(os.path.join(d, "r*_latency.txt")))
    if not runs:
        return None
    lat = [S.reporter_latency(r) for r in runs]
    row = {"runs": len(runs)}
    for c in COORDS:
        v = [x[c] for x in lat]
        row[c], row[c + "_spread"] = st.median(v), max(v) - min(v)
    for i in (0, 1):
        g = [gbs(f) for f in sorted(glob.glob(os.path.join(d, "r*_c%d.log" % i)))]
        g = [x for x in g if x is not None]
        row["gbs%d" % i] = st.median(g) if g else None
    return row


def ruler():
    """The 8-SM ruler of this session: per coordinate, (GB/s, reading) from spin (0 GB/s: resident,
    no traffic, the zero of smooth88) up through the smooth duties to top8."""
    pts = []
    for name in ["spin"] + ["smooth-D" + d for d in DUTIES] + ["top8"]:
        r = config(name)
        if r is None:
            raise SystemExit("compose.py: ruler point %s missing in %s" % (name, ROOT))
        pts.append((0.0 if name == "spin" else r["gbs0"], r))
    return {c: ([x for x, _ in pts], [r[c] for _, r in pts]) for c in COORDS}


def l_eq(reading, rul, c):
    """The equivalent stress of a reading, and whether it lies on the ruler (else clipped)."""
    xs, ys = rul[c]
    return S.invert(reading, xs, ys)


def shown(v):
    """An L_eq for the tables, marked ! when clipped to the ruler's end."""
    return "%6.1f%s" % (v[0], "" if v[1] else "!")


def table():
    rul = ruler()
    alone = {d: config("s4-D" + d) for d in DUTIES}
    print("ruler (8 SMs): GB/s, reporter p99 / median ns")
    for x, y99, y50 in zip(rul["p99"][0], rul["p99"][1], rul["median"][1]):
        print("  %6.1f  %7.1f  %7.1f" % (x, y99, y50))
    zero = [config(n) for n in ("spin4", "spin4+spin4")]
    print("\nzero (p99 ns): spin4 %s, spin4+spin4 %s" % tuple("%.1f" % z["p99"] if z else "-" for z in zero))

    print("\neach SAXPY alone on 4 SMs: GB/s, then per coordinate reading ns and L_eq GB/s")
    for d, r in alone.items():
        print("  D%s  %6.1f" % (d, r["gbs0"]) + "".join(
            "   %s %7.1f -> %s" % (c, r[c], shown(l_eq(r[c], rul, c))) for c in COORDS))

    rows = []
    print("\ncombinations: GB/s moved (each, sum / sum alone), then per coordinate "
          "L_eq(a+b), L_eq(a)+L_eq(b), ratio")
    for a in DUTIES:
        for b in DUTIES:
            r = config("s4-D%s+s4-D%s" % (a, b))
            if r is None:
                continue
            out = dict(a=a, b=b, gbs_a=r["gbs0"], gbs_b=r["gbs1"])
            out["traffic_ratio"] = (r["gbs0"] + r["gbs1"]) / (alone[a]["gbs0"] + alone[b]["gbs0"])
            line = "  D%s+D%s  %5.1f %5.1f  %.2f" % (a, b, r["gbs0"], r["gbs1"], out["traffic_ratio"])
            for c in COORDS:
                both, ok = l_eq(r[c], rul, c)
                parts = l_eq(alone[a][c], rul, c)[0] + l_eq(alone[b][c], rul, c)[0]
                out[c + "_leq"], out[c + "_sum"], out[c + "_inside"] = both, parts, ok
                out[c + "_ratio"] = both / parts if parts > 0 else float("nan")
                out[c + "_spread_ns"] = r[c + "_spread"]
                line += "   %s %s %6.1f %.2f" % (c, shown((both, ok)), parts, out[c + "_ratio"])
            rows.append(out)
            print(line)
    print("  (! = beyond the ruler, clipped to its end)")

    print("\nthe two partitions: L_eq p99 of (a, b) against (b, a)")
    by = {(r["a"], r["b"]): r for r in rows}
    for a in DUTIES:
        for b in DUTIES:
            if a < b and (a, b) in by and (b, a) in by:
                print("  D%s+D%s %6.1f   D%s+D%s %6.1f" % (a, b, by[a, b]["p99_leq"], b, a, by[b, a]["p99_leq"]))

    path = os.path.join(S.OUT, "compose-4sm.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % path)


def selftest():
    # A straight ruler, 600 ns at 0 GB/s plus 2 ns per GB/s: readings made additive must give 1.
    rul = {c: ([0.0, 25, 50, 100, 135], [600.0, 650, 700, 800, 870]) for c in COORDS}
    one = 600 + 2 * 20.0
    two = 600 + 2 * 40.0
    x1, ok1 = l_eq(one, rul, "p99")
    x2, ok2 = l_eq(two, rul, "p99")
    assert ok1 and ok2 and abs(x2 / (2 * x1) - 1) < 1e-9
    assert not l_eq(900.0, rul, "p99")[1]           # above the top: flagged, not extrapolated
    log = os.path.join("/tmp", "compose-selftest.log")
    with open(log, "w") as f:
        f.write("  bytes moved:  1.00 GB (10.0 GB/s)\n  bytes moved:  9.00 GB (55.5 GB/s)\n")
    assert gbs(log) == 55.5
    os.remove(log)
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["selftest"] else table()
