#!/usr/bin/env python3
"""onoff-latency.py - the target's latency next to ON/OFF and uniform SAXPY (family stress-onoff).

    python3 analysis/explore/onoff-latency.py [<bench dir>]      default /home/fscarfato/data/bench/stress-onoff

Daniel's readout (2026-10-10): median, p95, p99 and maximum of the target's per-inference gpu_ms,
per configuration, and for ON/OFF the same split into inferences run during ON, during OFF, and
across a transition.

How an inference is placed. The runner stamps every inference on CLOCK_MONOTONIC (t_start_ns before
the input fetch, t_end_ns when the host sees it complete); corunner logs every ON span on the same
clock ("phase: on <start> <end>", from the first launch to the last synchronize of the phase). An
inference lies in ON if its whole [t_start, t_end] is inside one ON span, in OFF if it overlaps none,
and across a transition otherwise. The uniform configurations (periods of ~3 ms) are not split: an
inference of any network overlaps several of their periods.

Statistics. Per run: the percentiles of that run's timed inferences. Per configuration: the mean over
the runs and the range across them (the run spread). The split pools the runs, since a class can hold
few inferences per run (RetinaNet's ON at 25 %). The p99's 95 % bootstrap interval (1000 resamples of
each run, then the mean over runs) shows whether the run is long enough for a stable p99.
"""
import bisect
import csv
import glob
import os
import re
import sys

import numpy as np

BENCH = sys.argv[1] if len(sys.argv) > 1 else "/home/fscarfato/data/bench/stress-onoff"
QS = (50, 95, 99, 100)                       # median, p95, p99, max
rng = np.random.default_rng(1)


def inferences(path):
    """The timed inferences of one run: arrays of gpu_ms, t_start_ns, t_end_ns."""
    g, a, b = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["phase"] == "bench":
                g.append(float(r["gpu_ms"]))
                a.append(int(r["t_start_ns"]))
                b.append(int(r["t_end_ns"]))
    return np.array(g), np.array(a), np.array(b)


def on_spans(path):
    """The ON spans corunner logged, sorted, as two lists (starts, ends)."""
    s, e = [], []
    for m in re.finditer(r"phase:\s+on (\d+) (\d+)", open(path, errors="replace").read()):
        s.append(int(m.group(1)))
        e.append(int(m.group(2)))
    return s, e


def place(a, b, starts, ends):
    """'on', 'off' or 'cross' for the inference [a, b] against sorted, disjoint ON spans."""
    i = bisect.bisect_right(starts, a) - 1   # the last span starting at or before a
    if i >= 0 and b <= ends[i]:
        return "on"
    j = bisect.bisect_left(starts, b)        # spans starting before b
    overlaps = (i >= 0 and a < ends[i]) or (j > i + 1)
    return "cross" if overlaps else "off"


def pct(v):
    return [float(np.percentile(v, q)) for q in QS]


def p99_ci(v):
    """Half-width of the 95 % bootstrap interval of the p99, relative to the p99."""
    boots = [np.percentile(rng.choice(v, v.size), 99) for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return (hi - lo) / 2 / np.percentile(v, 99)


def main():
    configs = sorted(d for d in os.listdir(BENCH) if os.path.isdir(os.path.join(BENCH, d)) and d[0] != "_")
    alone = {}
    rows = []
    for c in configs:
        net = c.split("-")[0]
        runs = sorted(glob.glob(os.path.join(BENCH, c, "r*_latencies.csv")))
        per_run, cis, split = [], [], {"on": [], "off": [], "cross": []}
        for path in runs:
            g, a, b = inferences(path)
            per_run.append(pct(g))
            cis.append(p99_ci(g))
            itf = path.replace("_latencies.csv", "_itf.log")
            if "-onoff-" in c and os.path.exists(itf):
                starts, ends = on_spans(itf)
                for x, s, e in zip(g, a, b):
                    split[place(s, e, starts, ends)].append(x)
        m = np.mean(per_run, axis=0)
        spread = np.ptp([r[2] for r in per_run])
        if c.endswith("-alone"):
            alone[net] = m
        rows.append((c, net, len(runs), m, spread, max(cis), split))

    print("%-28s %4s %9s %9s %9s %9s %8s %8s %7s" % ("config", "runs", "median", "p95", "p99", "max",
                                                    "p99 +%", "spread", "CI p99"))
    for c, net, n, m, spread, ci, split in rows:
        ref = alone.get(net)
        up = 100 * (m[2] / ref[2] - 1) if ref is not None else float("nan")
        print("%-28s %4d %9.4f %9.4f %9.4f %9.4f %+8.1f %7.1fpt %6.2f%%" %
              (c, n, m[0], m[1], m[2], m[3], up, 100 * spread / ref[2] if ref is not None else 0, 100 * ci))
        if any(split.values()):
            total = sum(len(v) for v in split.values())
            for k in ("on", "off", "cross"):
                v = np.array(split[k])
                if v.size:
                    q = pct(v)
                    print("    %-6s n=%-7d (%4.1f%%) median %.4f  p95 %.4f  p99 %.4f (%+.1f%%)  max %.4f" %
                          (k, v.size, 100 * v.size / total, q[0], q[1], q[2],
                           100 * (q[2] / ref[2] - 1) if ref is not None else 0, q[3]))


if __name__ == "__main__":
    # Self-check of the placement rule on two ON spans [10, 20] and [30, 40].
    S, E = [10, 30], [20, 40]
    assert place(12, 18, S, E) == "on" and place(21, 29, S, E) == "off"
    assert place(18, 22, S, E) == "cross" and place(5, 12, S, E) == "cross"
    assert place(15, 35, S, E) == "cross" and place(1, 9, S, E) == "off" and place(41, 50, S, E) == "off"
    main()
