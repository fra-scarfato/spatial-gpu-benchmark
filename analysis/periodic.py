#!/usr/bin/env python3
"""periodic.py - the networks released like cameras instead of back to back (the professor, 2026-11-10).

    python3 analysis/periodic.py <family>         per configuration and load level: each network's p99 GPU
                                                  time, response and wait, the miss share, the energy and
                                                  the GPU clock of the level; the highest level all the
                                                  networks sustain, per gamma
    python3 analysis/periodic.py check <prefix>   the structure of one run (the queues' smoke step)
    python3 analysis/periodic.py selftest

A periodic run (runner --levels; the schedule is Periodic in include/config.hpp) has one process per
network, each stepping through the same ascending load levels: a window of release ticks, then a 1 s gap.
Every job has a release, a start and an end on CLOCK_MONOTONIC and a GPU time on its stream (gpu_ms):

    response = end - release     what a camera frame waits for its result
    wait     = start - release   the time in the network's FIFO, behind its own previous job
    GPU time = gpu_ms            the inference itself, slowed by the other networks

No job is ever dropped, so the deadline is applied here, at any value. Each network's budget is
D = gamma x its p99 alone on the whole GPU, back to back (the alone runs of energy-tenants), the same at
every level. A level passes when, for every network, the median over the runs of the p99 response is
within D; the highest level a configuration sustains is the last one before the first that fails (a
level a network never reached, because it stopped on overload, fails). The miss share (jobs over D) and
the stretching it implies are the metrics of Zahaf et al., RTNS 2026, Section 8.2. The energy of a level
is the rails integrated over its window, from the first release to the last end of any network: every
arm of a family releases the same jobs in the same windows, so the energies are at equal work.
"""
import csv
import glob
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geometry as G

GAMMAS = (2, 3, 5)      # the budgets shown; the thesis's gamma is the user's and the professor's choice


def periodic_jobs(path):
    """{level Hz: [(release, start, end, gpu_ms)]} of one latency CSV written under --levels, its "bench"
    rows, times in seconds of CLOCK_MONOTONIC; {} for a closed-loop file (level_hz 0 or absent)."""
    out = {}
    for r in csv.DictReader(open(path)):
        hz = float(r.get("level_hz") or 0)
        if r["phase"] == "bench" and hz > 0:
            out.setdefault(hz, []).append((int(r["t_release_ns"]) / 1e9, int(r["t_start_ns"]) / 1e9,
                                           int(r["t_end_ns"]) / 1e9, float(r["gpu_ms"])))
    return out


def run_networks(prefix):
    """{network: periodic_jobs} of one run: <prefix>_<network>_latencies.csv, or <prefix>_latencies.csv
    named after its configuration (a solo run)."""
    run, cfg = os.path.basename(prefix), os.path.basename(os.path.dirname(prefix))
    return {os.path.basename(f)[len(run) + 1:-len("_latencies.csv")] or cfg.split("-")[0]: periodic_jobs(f)
            for f in sorted(glob.glob(prefix + "_*latencies.csv"))}


def check(prefix):
    """The structure of one periodic run: per network and level, releases evenly spaced (to the 1 ns of
    rounding), no job starting before its release or ending before it starts, and each level's first
    release at least the 1 s gap after the previous level's last. Prints the jobs and the p99 wait per
    level; AssertionError on a violation."""
    nets = run_networks(prefix)
    assert nets and all(nets.values()), "no periodic rows for " + prefix
    for net, jobs in nets.items():
        last = None
        for hz in sorted(jobs):
            rel = np.array([j[0] for j in jobs[hz]])
            if len(rel) > 1:
                gaps = np.diff(rel)
                assert gaps.max() - gaps.min() < 3e-9, "%s %g Hz: uneven releases" % (net, hz)
            assert all(r <= s < e for r, s, e, g in jobs[hz]), "%s %g Hz: a job before its release" % (net, hz)
            assert last is None or rel[0] - last >= 1.0, "%s %g Hz: no gap before the level" % (net, hz)
            last = rel[-1]
            wait = [1e3 * (s - r) for r, s, e, g in jobs[hz]]
            print("  %-10s %7.2f Hz  %5d jobs  p99 wait %.3f ms" % (net, hz, len(jobs[hz]), np.percentile(wait, 99)))


def clock_log(path):
    """[(uptime_s, GPU Hz)] from a _power.log that records the clock (9 columns, since 2026-11-10)."""
    out = []
    for line in open(path, errors="replace"):
        f = line.split()
        if len(f) == 9 and f[0][0].isdigit():
            out.append((float(f[0]), float(f[7])))
    return out


def level_rows(fam):
    """One row per configuration, run, level and network of a periodic family (bench/<fam>/<cfg>/r*)."""
    c99 = {n: v[2] for n, v in G.tenants_alone("energy-tenants").items()}
    d = os.path.join(G.P.BENCH, fam)
    if not os.path.isdir(d):
        sys.exit("no %s runs in %s" % (fam, d))
    rows = []
    for cfg in sorted(os.listdir(d)):
        # The periodic families always run with POWER=1, so each run is the prefix of its power log.
        prefixes = sorted(f[:-len("_power.log")] for f in glob.glob(os.path.join(d, cfg, "r*_power.log")))
        for pre in prefixes:
            nets = run_networks(pre)
            if not nets or not all(nets.values()):
                continue
            power = G.power_log(pre + "_power.log")
            clock = clock_log(pre + "_power.log")
            for hz in sorted({h for jobs in nets.values() for h in jobs}):
                t0 = min(j[0] for jobs in nets.values() for j in jobs.get(hz, []))
                t1 = max(j[2] for jobs in nets.values() for j in jobs.get(hz, []))
                clk = [c for t, c in clock if t0 <= t <= t1]
                for net, jobs in nets.items():
                    js = jobs.get(hz)
                    if not js:
                        continue
                    resp = np.array([1e3 * (e - r) for r, s, e, g in js])
                    row = dict(family=fam, config=cfg, run=os.path.basename(pre), level_hz=hz, network=net,
                               jobs=len(js), gpu_p99_ms=round(float(np.percentile([g for r, s, e, g in js], 99)), 4),
                               resp_p99_ms=round(float(np.percentile(resp, 99)), 4),
                               wait_p99_ms=round(float(np.percentile([1e3 * (s - r) for r, s, e, g in js], 99)), 4),
                               level_s=round(t1 - t0, 3), level_J=round(G.rails_energy(power, t0, t1), 2),
                               clk_min_MHz=round(min(clk) / 1e6, 1) if clk else "",
                               clk_max_MHz=round(max(clk) / 1e6, 1) if clk else "")
                    for g in GAMMAS:
                        row["miss_g%d" % g] = round(float(np.mean(resp > g * c99[net])), 5)
                    rows.append(row)
    return rows, c99


def cmd_family(fam):
    rows, c99 = level_rows(fam)
    G.write("periodic-%s.csv" % fam, rows)
    print("budgets: D = gamma x p99 alone on 16 SMs, back to back: " +
          "  ".join("%s %.2f ms" % (n, v) for n, v in sorted(c99.items())))
    by = {}
    for r in rows:
        by.setdefault(r["config"], {}).setdefault(r["level_hz"], {}).setdefault(r["network"], []).append(r)
    for cfg, levels in by.items():
        nets = sorted({n for lv in levels.values() for n in lv})
        runs = len({r["run"] for r in rows if r["config"] == cfg})
        print("\n%s (%d runs): per network p99 GPU / response / wait (ms), medians of the runs; level energy, clock" % (cfg, runs))
        ok_until = {g: None for g in GAMMAS}
        failed = set()
        for hz in sorted(levels):
            lv = levels[hz]
            med = lambda n, k: st.median(r[k] for r in lv[n])
            cells = "  ".join("%s %.2f/%.2f/%.2f" % (n, med(n, "gpu_p99_ms"), med(n, "resp_p99_ms"), med(n, "wait_p99_ms"))
                              if n in lv else "%s -" % n for n in nets)
            every = [r for n in lv for r in lv[n]]
            energy = st.median(r["level_J"] for r in lv[nets[0]]) if nets[0] in lv else float("nan")
            clk = [r for r in every if r["clk_min_MHz"] != ""]
            clock = "%.1f-%.1f MHz" % (min(r["clk_min_MHz"] for r in clk), max(r["clk_max_MHz"] for r in clk)) if clk else "clock ?"
            complete = all(n in lv and len(lv[n]) == runs for n in nets)
            for g in GAMMAS:
                ok = complete and all(med(n, "resp_p99_ms") <= g * c99[n] for n in nets)
                if ok and g not in failed:
                    ok_until[g] = hz
                elif not ok:
                    failed.add(g)
            print("  %7.2f Hz  %s  | %6.0f J  %s" % (hz, cells, energy, clock))
        print("  highest level sustained: " + "  ".join(
            "gamma %d: %s" % (g, "%g Hz" % ok_until[g] if ok_until[g] else "none") for g in GAMMAS))


def selftest():
    import tempfile
    hdr = "phase,index,image,prep_ms,wall_ms,gpu_ms,t_start_ns,t_end_ns,level_hz,t_release_ns\n"
    with tempfile.TemporaryDirectory() as d:
        # two levels, 10 Hz then 20 Hz, 3 jobs each, the second window 1.2 s after the first one's end
        rows = ["warmup,0,x,0,1,1,1000,2000,0,1000"]
        for i, (hz, rel) in enumerate([(10, 100.0), (10, 100.1), (10, 100.2), (20, 101.5), (20, 101.55), (20, 101.6)]):
            r = round(rel * 1e9)
            rows.append("bench,%d,x,0,1,%g,%d,%d,%g,%d" % (i, 1 + i, r + 50000, r + 2000000, hz, r))
        p = os.path.join(d, "cfg", "r1")
        os.makedirs(os.path.dirname(p))
        open(p + "_a_latencies.csv", "w").write(hdr + "\n".join(rows) + "\n")
        nets = run_networks(p)
        assert sorted(nets["a"]) == [10.0, 20.0] and len(nets["a"][20.0]) == 3
        assert abs(nets["a"][10.0][1][0] - 100.1) < 1e-9
        check(p)
        # an uneven release must fail the check
        open(p + "_a_latencies.csv", "w").write(hdr + "\n".join(rows[:2] + [rows[2].replace("100100000000", "100120000000")] + rows[3:]) + "\n")
        try:
            check(p)
            raise SystemExit("check missed an uneven release")
        except AssertionError:
            pass
        open(os.path.join(d, "r1_power.log"), "w").write(
            "uptime_s a b c d e f gpu_hz gpu_mC\n100.0 5000 2000 0 0 0 0 1300500000 45000\n100.2 5000 2000 0 0 0 0 816000000 45000\n")
        assert clock_log(os.path.join(d, "r1_power.log")) == [(100.0, 1300500000.0), (100.2, 816000000.0)]
        assert len(G.power_log(os.path.join(d, "r1_power.log"))) == 2      # nine columns still read
    print("  selftest passed")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
    elif len(sys.argv) == 3 and sys.argv[1] == "check":
        check(sys.argv[2])
        print("  check passed")
    elif len(sys.argv) == 2:
        cmd_family(sys.argv[1])
    else:
        sys.exit(__doc__)
