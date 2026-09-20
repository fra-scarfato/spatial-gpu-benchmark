#!/usr/bin/env python3
"""aggregate.py — turn the per-run statistics into the numbers the thesis needs.

THE UNIT OF REPLICATION IS THE RUN, NOT THE INFERENCE. That single decision
drives everything below:

  level 1 (within a run)  the runner has already summarised the run: its
                          <name>_stats.csv holds median/mean/p99/std over the
                          ~10k-60k inferences of that process. We read that.
  level 2 (across runs)   the point estimate is the MEDIAN OF THE 5 RUN-LEVEL
                          VALUES, and the run-to-run spread is reported as
                          [min, max].

The tempting alternative -- pool every raw inference from the 5 runs into one
vector and take its p99 -- is the classic pseudo-replication error: it treats
~300k correlated observations from 5 processes as independent, and therefore
understates the uncertainty several-fold. The 5 fresh processes exist precisely
to provide a between-run error bar, so the aggregation must expose it.

Outputs (into --out):
  raw.csv       tidy/long: one row per (experiment, config, network, run, metric)
  summary.csv   across-run point estimate and spread per config/network
  tables.txt    the four derived tables, also printed to stdout
"""

import argparse
import csv
import glob
import os
import re
import statistics as stats
import sys

# exp1_mobilenet_gc50_r3_stats.csv
# exp2_50_50_r3_mobilenet_stats.csv
NAME_RE = re.compile(
    r'^(?P<exp>exp\d+)_(?P<config>.+?)_r(?P<run>\d+)'
    r'(?:_(?P<net>mobilenet|resnet))?_stats\.csv$')

METRICS = ("gpu_ms", "wall_ms", "prep_ms")
# Column order written by Network::writeStats.
COLS = ("metric", "n", "mean", "median", "std", "p95", "p99", "max", "min",
        "p99_minus_median", "count", "duration_s")

# The SM counts the configs asked for, as a percentage of the 16-SM device.
# Percentages are what the protocol specifies and what was granted and verified
# on this board (16 SMs, alignment 2: 25->4, 50->8, 75->12).
EXP1_PCT = {"whole": 100, "gc100": 100, "gc75": 75, "gc50": 50, "gc25": 25}
EXP2_PCT = {"baseline": {"mobilenet": 100, "resnet": 100},
            "50_50":    {"mobilenet": 50,  "resnet": 50},
            "75_25":    {"mobilenet": 75,  "resnet": 25},
            "25_75":    {"mobilenet": 25,  "resnet": 75}}
TOTAL_SMS = 16


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def read_stats(path):
    """-> {metric: {col: float}}"""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = {c: float(row[c]) for c in COLS[1:]}
    return out


def load(outdir):
    """-> data[(exp, config, run)][network][metric] = {col: value}"""
    data = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*_stats.csv"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        exp, config, run = m.group("exp"), m.group("config"), int(m.group("run"))
        net = m.group("net")
        # exp1 encodes the network in the config name: "mobilenet_gc50".
        # exp3's single-process control has no label either (the runner only
        # adds one when there are several networks in the run), and it is the
        # MobileNet client.
        if net is None:
            if exp == "exp1":
                net = config.split("_")[0]
            elif config == "mps_single":
                net = "mobilenet"
            else:
                raise ValueError(f"cannot determine the network for {path!r}")
            assert net in ("mobilenet", "resnet"), f"unexpected config {config!r}"
        data.setdefault((exp, config, run), {})[net] = read_stats(path)
    return data


def aggr(data, exp, config, net, metric, col):
    """Across-run aggregate of one statistic -> (median, min, max, n_runs)."""
    vals = [data[(exp, config, r)][net][metric][col]
            for r in sorted(runs_of(data, exp, config))
            if (exp, config, r) in data and net in data[(exp, config, r)]]
    if not vals:
        return None
    return stats.median(vals), min(vals), max(vals), len(vals)


def runs_of(data, exp, config):
    return {r for (e, c, r) in data if e == exp and c == config}


def exp1_configs(net):
    return [f"{net}_{k}" for k in ("whole", "gc100", "gc75", "gc50", "gc25")]


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------
def hdr(title):
    return f"\n=== {title} " + "=" * max(0, 74 - len(title)) + "\n"


def table_exp1(data, lines):
    lines.append(hdr("EXPERIMENT 1 - isolated: latency vs SM share"))
    lines.append("  Latency is the median of the 5 run medians; [min,max] is the\n"
                 "  run-to-run spread. Ratio is against the same network's `whole` run.\n")
    lines.append(f"  {'network':<10} {'config':<9} {'SMs':>4} "
                 f"{'median ms':>10} {'[min, max]':>18} {'ratio':>7} {'p99 ms':>8}")
    for net in ("mobilenet", "resnet"):
        base = aggr(data, "exp1", f"{net}_whole", net, "gpu_ms", "median")
        for cfg in exp1_configs(net):
            a = aggr(data, "exp1", cfg, net, "gpu_ms", "median")
            p99 = aggr(data, "exp1", cfg, net, "gpu_ms", "p99")
            if a is None:
                continue
            med, lo, hi, n = a
            ratio = f"{med / base[0]:.3f}" if base else "-"
            sms = round(TOTAL_SMS * EXP1_PCT[cfg.split('_', 1)[1]] / 100)
            lines.append(f"  {net:<10} {cfg.split('_',1)[1]:<9} {sms:>4} "
                         f"{med:>10.4f} {'[' + format(lo, '.4f') + ',' + format(hi, '.4f') + ']':>18} "
                         f"{ratio:>7} {p99[0]:>8.4f}")
    return lines


def table_exp2(data, lines):
    lines.append(hdr("EXPERIMENT 2 - concurrent (MobileNet + ResNet), disjoint partitions"))
    lines.append("  Aggregate throughput = (count_mob + count_res) / max(duration) per run,\n"
                 "  median over the 5 runs.\n")
    lines.append(f"  {'split':<10} {'network':<10} {'SMs':>4} {'median ms':>10} "
                 f"{'p99 ms':>8} {'std ms':>8} {'tail ms':>8} {'agg inf/s':>10}")
    order = ["baseline", "50_50", "75_25", "25_75"]
    for cfg in order:
        if not any((e, c) == ("exp2", cfg) for (e, c, _) in data):
            continue
        thr = []
        for r in sorted(runs_of(data, "exp2", cfg)):
            rec = data.get(("exp2", cfg, r))
            if not rec or len(rec) < 2:
                continue
            cnt = sum(rec[n]["gpu_ms"]["count"] for n in rec)
            dur = max(rec[n]["gpu_ms"]["duration_s"] for n in rec)
            thr.append(cnt / dur)
        agg_thr = f"{stats.median(thr):.0f}" if thr else "-"
        for net in ("mobilenet", "resnet"):
            a = aggr(data, "exp2", cfg, net, "gpu_ms", "median")
            if a is None:
                continue
            med, lo, hi, n = a
            p99 = aggr(data, "exp2", cfg, net, "gpu_ms", "p99")[0]
            sd = aggr(data, "exp2", cfg, net, "gpu_ms", "std")[0]
            tl = aggr(data, "exp2", cfg, net, "gpu_ms", "p99_minus_median")[0]
            sms = round(TOTAL_SMS * EXP2_PCT[cfg][net] / 100)
            lines.append(f"  {cfg:<10} {net:<10} {sms:>4} {med:>10.4f} "
                         f"{p99:>8.4f} {sd:>8.4f} {tl:>8.4f} {agg_thr:>10}")
        lines.append("")
    return lines


def table_residual(data, lines):
    lines.append(hdr("RESIDUAL INTERFERENCE - concurrent / isolated, same SM count"))
    lines.append("  Isolated = Experiment 1 at the same SM count. This is the number\n"
                 "  SM partitioning could NOT remove: memory bandwidth, L2, work queues.\n")
    lines.append(f"  {'network':<10} {'split':<10} {'SMs':>4} "
                 f"{'isolated':>10} {'concurrent':>11} {'overhead':>9}")
    for cfg in ["baseline", "50_50", "75_25", "25_75"]:
        for net in ("mobilenet", "resnet"):
            conc = aggr(data, "exp2", cfg, net, "gpu_ms", "median")
            if conc is None:
                continue
            # baseline shares the whole GPU unpartitioned -> compare to `whole`
            ref_cfg = f"{net}_whole" if cfg == "baseline" else \
                      f"{net}_gc{EXP2_PCT[cfg][net]}"
            iso = aggr(data, "exp1", ref_cfg, net, "gpu_ms", "median")
            if iso is None:
                continue
            sms = round(TOTAL_SMS * EXP2_PCT[cfg][net] / 100)
            lines.append(f"  {net:<10} {cfg:<10} {sms:>4} {iso[0]:>10.4f} "
                         f"{conc[0]:>11.4f} {100 * (conc[0] / iso[0] - 1):>8.1f}%")
        lines.append("")
    return lines


def table_jitter(data, lines):
    lines.append(hdr("JITTER - the tail is what a deadline sees"))
    lines.append(f"  {'network':<10} {'config':<12} {'std ms':>9} "
                 f"{'p99-med ms':>11} {'max ms':>9}")
    for net in ("mobilenet", "resnet"):
        for cfg in exp1_configs(net):
            sd = aggr(data, "exp1", cfg, net, "gpu_ms", "std")
            if sd is None:
                continue
            lines.append(f"  {net:<10} {'iso ' + cfg.split('_',1)[1]:<12} "
                         f"{sd[0]:>9.4f} "
                         f"{aggr(data,'exp1',cfg,net,'gpu_ms','p99_minus_median')[0]:>11.4f} "
                         f"{aggr(data,'exp1',cfg,net,'gpu_ms','max')[0]:>9.4f}")
    for cfg in ["baseline", "50_50", "75_25", "25_75"]:
        for net in ("mobilenet", "resnet"):
            sd = aggr(data, "exp2", cfg, net, "gpu_ms", "std")
            if sd is None:
                continue
            lines.append(f"  {net:<10} {'conc ' + cfg:<12} {sd[0]:>9.4f} "
                         f"{aggr(data,'exp2',cfg,net,'gpu_ms','p99_minus_median')[0]:>11.4f} "
                         f"{aggr(data,'exp2',cfg,net,'gpu_ms','max')[0]:>9.4f}")
    return lines


# ---------------------------------------------------------------------
# Experiment 3 — MPS, two processes
#
# Note the difference from Experiments 1-2: there, one runner process owns
# both networks, so a single "run" yields a matched pair of stats files and the
# two windows are identical by construction. Here the two networks are two
# INDEPENDENT processes with no shared clock, so their windows are aligned only
# to ~0.5% (the difference in warm-up length). Everything below therefore
# aggregates the pair as one record, but the caveat is stamped on the table.
# ---------------------------------------------------------------------
EXP3_ORDER = ["mps_single", "mps_off", "mps_default",
              "mps_atp_50_50", "mps_atp_75_25", "mps_atp_25_75",
              "mps_static_50_50", "mps_static_75_25", "mps_static_25_75"]


def exp3_configs(data):
    present = {c for (e, c, _) in data if e == "exp3"}
    return [c for c in EXP3_ORDER if c in present] + sorted(present - set(EXP3_ORDER))


def _run_throughput(data, exp, config, run):
    """Total inferences over the shared window, for one run."""
    rec = data.get((exp, config, run))
    if not rec:
        return None
    cnt = sum(v["gpu_ms"]["count"] for v in rec.values())
    dur = max(v["gpu_ms"]["duration_s"] for v in rec.values())
    return cnt / dur if dur else None


def table_exp3(data, lines):
    cfgs = exp3_configs(data)
    if not cfgs:
        return lines
    lines.append(hdr("EXPERIMENT 3 - two processes, MPS"))
    lines.append(
        "  Two independent processes (MobileNet + ResNet), no green contexts.\n"
        "  Aggregate throughput = total inferences / the longer of the two windows.\n"
        "  CAVEAT: separate processes share no clock, so their timed windows are\n"
        "  aligned only to ~0.5% (the difference in warm-up length).\n")
    lines.append(f"  {'config':<20} {'network':<10} {'median ms':>10} "
                 f"{'p99 ms':>8} {'std ms':>8} {'agg inf/s':>10}")
    for cfg in cfgs:
        thr = [t for t in (_run_throughput(data, "exp3", cfg, r)
                           for r in sorted(runs_of(data, "exp3", cfg))) if t]
        agg = f"{stats.median(thr):.0f}" if thr else "-"
        for net in ("mobilenet", "resnet"):
            a = aggr(data, "exp3", cfg, net, "gpu_ms", "median")
            if a is None:
                continue
            lines.append(
                f"  {cfg:<20} {net:<10} {a[0]:>10.4f} "
                f"{aggr(data,'exp3',cfg,net,'gpu_ms','p99')[0]:>8.4f} "
                f"{aggr(data,'exp3',cfg,net,'gpu_ms','std')[0]:>8.4f} {agg:>10}")
        lines.append("")
    return lines


def table_mps_effects(data, lines):
    """The comparisons that need BOTH the MPS run and a reference.

    These are the rows that make Experiment 3 interpretable:
      - MPS per-client cost: does MPS itself slow a lone client down? Without
        this, `mps_default` vs `mps_off` cannot be attributed to co-scheduling.
      - Process-model cost: two processes vs one, with no MPS on either side.
        This is the measured bound on the GC-vs-MPS confound.
      - Co-scheduling: what MPS is actually for.
    """
    def med(exp, cfg, net):
        a = aggr(data, exp, cfg, net, "gpu_ms", "median")
        return a[0] if a else None

    checks = [
        ("MPS per-client cost", "mps_single vs isolated (both 1 process)",
         ("exp3", "mps_single", "mobilenet"), ("exp1", "mobilenet_whole", "mobilenet")),
        ("Process-model cost", "2 processes vs 1 process, no MPS",
         ("exp3", "mps_off", "mobilenet"), ("exp2", "baseline", "mobilenet")),
        ("Process-model cost", "2 processes vs 1 process, no MPS",
         ("exp3", "mps_off", "resnet"), ("exp2", "baseline", "resnet")),
        ("MPS co-scheduling", "2 processes, MPS vs no MPS",
         ("exp3", "mps_default", "mobilenet"), ("exp3", "mps_off", "mobilenet")),
        ("MPS co-scheduling", "2 processes, MPS vs no MPS",
         ("exp3", "mps_default", "resnet"), ("exp3", "mps_off", "resnet")),
    ]
    for cfg in ("mps_atp_50_50", "mps_atp_75_25", "mps_atp_25_75"):
        split = cfg[len("mps_atp_"):]
        for net in ("mobilenet", "resnet"):
            checks.append((f"ATP {split}", f"{net}: vs mps_default (no ATP)",
                           ("exp3", cfg, net), ("exp3", "mps_default", net)))
    for cfg in ("mps_static_50_50", "mps_static_75_25", "mps_static_25_75"):
        split = cfg[len("mps_static_"):]
        for net in ("mobilenet", "resnet"):
            checks.append((f"static {split}", f"{net}: vs the 1-process green context, same ratio",
                           ("exp3", cfg, net), ("exp2", split, net)))

    rows = []
    for label, desc, (e1, c1, n1), (e2, c2, n2) in checks:
        a, b = med(e1, c1, n1), med(e2, c2, n2)
        if a is None or b is None:
            continue
        rows.append((label, desc, b, a, 100.0 * (a / b - 1.0)))
    if not rows:
        return lines

    lines.append(hdr("MPS EFFECTS - every row pairs a control with a measurement"))
    lines.append(
        "  `reference` is the left-hand configuration, `measured` the right-hand\n"
        "  one; delta = measured/reference - 1, so NEGATIVE means the change made\n"
        "  things faster.\n")
    lines.append(f"  {'effect':<21} {'what is being compared':<48} "
                 f"{'ref ms':>8} {'meas ms':>9} {'delta':>8}")
    for label, desc, ref, meas, delta in rows:
        lines.append(f"  {label:<21} {desc:<48} {ref:>8.4f} {meas:>9.4f} {delta:>7.1f}%")
    lines.append("")
    lines.append("  Reading of the static rows: a green context is INTRA-process and a\n"
                 "  static MPS partition is INTER-process, so those rows carry a\n"
                 "  process-model difference as well as a mechanism difference.\n"
                 "  The `Process-model cost` rows bound that difference.")
    return lines


# ---------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------
def write_raw(data, outdir):
    path = os.path.join(outdir, "raw.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "config", "network", "run", "metric"] + list(COLS[1:]))
        for (exp, config, run) in sorted(data):
            for net in sorted(data[(exp, config, run)]):
                for metric in METRICS:
                    rec = data[(exp, config, run)][net].get(metric)
                    if rec is None:
                        continue
                    w.writerow([exp, config, net, run, metric] + [rec[c] for c in COLS[1:]])
    return path


def write_summary(data, outdir):
    path = os.path.join(outdir, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "config", "network", "n_runs",
                    "median__median", "median__min", "median__max",
                    "p99__median", "std__median", "tail__median",
                    "throughput_inf_s"])
        for (exp, config) in sorted({(e, c) for (e, c, _) in data}):
            nets = sorted({n for (e, c, r) in data if (e, c) == (exp, config)
                           for n in data[(e, c, r)]})
            for net in nets:
                med = aggr(data, exp, config, net, "gpu_ms", "median")
                if med is None:
                    continue
                thr = aggr(data, exp, config, net, "gpu_ms", "count")
                dur = aggr(data, exp, config, net, "gpu_ms", "duration_s")
                tput = (thr[0] / dur[0]) if thr and dur else 0.0
                w.writerow([exp, config, net, med[3],
                            f"{med[0]:.6f}", f"{med[1]:.6f}", f"{med[2]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','p99')[0]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','std')[0]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','p99_minus_median')[0]:.6f}",
                            f"{tput:.2f}"])
    return path


def check_consistency(data, lines):
    """Flag runs whose median is a clear outlier among their siblings.

    This is the after-the-fact safety net: a contaminated run usually shows up
    as one run several times slower than the other four.
    """
    flagged = []
    for (exp, config) in {(e, c) for (e, c, _) in data}:
        for net in {n for (e, c, r) in data if (e, c) == (exp, config)
                    for n in data[(e, c, r)]}:
            vals = {r: data[(exp, config, r)][net]["gpu_ms"]["median"]
                    for r in runs_of(data, exp, config)
                    if (exp, config, r) in data and net in data[(exp, config, r)]}
            if len(vals) < 3:
                continue
            m = stats.median(vals.values())
            for r, v in sorted(vals.items()):
                if v > 1.25 * m:
                    flagged.append((exp, config, net, r, v, m))
    if flagged:
        lines.append(hdr("WARNING - SUSPECT RUNS (median > 25% above the family median)"))
        lines.append("  These were very likely taken while something else used the GPU.\n"
                     "  Re-run them (delete the *_stats.csv and re-run ./sweep.sh).\n")
        for exp, config, net, r, v, m in flagged:
            lines.append(f"  {exp} {config} {net} run {r}: {v:.4f} ms "
                         f"vs family median {m:.4f} ms  ({100*(v/m-1):+.0f}%)")
    return lines


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "bench"),
        help="directory holding the *_stats.csv files (default: ../bench)")
    args = ap.parse_args()

    data = load(args.out)
    if not data:
        sys.exit(f"no *_stats.csv found in {args.out}")

    n_runs = len({r for (_, _, r) in data})
    print(f"[aggregate] {len(data)} runs, {n_runs} run index(es), from {args.out}")
    if n_runs < 5:
        print(f"[aggregate] NOTE: only {n_runs} run(s) per configuration. "
              "The protocol asks for 5.")

    lines = []
    table_exp1(data, lines)
    table_exp2(data, lines)
    table_residual(data, lines)
    table_jitter(data, lines)
    table_exp3(data, lines)
    table_mps_effects(data, lines)
    check_consistency(data, lines)

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(args.out, "tables.txt"), "w") as fh:
        fh.write(text)

    print(f"[aggregate] wrote {write_raw(data, args.out)}")
    print(f"[aggregate] wrote {write_summary(data, args.out)}")
    print(f"[aggregate] wrote {os.path.join(args.out, 'tables.txt')}")


if __name__ == "__main__":
    main()
