#!/usr/bin/env python3
"""aggregate.py — turn the per-run statistics into the numbers the thesis needs.

The unit of replication is the RUN, not the inference, and that single decision
drives everything below.

  level 1 (within a run)  the runner has already summarised the run: its
                          <name>_stats.csv holds median/mean/p99/std over that
                          process's ~10k-60k inferences. We read that.
  level 2 (across runs)   the point estimate is the MEDIAN OF THE 5 RUN-LEVEL
                          VALUES, and the run-to-run spread is reported as
                          [min, max].

The tempting alternative -- pool every raw inference from the 5 runs into one
vector and take its p99 -- is the classic pseudo-replication error: it treats
~300k correlated observations from 5 processes as independent, and therefore
understates the uncertainty several-fold. The 5 fresh processes exist precisely
to provide a between-run error bar, so the aggregation has to expose it.

Input layout — the runner writes the results as a tree, and the experiment and
the configuration come from the DIRECTORY names, so the file names stay short:

  <out>/<experiment>/<config>/r<run>[_<label>]_stats.csv
  <out>/<experiment>/<config>/r<run>[_<label>]_latencies.csv
  <out>/<experiment>/<config>/r<run>[_<label>].log

Outputs, all into <out>/aggregate/ — derived numbers live beside the data, never
mixed into them:
  raw.csv       tidy/long: one row per (experiment, config, network, run, metric)
  summary.csv   across-run point estimate and spread per config/network
  tables.txt    the derived tables, also printed to stdout
"""

import argparse
import csv
import glob
import os
import re
import statistics as stats
import sys

# The file name carries only the run number and the optional network label:
#
#   r3_stats.csv             one network per process, or a process holding one
#   r3_mobilenet_stats.csv   network — the runner adds a label only when a run
#                            holds more than one network.
#
# The experiment and the configuration are NOT in the file name: they are the two
# directory levels above it. That is what keeps these short and the tree
# readable, and it is why the parser below walks paths rather than names.
RUN_RE = re.compile(r'^r(?P<run>\d+)(?:_(?P<net>mobilenet|resnet))?_stats\.csv$')

# Results-tree folder -> the short key used throughout this file.
#
# The folders are named for what the experiment is, because a directory called
# "exp1" tells a reader nothing and the numbering can move as the thesis gets
# written. The keys stay short here because they appear in every table function
# below; only this mapping has to know the folder names.
EXP_DIR = {
    "isolated-scaling":    "exp1",
    "concurrent-greenctx": "exp2",
    "concurrent-mps":      "exp3",
}

# Only gpu_ms is plotted; wall_ms includes the host round trip and prep_ms is
# zero by construction in fixed-input mode. All three are still carried through
# to raw.csv so the choice can be revisited without re-running anything.
METRICS = ("gpu_ms", "wall_ms", "prep_ms")

# The column order Network::writeStats emits. Keeping it in one place means
# read_stats and write_raw cannot drift apart.
COLS = ("metric", "n", "mean", "median", "std", "p95", "p99", "max", "min",
        "p99_minus_median", "count", "duration_s")

# How many SMs each configuration asked for, as a percentage of the device.
#
# Percentages rather than SM counts because that is what the protocol specifies
# and what the runner was told; the grant was read back and verified on this
# board (16 SMs, alignment 2, so 25%->4, 50%->8, 75%->12, 100%->16). If the
# board ever changes, these are the numbers to re-check against the logs.
EXP1_PCT = {"whole": 100, "gc100": 100, "gc75": 75, "gc50": 50, "gc25": 25}
EXP2_PCT = {"baseline": {"mobilenet": 100, "resnet": 100},
            "50_50":    {"mobilenet": 50,  "resnet": 50},
            "75_25":    {"mobilenet": 75,  "resnet": 25},
            "25_75":    {"mobilenet": 25,  "resnet": 75}}
TOTAL_SMS = 16


def read_stats(path):
    """One run's stats file -> {metric: {column: value}}."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = {c: float(row[c]) for c in COLS[1:]}
    return out


def load(outdir):
    """Read every stats file in a results tree.

    Walks <outdir>/<experiment>/<config>/r<run>[_<label>]_stats.csv: the
    experiment and the configuration come from the directory names, and only the
    run number and the network label come from the file name.

    The key is (exp, config, run) and not the file name because a single run of
    Experiment 2 produces two files — one per network — that belong together:
    they were measured in the same process during the same window, so pairing
    them is what makes their comparison meaningful. Experiment 3 gets the same
    pairing for a weaker reason: two processes, but launched together.
    """
    data = {}
    for path in sorted(glob.glob(os.path.join(outdir, "**", "*_stats.csv"),
                                 recursive=True)):
        m = RUN_RE.match(os.path.basename(path))
        if not m:
            continue
        cfgdir = os.path.dirname(path)
        config = os.path.basename(cfgdir)
        # The folder names say what each experiment IS; this file keeps short
        # keys, so the tables below and their lookup dictionaries do not have to
        # change whenever a folder is renamed for readability.
        exp = EXP_DIR.get(os.path.basename(os.path.dirname(cfgdir)))
        if exp is None:
            # Not inside the tree — a stray stats file at the top level, say.
            continue
        run = int(m.group("run"))
        net = m.group("net")
        # When there is no label in the file name, the network has to be
        # deduced. exp1 puts it in the config name ("mobilenet_gc50"), and
        # exp3's single-process control is the MobileNet client. Anything else
        # is a naming change we have not accounted for, so fail loudly rather
        # than silently mislabelling a row.
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


def runs_of(data, exp, config):
    """Which run indices exist for this configuration."""
    return {r for (e, c, r) in data if e == exp and c == config}


def aggr(data, exp, config, net, metric, col):
    """Aggregate one statistic across the runs, -> (median, min, max, n_runs).

    Returns None when there is nothing to aggregate, so callers can distinguish
    "this configuration was not run" from "this configuration measured zero"
    and skip the row instead of printing a misleading one.

    Both the point estimate and the spread come from the RUN-LEVEL values: the
    median of the five medians, not the median of the pooled samples. See the
    module docstring for why that matters.
    """
    vals = [data[(exp, config, r)][net][metric][col]
            for r in sorted(runs_of(data, exp, config))
            if (exp, config, r) in data and net in data[(exp, config, r)]]
    if not vals:
        return None
    return stats.median(vals), min(vals), max(vals), len(vals)


def exp1_configs(net):
    """The five Experiment 1 configurations for one network, in order."""
    return [f"{net}_{k}" for k in ("whole", "gc100", "gc75", "gc50", "gc25")]


def hdr(title):
    """A section heading padded out to a constant width, so the file scans."""
    return f"\n=== {title} " + "=" * max(0, 74 - len(title)) + "\n"


def table_exp1(data, lines):
    lines.append(hdr("EXPERIMENT 1 - isolated: latency vs SM share"))
    lines.append("  Latency is the median of the 5 run medians; [min,max] is the\n"
                 "  run-to-run spread. Ratio is against the same network's `whole` run.\n")
    lines.append(f"  {'network':<10} {'config':<9} {'SMs':>4} "
                 f"{'median ms':>10} {'[min, max]':>18} {'ratio':>7} {'p99 ms':>8}")
    for net in ("mobilenet", "resnet"):
        # The ratio is against this network's own unpartitioned run, so the
        # number answers "what did restricting the SMs cost this model?".
        base = aggr(data, "exp1", f"{net}_whole", net, "gpu_ms", "median")
        for cfg in exp1_configs(net):
            a = aggr(data, "exp1", cfg, net, "gpu_ms", "median")
            p99 = aggr(data, "exp1", cfg, net, "gpu_ms", "p99")
            if a is None:
                continue
            med, lo, hi, n = a
            ratio = f"{med / base[0]:.3f}" if base else "-"
            # cfg looks like "mobilenet_gc50"; split once, keep the tail, so
            # the SM lookup does not care which network this is.
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
        # Throughput needs both networks, so it is computed per run and only
        # then reduced. Dividing totals instead would weight a run that
        # happened to last longer more heavily than the others.
        thr = []
        for r in sorted(runs_of(data, "exp2", cfg)):
            rec = data.get(("exp2", cfg, r))
            if not rec or len(rec) < 2:
                continue
            cnt = sum(rec[n]["gpu_ms"]["count"] for n in rec)
            # max(), not min(): the two networks shared the board for as long
            # as the longer of them ran, and dividing by the shorter window
            # would invent throughput that never existed.
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
            # The unpartitioned baseline shares the whole GPU, so its isolated
            # reference is `whole`, not a green-context run.
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
    # "iso" = Experiment 1 (alone), "conc" = Experiment 2 (sharing), so the two
    # halves of the table can be read against each other line by line.
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


# Experiment 3 is the one place where two networks are two INDEPENDENT
# processes. In Experiments 1 and 2 a single process owns both networks, so one
# run yields a matched pair of stats files and the two windows are identical by
# construction. Here there is no shared clock, so the windows are aligned only
# to the difference in warm-up length, about 0.5%. The pair is still aggregated
# as one record; the caveat is stamped on the table so it cannot be forgotten.
EXP3_ORDER = ["mps_single", "mps_off", "mps_default",
              "mps_atp_50_50", "mps_atp_75_25", "mps_atp_25_75",
              "mps_static_50_50", "mps_static_75_25", "mps_static_25_75"]


def exp3_configs(data):
    """Known configurations first, in a deliberate order; anything unknown after.

    Sorting them so a half-finished sweep still produces a readable table, and
    a configuration added later still shows up rather than being dropped.
    """
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
    """The comparisons that each need a control AND a measurement.

    These rows are what make Experiment 3 interpretable, and they are the
    reason the matrix contains configurations that are not themselves
    interesting:
      - MPS per-client cost: does MPS slow a lone client down? Without this,
        mps_default vs mps_off cannot be attributed to co-scheduling, because
        the difference could just as easily be MPS's own overhead.
      - Process-model cost: two processes versus one, with no MPS on either
        side. This is the measured bound on the green-context-vs-MPS confound,
        which cannot be removed (those two mechanisms cannot run at the same
        process count) but can at least be sized.
      - Co-scheduling: what MPS is actually for.
    """
    def med(exp, cfg, net):
        a = aggr(data, exp, cfg, net, "gpu_ms", "median")
        return a[0] if a else None

    # (label, description, reference, measurement)
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
    # ATP rows compare against mps_default rather than mps_off: the question is
    # whether throttling the thread slots helps ON TOP of MPS, not whether it
    # helps compared to no MPS at all.
    for cfg in ("mps_atp_50_50", "mps_atp_75_25", "mps_atp_25_75"):
        split = cfg[len("mps_atp_"):]
        for net in ("mobilenet", "resnet"):
            checks.append((f"ATP {split}", f"{net}: vs mps_default (no ATP)",
                           ("exp3", cfg, net), ("exp3", "mps_default", net)))
    # The static rows are the cross-mechanism comparison, and the only place
    # where the process-model difference is real: a green context is
    # intra-process, a static MPS partition is inter-process.
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


def write_raw(data, outdir):
    """One row per (experiment, config, network, run, metric) — for plotting."""
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
    """The across-run numbers in one row per config/network.

    The column names are doubled on purpose: "median__median" is the median
    (across runs) of the median (within a run), which is exactly the two-level
    structure described in the module docstring. Collapsing that into a single
    word would hide which level a number came from.
    """
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

    The after-the-fact half of the interference guard. The sentinel in
    sweep.sh catches a busy board before a run starts, but it samples only 500
    inferences and can miss a co-tenant that is between iterations; a
    contaminated run, however, almost always shows up as one run several times
    slower than the other four. So this is where contamination that slipped
    past the sentinel gets noticed.

    The 25% threshold is deliberately loose: it has to clear the normal
    run-to-run drift (well under 1%) by a wide margin while still catching a
    genuinely spoiled run. Fewer than three runs is skipped, because with two
    values there is no "family" to compare against.
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


def main():
    # Defaulting relative to the script's own location, so the tool works from
    # any working directory: the results live beside the runner, not here.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "bench"),
        help="directory holding the *_stats.csv files (default: ../bench)")
    args = ap.parse_args()

    data = load(args.out)
    if not data:
        sys.exit(f"no *_stats.csv found under {args.out}")

    # Derived numbers go in their own folder, so the tree above stays
    # measurements only. Created on demand, which is also what gives the layout
    # something to point at before anything has been run.
    aggdir = os.path.join(args.out, "aggregate")
    os.makedirs(aggdir, exist_ok=True)

    # The number of distinct run indices, i.e. how many repetitions there are.
    # Worth reporting, because less than 5 means the between-run spread is
    # thinner than the protocol intends and the tables will look tighter than
    # the data deserves.
    n_runs = len({r for (_, _, r) in data})
    print(f"[aggregate] {len(data)} runs, {n_runs} run index(es), from {args.out}")
    if n_runs < 5:
        print(f"[aggregate] NOTE: only {n_runs} run(s) per configuration. "
              "The protocol asks for 5.")

    # Every table appends to one list of lines and returns it, so the tables
    # can be reordered or skipped without touching how the text is emitted.
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
    with open(os.path.join(aggdir, "tables.txt"), "w") as fh:
        fh.write(text)

    print(f"[aggregate] wrote {write_raw(data, aggdir)}")
    print(f"[aggregate] wrote {write_summary(data, aggdir)}")
    print(f"[aggregate] wrote {os.path.join(aggdir, 'tables.txt')}")


if __name__ == "__main__":
    main()
