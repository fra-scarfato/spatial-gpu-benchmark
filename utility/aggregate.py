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

ONE EXCEPTION, and it is deliberate. The observed maximum is reduced with max
over the runs, not with the median: the run-level `max` is the largest sample
that run saw, so the maximum over the runs IS the maximum over every inference
measured for the configuration (max of maxima = max of the union). Taking the
median of five maxima would discard exactly the rare event the column exists to
catch, because a stall usually spoils one run in five rather than all of them.
Only this one column departs from the median-of-runs rule; obs_max() is the
single place that does it, and everything else keeps the rule above.

Input layout — the runner writes the results as a tree, and the experiment and
the configuration come from the DIRECTORY names, so the file names stay short:

  <out>/<experiment>/<config>/r<run>[_<label>]_stats.csv
  <out>/<experiment>/<config>/r<run>[_<label>]_latencies.csv
  <out>/<experiment>/<config>/r<run>[_<label>].log

Outputs, all into <out>/aggregate/ — derived numbers live beside the data, never
mixed into them:
  <exp>.csv     tidy/long, one file per experiment: one row per
                (config, network, run, metric)
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

# Where the data lives. Not on the root filesystem, which is small and shared,
# and not in this repository, which is code: the engines, the image set, the
# labels and the results are on the NVMe. Override with the DATA variable.
DATA_ROOT = os.environ.get("DATA", "/mnt/nvme/f.scarfato")

# The file name carries only the run number and the optional network label:
#
#   r3_stats.csv             one network per process, or a process holding one
#   r3_mobilenet_stats.csv   network — the runner adds a label only when a run
#                            holds more than one network.
#
# The experiment and the configuration are NOT in the file name: they are the two
# directory levels above it. That is what keeps these short and the tree
# readable, and it is why the parser below walks paths rather than names.
# The label is whatever the runner was given with --net ...[:<label>], not a
# fixed pair: the regex used to name mobilenet and resnet only, which silently
# dropped every row of a run labelled with any other network. The known-name
# assertion below is what keeps a typo from entering as a new network.
RUN_RE = re.compile(r'^r(?P<run>\d+)(?:_(?P<net>[A-Za-z][A-Za-z0-9_+-]*))?_stats\.csv$')

# Results-tree folder -> the short key used throughout this file.
#
# The folders are named for what the experiment is, because a directory called
# "exp1" tells a reader nothing and the numbering can move as the thesis gets
# written. The keys stay short here because they appear in every table function
# below; only this mapping has to know the folder names.
EXP_DIR = {
    "isolated-scaling":    "exp1",
    "concurrent-greenctx": "exp2",
    # The work queue experiment is Experiment 2's setup with the work queue half
    # of the green context descriptor set explicitly, so it belongs in the SAME
    # tables: a devctx_8_50_50 row sitting next to the 50_50 row it must
    # reproduce is the whole point of running it. Mapping two folders onto one
    # key is what makes that free — the tables see one configuration space.
    "concurrent-wq":       "exp2",
    # Same idea again: the auxiliary-stream family is Experiment 2's pair with a
    # different ResNet build, so it needs its own key (its rows are not
    # comparable to the default build) but the same table shape.
    "concurrent-aux":      "exp5",
    # The CPU-pinning arms live in two folders because they were run at
    # different times; the config names do not collide, so one key is enough and
    # the aggregate picks up every replicate.
    # The same three intra-process families, measured with a single
    # submitting thread. Distinct keys on purpose: mapping them onto exp2/
    # exp4/exp5 would collide with the two-thread runs of the same config
    # names, and one set would silently overwrite the other.
    "concurrent-greenctx-1t": "exp2-1t",
    "concurrent-wq-1t":       "exp4-1t",
    "concurrent-aux-1t":      "exp5-1t",
    "pinning":             "exp6",
    "submit-model":        "exp7",
    "wq-s1":                "exp8",
    "wq-s4":                "exp8",
    "mem-s1":               "exp9",
    "mem-s2":               "exp9",
    "mem-s3":               "exp9",
    "mem-s4":               "exp9",
    "mem-s5":               "exp9",
    "mem-s6":               "exp9",
    "repro":               "exp6",
    "concurrent-mps":      "exp3",
}

# Only gpu_ms is plotted; wall_ms includes the host round trip and prep_ms is
# zero by construction in fixed-input mode. All three are still carried through
# to the per-experiment files so the choice can be revisited without re-running anything.
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

# Experiment 2's original four configurations, and the work queue arms that
# extend them. One list, built once, so the four tables that need it cannot
# disagree about which configurations exist.
EXP2_BASE = ("baseline", "50_50", "75_25", "25_75")
WQ_ARMS   = (("devctx", "8"), ("balanced", "8"),
             ("devctx", "2"), ("balanced", "2"),
             ("devctx", "1"))
WQ_SPLITS = ("50_50", "75_25")


def exp2_configs(data):
    """Every Experiment 2 configuration present, originals first.

    Ordering is deliberate: the original rows appear before the work queue arms
    that must reproduce them, so a reader can compare a row against the one it
    is meant to match without hunting. Configurations never run are skipped
    rather than shown empty, which is what keeps a partial sweep readable.
    """
    present = {c for (e, c, _) in data if e == "exp2"}
    out = [c for c in EXP2_BASE if c in present]
    for tag, lim in WQ_ARMS:
        for split in WQ_SPLITS:
            c = f"{tag}_{lim}_{split}"
            if c in present:
                out.append(c)
    for lim in ("8", "2"):
        c = f"unpart_{lim}"
        if c in present:
            out.append(c)
    return out


def exp2_pct(cfg):
    """{network: SM percentage} for an Experiment 2 configuration.

    The original four have a fixed table above; the work queue arms carry their
    split in their own names (devctx_8_50_50 means MobileNet 50, ResNet 50), and
    the unpartitioned pair shares the whole device. Reading the split back out of
    the name is why that name has the shape it has.
    """
    if cfg in EXP2_PCT:
        return EXP2_PCT[cfg]
    if cfg.startswith("unpart_"):
        return {"mobilenet": 100, "resnet": 100}
    parts = cfg.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return {"mobilenet": int(parts[-2]), "resnet": int(parts[-1])}
    raise ValueError(f"cannot read an SM split out of the config name {cfg!r}")


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
            # The interferer matrix: one network plus the synthetic participant, which
            # writes no stats file of its own. The participant is not a network, so the
            # single unlabelled stats file here is MobileNet. Every arm of this family
            # runs MobileNet; the other exp9 family (the budget arm) has two networks
            # and its files are labelled.
            elif exp == "exp9" and config.startswith(("mem", "compute", "mps_")):
                net = "mobilenet"
            elif exp.startswith("exp5") and config == "aux1_whole":
                # The isolated baseline of the auxiliary-stream family is the
                # aux-enabled ResNet on its own.
                net = "resnet"
            else:
                raise ValueError(f"cannot determine the network for {path!r}")
            # The network is the first field of the configuration name. A name that
            # does not parse must fail loudly rather than be filed under the wrong
            # network, so this list is the place a new network has to be declared.
            known = ("mobilenet", "resnet", "inception", "retinanet", "vit")
            assert net in known, f"unexpected config {config!r}"
        data.setdefault((exp, config, run), {})[net] = read_stats(path)
    return data


def runs_of(data, exp, config):
    """Which run indices exist for this configuration."""
    return {r for (e, c, r) in data if e == exp and c == config}


def aggr(data, exp, config, net, metric, col, how=stats.median):
    """Aggregate one statistic across the runs, -> (point, min, max, n_runs).

    `how` is the across-run reducer for the point estimate and defaults to the
    median, which is the rule for every column but one. The observed maximum
    passes max here -- through obs_max(), not directly -- for the reason in the
    module docstring.

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
    return how(vals), min(vals), max(vals), len(vals)


def obs_max(data, exp, config, net, metric="gpu_ms"):
    """The OBSERVED MAXIMUM of a configuration: max over the runs of the maxima.

    Not the median of the run maxima. The run-level `max` is the largest sample
    a run actually saw; a rare stall that costs a deadline normally shows up in
    one run out of five, and the median of five values would drop it on the
    floor. Because the maximum of a union is the maximum of the maxima, taking
    max() over the runs gives exactly the maximum over all the inferences
    pooled -- so this reads no extra data (the per-inference CSVs do not have to
    be rescanned) and every other statistic keeps the run as its unit of
    replication.

    Returns None when the configuration has no runs, so callers can skip it.
    """
    a = aggr(data, exp, config, net, metric, "max", how=max)
    return a[0] if a else None


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
    # Only mentioned when the work queue arms are actually present, so the
    # original four-configuration table reads exactly as it did before.
    if any(c.startswith(("devctx_", "balanced_", "unpart_"))
           for (e, c, _) in data if e == "exp2"):
        lines.append(
            "  The devctx_*/balanced_* rows are the SAME configurations with the green\n"
            "  context's work queue settings pinned in the name (scope_limit, the limit\n"
            "  being both the declared concurrency and CUDA_DEVICE_MAX_CONNECTIONS), and\n"
            "  unpart_* is the unpartitioned run at a pinned queue budget.\n"
            "  devctx_8_50_50 must reproduce 50_50: if it does not, the path that sets\n"
            "  the work queue configuration is wrong and the other arms mean nothing.\n")
    lines.append(f"  {'split':<16} {'network':<10} {'SMs':>4} {'median ms':>10} "
                 f"{'p99 ms':>8} {'std ms':>8} {'tail ms':>8} {'agg inf/s':>10}")
    order = exp2_configs(data)
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
            sms = round(TOTAL_SMS * exp2_pct(cfg)[net] / 100)
            lines.append(f"  {cfg:<16} {net:<10} {sms:>4} {med:>10.4f} "
                         f"{p99:>8.4f} {sd:>8.4f} {tl:>8.4f} {agg_thr:>10}")
        lines.append("")
    return lines


def table_residual(data, lines):
    lines.append(hdr("RESIDUAL INTERFERENCE - concurrent / isolated, same SM count"))
    lines.append("  Isolated = Experiment 1 at the same SM count. This is the number\n"
                 "  SM partitioning could NOT remove: memory bandwidth, L2, work queues.\n")
    lines.append(f"  {'network':<10} {'split':<16} {'SMs':>4} "
                 f"{'isolated':>10} {'concurrent':>11} {'overhead':>9}")
    for cfg in exp2_configs(data):
        for net in ("mobilenet", "resnet"):
            conc = aggr(data, "exp2", cfg, net, "gpu_ms", "median")
            if conc is None:
                continue
            # Unpartitioned configurations share the whole GPU, so their
            # isolated reference is `whole`, not a green-context run.
            pct = exp2_pct(cfg)[net]
            ref_cfg = f"{net}_whole" if pct == 100 else f"{net}_gc{pct}"
            iso = aggr(data, "exp1", ref_cfg, net, "gpu_ms", "median")
            if iso is None:
                continue
            sms = round(TOTAL_SMS * pct / 100)
            lines.append(f"  {net:<10} {cfg:<16} {sms:>4} {iso[0]:>10.4f} "
                         f"{conc[0]:>11.4f} {100 * (conc[0] / iso[0] - 1):>8.1f}%")
        lines.append("")
    return lines


def table_jitter(data, lines):
    lines.append(hdr("JITTER - the tail is what a deadline sees"))
    lines.append(f"  {'network':<10} {'config':<20} {'std ms':>9} "
                 f"{'p99-med ms':>11} {'obs-max ms':>11}")
    # "iso" = Experiment 1 (alone), "conc" = Experiment 2 (sharing), so the two
    # halves of the table can be read against each other line by line.
    #
    # The last column is the observed maximum, i.e. max over the runs and not
    # the median of the run maxima -- see obs_max. It is the one number here
    # that is not a typical value, so it is labelled obs-max to say so.
    for net in ("mobilenet", "resnet"):
        for cfg in exp1_configs(net):
            sd = aggr(data, "exp1", cfg, net, "gpu_ms", "std")
            if sd is None:
                continue
            lines.append(f"  {net:<10} {'iso ' + cfg.split('_',1)[1]:<18} "
                         f"{sd[0]:>9.4f} "
                         f"{aggr(data,'exp1',cfg,net,'gpu_ms','p99_minus_median')[0]:>11.4f} "
                         f"{obs_max(data,'exp1',cfg,net):>11.4f}")
    for cfg in exp2_configs(data):
        for net in ("mobilenet", "resnet"):
            sd = aggr(data, "exp2", cfg, net, "gpu_ms", "std")
            if sd is None:
                continue
            lines.append(f"  {net:<10} {'conc ' + cfg:<20} {sd[0]:>9.4f} "
                         f"{aggr(data,'exp2',cfg,net,'gpu_ms','p99_minus_median')[0]:>11.4f} "
                         f"{obs_max(data,'exp2',cfg,net):>11.4f}")
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


def table_dispersion(data, lines):
    """The dispersion view, with the observed maximum.

    One row per (configuration, network). Every column except obs-max is the
    median across the runs of that run-level statistic; obs-max is the maximum
    over every inference measured for the configuration (see obs_max).

    The `x iso` columns divide by the same network's isolated `whole` run of
    Experiment 1 — the only reference that stays valid across experiments,
    because Experiment 1 shows the isolated spread barely moves with the SM
    budget (0.9-1.2x), so a single reference is enough to read every row
    against, including the rows of Experiment 1 itself. Typical latency and
    throughput are in the tables above.
    """
    lines.append(hdr("DISPERSION - spread, tail and observed maximum"))
    lines.append(
        "  All columns reduce across the 5 runs with the median, EXCEPT obs-max,\n"
        "  which is the maximum over every inference measured for that configuration:\n"
        "  a rare stall shows up in one run of five, and a median would drop it.\n"
        "  `x iso` is against the same network's isolated `whole` run of Experiment 1.\n")
    lines.append(f"  {'config':<20} {'network':<9} {'std':>8} {'x iso':>7} "
                 f"{'p95':>8} {'p99':>8} {'tail':>8} {'x iso':>7} "
                 f"{'obs-max':>8} {'x iso':>7}")

    def ref(net, col):
        """The isolated `whole` value of one statistic for this network."""
        a = aggr(data, "exp1", f"{net}_whole", net, "gpu_ms", col)
        return a[0] if a else None

    def ratio(value, reference):
        return f"{value / reference:.2f}x" if reference else "-"

    def row(exp, cfg, net, label):
        if aggr(data, exp, cfg, net, "gpu_ms", "median") is None:
            return
        sd  = aggr(data, exp, cfg, net, "gpu_ms", "std")[0]
        p95 = aggr(data, exp, cfg, net, "gpu_ms", "p95")[0]
        p99 = aggr(data, exp, cfg, net, "gpu_ms", "p99")[0]
        tl  = aggr(data, exp, cfg, net, "gpu_ms", "p99_minus_median")[0]
        mx  = obs_max(data, exp, cfg, net)
        # The reference for obs-max has to be an observed maximum too, and not
        # the isolated run's median-of-maxima: both sides of the ratio must be
        # the same kind of number or the multiplier means nothing.
        iso_max = obs_max(data, "exp1", f"{net}_whole", net)
        lines.append(
            f"  {label:<20} {net:<9} {sd:>8.4f} {ratio(sd, ref(net,'std')):>7} "
            f"{p95:>8.4f} {p99:>8.4f} {tl:>8.4f} "
            f"{ratio(tl, ref(net,'p99_minus_median')):>7} "
            f"{mx:>8.4f} {ratio(mx, iso_max):>7}")

    lines.append("  -- Experiment 1: isolated, one network")
    for net in ("mobilenet", "resnet"):
        for cfg in exp1_configs(net):
            row("exp1", cfg, net, cfg.split("_", 1)[1])
    lines.append("")
    lines.append("  -- Experiment 2: two networks, one process, disjoint green contexts")
    for cfg in exp2_configs(data):
        for net in ("mobilenet", "resnet"):
            row("exp2", cfg, net, cfg)
    lines.append("")
    lines.append("  -- Experiment 3: two processes, MPS")
    for cfg in exp3_configs(data):
        for net in ("mobilenet", "resnet"):
            row("exp3", cfg, net, cfg)
    lines.append("")
    lines.append("  -- Experiment 5: the auxiliary-stream family")
    for cfg in aux_configs(data):
        for net in ("mobilenet", "resnet"):
            row("exp5", cfg, net, cfg)
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


# Experiment 5, the auxiliary-stream family. Ordered so the controlled
# comparisons come first: the scope at the coherent concurrency, then the same at
# the per-client reading of it, then the sweep, then the attribution arm.
EXP5_ARMS = (("devctx",   "3"), ("balanced", "3"),
             ("devctx",   "2"), ("balanced", "2"),
             ("balanced", "1"), ("balanced", "8"),
             ("balanced", "32"), ("devctx", "8"))
EXP5_SPLITS = ("50_50", "75_25")


def aux_configs(data):
    """Every auxiliary-stream configuration present, baseline first."""
    present = {c for (e, c, _) in data if e == "exp5"}
    out = [c for c in ("aux1_whole",) if c in present]
    for tag, lim in EXP5_ARMS:
        for split in EXP5_SPLITS:
            c = f"{tag}_{lim}_{split}"
            if c in present:
                out.append(c)
    return out


def aux_pct(cfg):
    """{network: SM percentage} for an auxiliary-stream configuration."""
    if cfg == "aux1_whole":
        return {"resnet": 100}
    parts = cfg.split("_")
    return {"mobilenet": int(parts[-2]), "resnet": int(parts[-1])}


def table_aux(data, lines):
    """Experiment 5: the pair on a ResNet that really issues two streams.

    Every column is the one the work queue question needs: median and throughput
    because those are what the two mechanisms already agreed on, p99, tail and
    observed maximum because those are where they differed, and the aggregate
    because a configuration that wins on latency can still lose on throughput.
    """
    cfgs = aux_configs(data)
    if not cfgs:
        return lines
    lines.append(hdr("EXPERIMENT 5 - ResNet built with one auxiliary stream"))
    lines.append(
        "  Experiment 2's pair, with ResNet built allowing one auxiliary stream so it\n"
        "  issues two streams of its own; the pair therefore has three concurrent\n"
        "  stream-ordered workloads. Every other setting is Experiment 4's.\n"
        "  Select with: utility/sweep.sh --exp aux\n")
    lines.append(f"  {'config':<22} {'network':<10} {'median ms':>10} {'p99 ms':>8} "
                 f"{'std ms':>8} {'tail ms':>8} {'obs-max':>9} {'agg inf/s':>10}")
    for cfg in cfgs:
        thr = [t for t in (_run_throughput(data, "exp5", cfg, r)
                           for r in sorted(runs_of(data, "exp5", cfg))) if t]
        agg = f"{stats.median(thr):.0f}" if thr else "-"
        for net in ("mobilenet", "resnet"):
            a = aggr(data, "exp5", cfg, net, "gpu_ms", "median")
            if a is None:
                continue
            lines.append(
                f"  {cfg:<22} {net:<10} {a[0]:>10.4f} "
                f"{aggr(data,'exp5',cfg,net,'gpu_ms','p99')[0]:>8.4f} "
                f"{aggr(data,'exp5',cfg,net,'gpu_ms','std')[0]:>8.4f} "
                f"{aggr(data,'exp5',cfg,net,'gpu_ms','p99_minus_median')[0]:>8.4f} "
                f"{obs_max(data,'exp5',cfg,net):>9.4f} {agg:>10}")
        lines.append("")
    return lines


def pinning_configs(data):
    """The CPU-affinity arms present, in the order they make sense to read."""
    present = {c for (e, c, _) in data if e == "exp6"}
    order = ("unpinned_50_50", "unpinned_75_25", "shm_75_25",
             "onecore", "onecore_75_25", "twocore")
    return [c for c in order if c in present] + sorted(present - set(order))


def table_pinning(data, lines):
    """Experiment 6: where the two submitting threads are allowed to run.

    Same engines, same green contexts, same SM split, same queue settings ---
    CPU affinity is the only variable. If the MobileNet tail is controlled from
    the host side, it should move here with nothing device-side touched, and the
    median should not move at all. That is what makes this the controlled test
    of the tail's origin.
    """
    cfgs = pinning_configs(data)
    if not cfgs:
        return lines
    lines.append(hdr("EXPERIMENT 6 - where the two submitting threads may run"))
    lines.append(
        "  The 75/25 pair with CPU affinity varied and nothing else. `shm_75_25`\n"
        "  additionally moves the latency CSV to RAM, to remove the disk as a\n"
        "  variable; `unpinned_50_50` is the same idea at the other split.\n")
    lines.append(f"  {'config':<22} {'network':<10} {'median ms':>10} {'p99 ms':>8} "
                 f"{'tail ms':>8} {'std ms':>8} {'obs-max':>9} {'tail/med':>9}")
    for cfg in cfgs:
        for net in ("mobilenet", "resnet"):
            a = aggr(data, "exp6", cfg, net, "gpu_ms", "median")
            if a is None:
                continue
            med = a[0]
            tl = aggr(data, "exp6", cfg, net, "gpu_ms", "p99_minus_median")[0]
            lines.append(
                f"  {cfg:<22} {net:<10} {med:>10.4f} "
                f"{aggr(data,'exp6',cfg,net,'gpu_ms','p99')[0]:>8.4f} {tl:>8.4f} "
                f"{aggr(data,'exp6',cfg,net,'gpu_ms','std')[0]:>8.4f} "
                f"{obs_max(data,'exp6',cfg,net):>9.4f} {tl/med:>9.3f}")
        lines.append("")
    return lines


# Two submitting threads against one.
#
# The two-thread runs are the naive application -- one thread per network -- and
# every table above was built on them. The one-thread runs are the same
# configurations with a single thread driving both networks, keeping one
# inference in flight per network. Nothing else differs: same engines, same green
# contexts, same joint SM split, same queue settings, same protocol.
#
# The comparison exists because the difference is not where the medians are. The
# tail is what a real-time deployment cares about, and it is the tail that moves,
# by several times, while the median moves by a few percent at most.
SUBMITTER_PAIRS = (
    ("Experiment 2, green contexts",   "exp2", "exp2-1t"),
    # The work queue family has no exp4 key of its own: aggregate.py maps its
    # folder onto exp2 so its rows sit beside the Experiment 2 row each one has to
    # reproduce. Only the one-thread family is separate, so the pair is exp2 here.
    ("Experiment 4, work queues",      "exp2", "exp4-1t"),
    ("Experiment 5, auxiliary stream", "exp5", "exp5-1t"),
)


def table_submitter(data, lines):
    lines.append(hdr("THE SUBMITTING THREAD: two threads against one"))
    lines.append(
        "  `x` is the one-thread value divided by the two-thread value, so below 1\n"
        "  means the single submitter is better. tail is p99 - median; max is the\n"
        "  observed maximum. Values in milliseconds unless the column says inf/s.\n")
    for title, two, one in SUBMITTER_PAIRS:
        cfgs = sorted({c for (e, c, _) in data if e == two})
        shared = [c for c in cfgs if any(e == one for (e, c2, _) in data if c2 == c)]
        if not shared:
            continue
        lines.append(f"  {title}:  {two}  vs  {one}")
        lines.append(f"    {'config':<22} {'net':<10} {'2t med':>8} {'1t med':>8} {'x':>6} "
                     f"{'2t tail':>8} {'1t tail':>8} {'x':>6} {'2t max':>9} {'1t max':>9}")
        for cfg in shared:
            for net in ("mobilenet", "resnet"):
                a = aggr(data, two, cfg, net, "gpu_ms", "median")
                b = aggr(data, one, cfg, net, "gpu_ms", "median")
                if a is None or b is None:
                    continue
                at = aggr(data, two, cfg, net, "gpu_ms", "p99_minus_median")[0]
                bt = aggr(data, one, cfg, net, "gpu_ms", "p99_minus_median")[0]
                am = obs_max(data, two, cfg, net)
                bm = obs_max(data, one, cfg, net)
                lines.append(
                    f"    {cfg:<22} {net:<10} {a[0]:>8.4f} {b[0]:>8.4f} {b[0] / a[0]:>6.3f} "
                    f"{at:>8.4f} {bt:>8.4f} {(bt / at if at else 0.0):>6.3f} "
                    f"{am:>9.4f} {bm:>9.4f}")
            # The aggregate is a property of the configuration, not of one network.
            ta = [x for r in sorted(runs_of(data, two, cfg))
                  if (x := _run_throughput(data, two, cfg, r)) is not None]
            tb = [x for r in sorted(runs_of(data, one, cfg))
                  if (x := _run_throughput(data, one, cfg, r)) is not None]
            if ta and tb:
                lines.append(f"    {cfg:<22} {'aggregate':<10} "
                             f"{stats.median(ta):>8.0f} {stats.median(tb):>8.0f} "
                             f"{stats.median(tb) / stats.median(ta):>6.3f} inf/s")
            lines.append("")
    return lines


def write_raw(data, outdir):
    """One file per experiment: one row per (config, network, run, metric).

    Split by experiment because that is the boundary a reader works inside. The
    configurations of one experiment were measured against each other, in the same
    folder, under the same conditions, and nothing outside that experiment is a
    valid comparison; a file per experiment means a reader never has to filter the
    other seven out. The experiment stays a column, so the files concatenate back
    into the single table this used to be.
    """
    paths = []
    for exp in sorted({e for (e, _, _) in data}):
        path = os.path.join(outdir, "{}.csv".format(exp))
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["experiment", "config", "network", "run", "metric"] + list(COLS[1:]))
            for (e, config, run) in sorted(data):
                if e != exp:
                    continue
                for net in sorted(data[(e, config, run)]):
                    for metric in METRICS:
                        rec = data[(e, config, run)][net].get(metric)
                        if rec is None:
                            continue
                        w.writerow([e, config, net, run, metric] + [rec[c] for c in COLS[1:]])
        paths.append(path)
    return paths


def write_summary(data, outdir):
    """The across-run numbers in one row per config/network.

    The column names are doubled on purpose: "median__median" is the median
    (across runs) of the median (within a run), which is exactly the two-level
    structure described in the module docstring. Collapsing that into a single
    word would hide which level a number came from.

    observed_max is the one column NOT named that way, because it is not the
    median of the run-level values: it is the maximum over the runs of the run
    maxima, i.e. over every inference measured. See obs_max().
    """
    path = os.path.join(outdir, "summary.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "config", "network", "n_runs",
                    "median__median", "median__min", "median__max",
                    "std__median", "p95__median", "p99__median", "tail__median",
                    "observed_max", "throughput_inf_s"])
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
                            f"{aggr(data,exp,config,net,'gpu_ms','std')[0]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','p95')[0]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','p99')[0]:.6f}",
                            f"{aggr(data,exp,config,net,'gpu_ms','p99_minus_median')[0]:.6f}",
                            f"{obs_max(data,exp,config,net):.6f}",
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
                     "  Re-run them (delete the *_stats.csv and re-run ./utility/sweep.sh).\n")
        for exp, config, net, r, v, m in flagged:
            lines.append(f"  {exp} {config} {net} run {r}: {v:.4f} ms "
                         f"vs family median {m:.4f} ms  ({100*(v/m-1):+.0f}%)")
    return lines


def main():
    # The results tree is on the NVMe: not in this repository, and not on the
    # root filesystem either. DATA_ROOT is where every tool agrees it lives.
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "bench"),
        help="directory holding the *_stats.csv files (default: $DATA/bench)")
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
    table_dispersion(data, lines)
    table_aux(data, lines)
    table_pinning(data, lines)
    table_submitter(data, lines)
    table_mps_effects(data, lines)
    check_consistency(data, lines)

    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(aggdir, "tables.txt"), "w") as fh:
        fh.write(text)

    for path in write_raw(data, aggdir):
        print(f"[aggregate] wrote {path}")
    print(f"[aggregate] wrote {write_summary(data, aggdir)}")
    print(f"[aggregate] wrote {os.path.join(aggdir, 'tables.txt')}")


if __name__ == "__main__":
    main()
