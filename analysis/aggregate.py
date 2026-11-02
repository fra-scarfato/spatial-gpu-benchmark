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

The LaTeX tables are rendered from the same data by results-tables.py and tex-tables.py, and a run
whose median is far from its siblings' is reported here as suspect.
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

# Results-tree folder -> the short key the table generators use. The folders are named for what
# the experiment is; the keys stay short because every table definition refers to them, and only
# this mapping has to know the folder names. It lives in lib/families.py, which is the module about
# families, so there is one copy of it rather than a copy and a parser.
from lib.families import EXP_DIR, TARGET_FIRST  # noqa: E402


# Only gpu_ms is plotted; wall_ms includes the host round trip and prep_ms is
# zero by construction in fixed-input mode. All three are still carried through
# to the per-experiment files so the choice can be revisited without re-running anything.
METRICS = ("gpu_ms", "wall_ms", "prep_ms")

# The column order Network::writeStats emits. Keeping it in one place means
# read_stats and write_raw cannot drift apart.
COLS = ("metric", "n", "mean", "median", "std", "p95", "p99", "max", "min",
        "p99_minus_median", "count", "duration_s")


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
        # The folder names say what each experiment IS; the generators use short
        # keys, so they do not have to change whenever a folder is renamed.
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
            elif exp == "exp11" and config.endswith("-alone"):
                # The additivity family: only the target runs, and the configuration names it.
                net = config[: -len("-alone")]
            elif exp == "exp9" and config.startswith(("mem", "compute", "spin", "mps_", "mps-")):
                net = "mobilenet"
            elif exp in TARGET_FIRST:
                # The stress model's and the L2 chapter's families: one measured target, named first in the configuration
                # (mobilenet-saxpy-L030, resnet+inception); the co-runner writes no stats file here.
                net = re.split(r"[-+]", config)[0]
            elif exp.startswith("exp5") and config == "aux1_whole":
                # The isolated baseline of the auxiliary-stream family is the
                # aux-enabled ResNet on its own.
                net = "resnet"
            else:
                raise ValueError(f"cannot determine the network for {path!r}")
            # The network is the first field of the configuration name. A name that
            # does not parse must fail loudly rather than be filed under the wrong
            # network, so this list is the place a new network has to be declared.
            known = ("mobilenet", "resnet", "inception", "retinanet", "vit",
             "retinanet_a", "retinanet_b",
             # the validation networks of the p99 models (stress-new, stress-validation-new)
             "mobilenetint8", "resnetint8", "inceptionint8", "retinanetint8", "efficientnet")
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


def _run_throughput(data, exp, config, run):
    """Total inferences over the shared window, for one run."""
    rec = data.get((exp, config, run))
    if not rec:
        return None
    cnt = sum(v["gpu_ms"]["count"] for v in rec.values())
    dur = max(v["gpu_ms"]["duration_s"] for v in rec.values())
    return cnt / dur if dur else None


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


def check_consistency(data):
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
    for exp, config, net, r, v, m in sorted(flagged):
        print(f"[aggregate] SUSPECT {exp} {config} {net} run {r}: {v:.4f} ms against the "
              f"family median {m:.4f} ms ({100*(v/m-1):+.0f}%); probably taken while something "
              f"else used the GPU. Re-run it: delete its *_stats.csv, re-run ./measure/sweep.sh.")
    return flagged


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

    check_consistency(data)
    for path in write_raw(data, aggdir):
        print(f"[aggregate] wrote {path}")
    print(f"[aggregate] wrote {write_summary(data, aggdir)}")


if __name__ == "__main__":
    main()
