#!/usr/bin/env python3
"""Every experiment, what it feeds, and where its data is.

Reads the results tree, the family-to-key mapping in aggregate.py, and which keys the section's
table generator uses. A family whose key appears in results-tables.py feeds the results section;
the rest are the appendix, reachable with `sweep.sh --exp appendix`.

Writes $DATA/bench/OVERVIEW.md and prints it.

    utility/overview.py
"""
import csv
import datetime
import glob
import os
import re
import statistics
import sys

BENCH = os.environ.get("DATA", "/mnt/nvme/f.scarfato") + "/bench"
HERE = os.path.dirname(os.path.abspath(__file__))


def family_keys():
    """The folder -> aggregate key mapping, read from the source rather than imported."""
    src = open(os.path.join(HERE, "aggregate.py")).read()
    block = src[src.index("EXP_DIR = {"):]
    block = block[:block.index("}")]
    return dict(re.findall(r'"([a-z0-9-]+)":\s*"([a-z0-9-]+)"', block))


def section_keys():
    """The keys the section's table generator reads."""
    src = open(os.path.join(HERE, "results-tables.py")).read()
    return set(re.findall(r'"(exp[0-9]+(?:-[0-9a-z]+)?)"', src))


def table_labels():
    """The table labels the generator emits, in order."""
    return re.findall(r'sys\.stdout|"(tab:[a-z0-9-]+)"',
                      open(os.path.join(HERE, "results-tables.py")).read())


def stats_for(d):
    runs = sorted(glob.glob(os.path.join(d, "*", "r*_stats.csv")))
    days = sorted({datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat() for f in runs})
    return len(runs), (days[0] if days else "---"), (days[-1] if days else "---")


def configs_of(d):
    out = []
    for sub in sorted(glob.glob(os.path.join(d, "*"))):
        if not os.path.isdir(sub):
            continue
        runs = glob.glob(os.path.join(sub, "r*_stats.csv"))
        nets = sorted({os.path.basename(f).split("_", 2)[2][:-len("_stats.csv")]
                       for f in runs if len(os.path.basename(f).split("_")) > 2})
        out.append((os.path.basename(sub), len(runs), nets))
    return out


def main():
    keys = family_keys()
    sec = section_keys()
    rows, total = [], 0
    for d in sorted(glob.glob(os.path.join(BENCH, "*"))):
        fam = os.path.basename(d)
        if not os.path.isdir(d) or fam in ("aggregate", "trace"):
            continue
        key = keys.get(fam, "?")
        n, first, last = stats_for(d)
        total += n
        rows.append(dict(family=fam, key=key, configs=len(configs_of(d)), runs=n,
                         first=first, last=last,
                         feeds="results section" if key in sec else "appendix"))

    lines = []
    add = lines.append
    add("# The experiments, what they feed, and where they are")
    add("")
    add("Data root `{}`, one directory per family, one per configuration inside it,".format(BENCH))
    add("`r<k>_stats.csv` and `r<k>_latencies.csv` per run, with the run's own log beside them.")
    add("Regenerate the tables with `utility/results-tables.py`; regenerate any family with")
    add("`utility/sweep.sh --exp <family>`.")
    add("")
    add("| family | key | configs | runs | first | last | feeds |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        add("| `{}` | {} | {} | {} | {} | {} | {} |".format(
            r["family"], r["key"], r["configs"], r["runs"], r["first"], r["last"], r["feeds"]))
    add("| **total** | | | **{}** | | | |".format(total))
    add("")
    add("## The families the results section reads, and the tables they feed")
    add("")
    add("| key | families | tables |")
    add("|---|---|---|")
    table_of = {
        "exp1": "tab:iso-rates, tab:iso-dispersion, references everywhere",
        "exp2": "tab:two-rates, tab:two-dispersion",
        "exp2-1t": "tab:two-rates, tab:two-dispersion, tab:compare-rates, tab:compare-dispersion",
        "exp3": "tab:mps-rates, tab:mps-dispersion, tab:compare-rates, tab:compare-dispersion",
        "exp5": "tab:aux-builds",
        "exp7": "tab:cpu-rates, tab:cpu-dispersion",
        "exp8": "tab:aux-builds, tab:wq-rows",
        "exp9": "tab:mem-budget, tab:mem-budget-dispersion, tab:mem-matrix",
    }
    for k in sorted(sec):
        fams = sorted(f for f, kk in keys.items() if kk == k)
        add("| {} | {} | {} |".format(k, ", ".join("`%s`" % f for f in fams),
                                      table_of.get(k, "---")))
    add("")
    add("## Every configuration, with its runs")
    add("")
    for r in rows:
        add("### `{}` ({}, {})".format(r["family"], r["key"], r["feeds"]))
        add("")
        add("| configuration | runs | networks with a stats file |")
        add("|---|---|---|")
        for cfg, n, nets in configs_of(os.path.join(BENCH, r["family"])):
            add("| `{}` | {} | {} |".format(cfg, n, ", ".join(nets) if nets else "---"))
        add("")

    text = "\n".join(lines) + "\n"
    dest = os.path.join(BENCH, "OVERVIEW.md")
    open(dest, "w").write(text)
    print(text)
    sys.stderr.write("  wrote {}\n".format(dest))


if __name__ == "__main__":
    main()
