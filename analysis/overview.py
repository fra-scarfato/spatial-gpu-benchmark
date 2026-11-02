#!/usr/bin/env python3
"""Every experiment, what it feeds, and where its data is.

Writes $DATA/bench/OVERVIEW.md and prints it. The families, the groups and the tree walk all come
from lib/families.py, so this file is only the rendering: a family feeds the results section when
the sweep's section group says so, and everything else is the appendix.

    analysis/overview.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lib import families  # noqa: E402

TABLE_OF = {
    "exp1": "tab:iso-rates, tab:iso-dispersion, references everywhere",
    "exp2-1t": "tab:two-rates, tab:two-dispersion, tab:compare-rates, tab:compare-dispersion",
    "exp3": "tab:mps-rates, tab:mps-dispersion, tab:compare-rates, tab:compare-dispersion",
    "exp5": "tab:aux-builds",
    "exp7": "tab:cpu-rates, tab:cpu-dispersion",
    "exp8": "tab:aux-builds, tab:wq-rows",
    "exp9": "tab:mem-budget, tab:mem-budget-dispersion, tab:mem-matrix",
}


def configurations(bench, family):
    """Every configuration of a family, with its replicate count and its networks."""
    out = []
    for cfg in sorted(p for p in (bench / family).iterdir() if p.is_dir()):
        files = sorted(cfg.glob("r*_stats.csv"))
        if not files:
            continue
        nets = sorted({f.name.split("_", 2)[2][: -len("_stats.csv")]
                       for f in files if len(f.name.split("_")) > 2})
        out.append((cfg.name, len({f.name.split("_")[0] for f in files}), nets))
    return out


def main():
    bench = families.bench_root()
    section = set(families.groups()["section"])
    keys = families.key_map()

    rows = []
    for fam in families.families_on_disk(bench):
        st = families.state(bench, fam)
        if st:
            st["feeds"] = "results section" if fam in section else "appendix"
            rows.append(st)

    lines = []
    add = lines.append
    add("# The experiments, what they feed, and where they are")
    add("")
    add("Data root `{}`, one directory per family, one per configuration inside it,".format(bench))
    add("`r<k>_stats.csv` and `r<k>_latencies.csv` per run, with the run's own log beside them.")
    add("Regenerate the tables with `analysis/tables/results-tables.py`; regenerate any family with")
    add("`measure/sweep.sh --exp <family>`, or a whole group with `./pipeline.py`.")
    add("")
    add("| family | key | configurations | runs each | stored runs | first | last | feeds |")
    add("|---|---|---|---|---|---|---|---|")
    for r in rows:
        add("| `{}` | {} | {} | {} | {} | {} | {} | {} |".format(
            r["family"], r["key"], r["configs"], r["each"], r["stored"],
            r["first"], r["last"], r["feeds"]))
    add("| **total** | | **{}** | | **{}** | | | |".format(
        sum(r["configs"] for r in rows), sum(r["stored"] for r in rows)))
    add("")
    add("## The families the results section reads, and the tables they feed")
    add("")
    add("| key | families | tables |")
    add("|---|---|---|")
    for key in sorted({keys[f] for f in section if f in keys}):
        fams = sorted(f for f in section if keys.get(f) == key)
        add("| {} | {} | {} |".format(key, ", ".join("`%s`" % f for f in fams),
                                      TABLE_OF.get(key, "---")))
    add("")
    add("## Every configuration, with its runs")
    add("")
    for r in rows:
        add("### `{}` ({}, {})".format(r["family"], r["key"], r["feeds"]))
        add("")
        add("| configuration | runs | networks with a stats file |")
        add("|---|---|---|")
        for cfg, n, nets in configurations(bench, r["family"]):
            add("| `{}` | {} | {} |".format(cfg, n, ", ".join(nets) if nets else "---"))
        add("")

    text = "\n".join(lines) + "\n"
    dest = bench / "OVERVIEW.md"
    dest.write_text(text)
    print(text)
    sys.stderr.write("  wrote {}\n".format(dest))


if __name__ == "__main__":
    main()
