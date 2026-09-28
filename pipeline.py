#!/usr/bin/env python3
"""The one entry point: measure, reduce, render tables, write the overview.

Interactive when run bare, scriptable with the same words:

    ./pipeline.py                       menu: what is measured, then what to run
    ./pipeline.py section               the families the results section renders
    ./pipeline.py appendix              the rest, which the history and the tools read
    ./pipeline.py mem --runs 3          one group, fewer replicates
    ./pipeline.py mem-s2                a single family
    ./pipeline.py --tables              regenerate and split the tables, measure nothing
    ./pipeline.py all --stage measure   stop after measuring
    ./pipeline.py section --traces --soc    also take the kernel traces and the DRAM counters

Stages, in order, each skippable with --stage:
    engines   rebuild every TensorRT engine from its ONNX file
    measure   utility/sweep.sh on the selection
    reduce    utility/aggregate.py
    tables    both generators, split into one file per table
    overview  utility/overview.py

The measurement is never started without the sweep's own guard, which the pipeline runs first so
a free governor is reported before a warm-up rather than after it.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "utility"))

from lib import families  # noqa: E402

STAGES = ("engines", "measure", "reduce", "tables", "overview")


def say(text=""):
    print(text, flush=True)


def run(cmd, dry=False, capture_to=None):
    say("  $ " + " ".join(str(c) for c in cmd))
    if dry:
        return 0
    if capture_to:
        with open(capture_to, "w") as fh:
            return subprocess.call([str(c) for c in cmd], cwd=ROOT, stdout=fh)
    return subprocess.call([str(c) for c in cmd], cwd=ROOT)


def state_table():
    """The groups and what they hold, with the families that are not measured at all."""
    bench = families.bench_root()
    known = families.groups()
    say("  data root:  {}".format(families.data_root()))
    say("  on disk:    {} families".format(len(families.families_on_disk(bench))))
    say()
    say("  {:<10} {:>7} {:>9} {:>11}  {}".format(
        "group", "families", "configs", "stored runs", "measured"))
    for name in ("section", "appendix", "mem", "all"):
        s = families.group_state(name, bench)
        if not s:
            say("  {:<10} {:>7}".format(name, len(known.get(name, []))) + "   not measured")
            continue
        missing = [f for f in known.get(name, []) if not families.state(bench, f)]
        say("  {:<10} {:>7} {:>9} {:>11}  {}{}".format(
            name, len(known.get(name, [])), s["configs"], s["stored"],
            "{} -> {}".format(s["first"], s["last"]),
            ("   missing: " + ", ".join(missing)) if missing else ""))
    say()


def clock_state():
    """The GPU governor, read from the same directory the sweep refuses on."""
    dev = families.gpu_devfreq()
    if dev is None:
        return None, "?", "?"
    try:
        lo = (dev / "min_freq").read_text().strip()
        hi = (dev / "max_freq").read_text().strip()
    except OSError:
        return None, "?", "?"
    return lo == hi, lo, hi


def group_help(known):
    return ", ".join("{} ({})".format(k, len(v)) for k, v in known.items() if k != "all") + ", all"


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("selection", nargs="*",
                   help="a group (section, appendix, mem, all), a family, or nothing for the menu")
    p.add_argument("--stage", default="all",
                   help="all, or one of: " + ", ".join(STAGES))
    p.add_argument("--runs", type=int, default=5, help="replicates per configuration (default 5)")
    p.add_argument("--static-chunks", type=int, default=8,
                   help="enable the MPS static arms (default 8; 0 disables them)")
    p.add_argument("--smoke", action="store_true", help="one fast run per configuration, not a measurement")
    p.add_argument("--force", action="store_true", help="re-measure configurations whose output exists")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--traces", action="store_true", help="also take the kernel traces and reduce them")
    p.add_argument("--soc", action="store_true", help="also take the DRAM counters (needs root)")
    p.add_argument("--state", action="store_true", help="print what is measured and stop")
    return p.parse_args()


def main():
    args = parse()
    known = families.groups()
    bench = families.bench_root()

    if args.state:
        state_table()
        return 0

    selection = ",".join(args.selection) if args.selection else None
    if selection is None:
        state_table()
        say("  groups: " + group_help(known))
        say()
        try:
            selection = input("  which? [section/appendix/mem/all/<family>] ").strip() or "section"
        except EOFError:
            selection = "section"
        stages = input("  stages [{}] (space-separated, or 'all') ".format(" ".join(STAGES))).strip()
        if stages:
            wanted = tuple(stages.split())
        else:
            wanted = STAGES
        if "all" in wanted:
            wanted = STAGES
    else:
        wanted = STAGES if args.stage == "all" else tuple(args.stage.split(","))

    for stage in wanted:
        if stage not in STAGES:
            say("  unknown stage '{}'; expected one of {}".format(stage, ", ".join(STAGES)))
            return 2

    chosen = families.expand(selection)
    say()
    say("  selection:  {} ({})".format(selection, " ".join(chosen)))
    say("  stages:     " + " ".join(wanted))
    if not args.dry_run and not args.yes:
        answer = input("  proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            say("  nothing done")
            return 1

    if "measure" in wanted:
        ok, lo, hi = clock_state()
        if ok is False:
            say("  the GPU governor is free ({} to {}): run `sudo jetson_clocks` first".format(lo, hi))
            if not args.dry_run:
                return 1
        elif ok is None:
            say("  (could not read the GPU governor; the sweep checks it again)")

    if "engines" in wanted:
        say("\n== engines ==")
        run(["make", "-f", "engines.mk"], args.dry_run)

    if "measure" in wanted:
        say("\n== measure ==")
        cmd = ["./utility/sweep.sh", "--exp", selection, "--runs", str(args.runs)]
        if args.static_chunks:
            cmd += ["--static-chunks", str(args.static_chunks)]
        if args.smoke:
            cmd.append("--smoke")
        if args.force:
            cmd.append("--force")
        if args.dry_run:
            cmd.append("--dry-run")
        if run(cmd, args.dry_run) != 0 and not args.dry_run:
            return 1

    if "reduce" in wanted:
        say("\n== reduce ==")
        if run([sys.executable, "utility/aggregate.py", "--out", str(bench)], args.dry_run) != 0:
            return 1

    if "tables" in wanted:
        say("\n== tables ==")
        out = bench / "aggregate"
        out.mkdir(parents=True, exist_ok=True)
        if run([sys.executable, "utility/results-tables.py"], args.dry_run,
               capture_to=out / "tables.tex") != 0:
            return 1
        if (ROOT / "utility/tex-tables.py").exists():
            run([sys.executable, "utility/tex-tables.py"], args.dry_run,
                capture_to=out / "tables-history.tex")
        if run([sys.executable, "utility/split-tables.py", str(out / "tables.tex"),
                "--out", str(out / "tables")], args.dry_run) != 0:
            return 1
        if (out / "tables-history.tex").exists() and not args.dry_run:
            run([sys.executable, "utility/split-tables.py", str(out / "tables-history.tex"),
                 "--out", str(out / "tables-history")])

    if "overview" in wanted:
        say("\n== overview ==")
        run([sys.executable, "utility/overview.py"], args.dry_run)

    if args.traces:
        say("\n== kernel traces ==")
        run(["./utility/take-traces.sh"], args.dry_run)
        run([sys.executable, "utility/itf-trace.py",
             str(families.data_root() / "scratch/traces/trc")], args.dry_run)

    if args.soc:
        say("\n== DRAM counters (will ask for your sudo password) ==")
        run(["sudo", "./utility/soc-traces.sh", "itf"], args.dry_run)

    say("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
