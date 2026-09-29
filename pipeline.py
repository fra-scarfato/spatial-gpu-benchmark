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

# What a default run does: quick, and measurement only.
STAGES = ("engines", "measure", "reduce", "tables", "overview")

# The stages, in the two groups the pipeline distinguishes. A measurement produces the numbers the
# thesis reports; a diagnostic profiles what it measures and therefore perturbs it, so its output
# is evidence about mechanisms and not a latency. Four diagnostics need root.
STAGES_MEASUREMENT = STAGES
STAGES_DIAGNOSTIC = ("traces", "soc", "soc-budget", "ncu", "probes")
EXTRA_STAGES = STAGES_DIAGNOSTIC


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
                   help="all (the default set), or any of: " + ", ".join(STAGES + EXTRA_STAGES))
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
        say("  measurement stages: {}   (default)".format(" ".join(STAGES)))
        say("  measurement, long:  {}".format(" ".join(STAGES_MEASUREMENT[len(STAGES):])))
        say("  diagnostics:        {}   (perturb what they measure; 4 need root)"
            .format(" ".join(STAGES_DIAGNOSTIC)))
        stages = input("  stages, space-separated, or 'all' for the default set ").strip()
        if stages:
            wanted = tuple(stages.split())
        else:
            wanted = STAGES
        if "all" in wanted:
            wanted = STAGES
    else:
        wanted = STAGES if args.stage == "all" else tuple(args.stage.split(","))

    for stage in wanted:
        if stage not in STAGES + EXTRA_STAGES:
            say("  unknown stage '{}'; expected one of {}".format(
                stage, ", ".join(STAGES + EXTRA_STAGES)))
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
        # The additivity family has a second half the sweep cannot run: three clients need three
        # partitions and the job spec has room for two. It is a measurement of the same family, so
        # it belongs to this stage and not to a stage of its own.
        if "interference-additivity" in chosen:
            say("\n   and its MPS static half (three partitions, three processes) ==")
            models = families.data_root() / "models"
            specs = [str(models / "mobilenetv2-12/engine/mobilenetv2-12.engine") + ":mobilenet",
                     str(models / "retinanet-9/engine/retinanet-9.engine") + ":retinanet",
                     str(models / "inception_v3-onnx-float/engine/inception_v3.engine") + ":inception"]
            if run(["./utility/mps-additivity-block.sh", "--block", str(args.runs)] + specs,
                   args.dry_run) != 0:
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

    if "probes" in wanted:
        say("\n== the probes ==")
        run(["make", "-C", "utility"], args.dry_run)
        for probe in ("auxprobe", "wqprobe", "devprobe"):
            run(["./utility/" + probe], args.dry_run)

    if "soc-budget" in wanted:
        say("\n== the SoC pressure curve, one network at a time (needs root) ==")
        models = families.data_root() / "models"
        engines = {"mobilenetv2-12": "mobilenetv2-12/engine/mobilenetv2-12.engine",
                   "resnet50-v2-7": "resnet50-v2-7/engine/resnet50-v2-7.engine",
                   "inception_v3-onnx-float": "inception_v3-onnx-float/engine/inception_v3.engine",
                   "retinanet-9": "retinanet-9/engine/retinanet-9.engine"}
        for model, path in engines.items():
            for pct in (100, 75, 50, 25):
                run(["sudo", "./utility/soc-budget-sweep.sh", "greenctx", str(pct),
                     "{}:{}".format(models / path, pct)], args.dry_run)
        say("  the same points under MPS static, to check the mechanism does not move them:")
        for spec in ("mobilenetv2-12:50", "retinanet-9:50", "mobilenetv2-12:25"):
            model, pct = spec.split(":")
            run(["sudo", "./utility/soc-budget-sweep.sh", "mps-static", pct,
                 "{}:{}".format(models / engines[model], pct)], args.dry_run)
        for a, b, pa, pb in (("mobilenetv2-12", "resnet50-v2-7", 50, 50),
                             ("mobilenetv2-12", "resnet50-v2-7", 50, 25),
                             ("mobilenetv2-12", "retinanet-9", 50, 25)):
            run(["sudo", "./utility/soc-budget-sweep.sh", "greenctx", str(pa),
                 "{}:{}".format(models / engines[a], pa),
                 "{}:{}".format(models / engines[b], pb)], args.dry_run)

    if "soc" in wanted:
        say("\n== the SoC counters on the interferer pairs (needs root) ==")
        run(["sudo", "./utility/soc-traces.sh", "itf"], args.dry_run)

    if "ncu" in wanted:
        say("\n== ncu: does it work here, then the traffic per inference (needs root) ==")
        models = families.data_root() / "models"
        mobile = models / "mobilenetv2-12/engine/mobilenetv2-12.engine"
        # no protocol bounds in this list: each command adds its own, because slicing it is how a
        # flag and its value ended up on opposite sides of a cut and the runner threw stod
        base = ["./runner", "--submit-threads", "1", "--dataset",
                str(families.data_root() / "dataset/imagenet-1k"), "--fixed", "32"]
        check = ["--min-inferences", "200", "--min-duration", "2"]
        check_m = ["--min-inferences", "600", "--min-duration", "3"]
        run(["sudo", "/usr/local/cuda/bin/ncu", "--launch-count", "1",
             "--metrics", "dram__bytes.sum"] + base + check +
            ["--allocation", "whole", "--net", "{}:100:mobilenet".format(mobile),
             "--csv", "/dev/shm/ncu-check"], args.dry_run)
        run(["sudo", "/usr/local/cuda/bin/ncu", "--launch-count", "1",
             "--metrics", "dram__bytes.sum"] + base + check +
            ["--allocation", "greenctx", 
             "--net", "{}:50:mobilenet".format(mobile), "--csv", "/dev/shm/ncu-check"], args.dry_run)
        say("  if both pass, the per-inference traffic, per network and budget:")
        for path, pct in ((mobile, 50), (mobile, 25),
                          (models / "retinanet-9/engine/retinanet-9.engine", 50),
                          (models / "retinanet-9/engine/retinanet-9.engine", 25),
                          (models / "inception_v3-onnx-float/engine/inception_v3.engine", 50),
                          (models / "inception_v3-onnx-float/engine/inception_v3.engine", 25),
                          (models / "resnet50-v2-7/engine/resnet50-v2-7.engine", 50),
                          (models / "resnet50-v2-7/engine/resnet50-v2-7.engine", 25)):
            run(["sudo", "/usr/local/cuda/bin/ncu", "--launch-skip", "60", "--launch-count", "400",
                 "--metrics", "dram__bytes_read.sum,dram__bytes_write.sum", "--csv",
                 "--log-file", "/tmp/ncu-{}-{}.csv".format(Path(str(path)).stem, pct)] + base + check_m +
                ["--allocation", "greenctx",
                 
                 "--net", "{}:{}:{}".format(path, pct, Path(str(path)).stem),
                 "--csv", "/dev/shm/ncu"], args.dry_run)


    say("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
