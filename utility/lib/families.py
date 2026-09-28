"""Where the experiments are, what they are called, and what state they are in.

Two things are read rather than repeated: the family-to-key map from `aggregate.py`, which is
where the reduction uses it, and the group lists from `sweep.sh`, which is where the sweep uses
them. A copy of either would be a copy that can disagree with the thing that matters.

    from lib import families
    families.groups()                 # {'section': [...], 'appendix': [...], ...}
    families.state(bench, "memory-interferers")   # configurations, replicates, dates
"""
import datetime
import os
import re
from pathlib import Path

UTILITY = Path(__file__).resolve().parent.parent
ROOT = UTILITY.parent
DEFAULT_DATA = "/mnt/nvme/f.scarfato"

GROUP_VARS = {"section": "SECTION_FAMILIES", "appendix": "APPENDIX_FAMILIES",
              "mem": "MEM_FAMILIES", "all": "ALL_FAMILIES"}


def data_root():
    return Path(os.environ.get("DATA", DEFAULT_DATA))


def bench_root():
    return data_root() / "bench"


# The family -> aggregate key map. Two folders share a key when their rows belong in the
# same tables: concurrent-wq is the work-queue half of concurrent-greenctx's setup, and
# the memory families all feed the same tables, so the tables see one configuration space.
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
    "workqueue-declaration":                "exp8",
    "workqueue-three-clients":                "exp8",
    "memory-budget-greenctx":               "exp9",
    "memory-interferers":               "exp9",
    "memory-mps-interferers":               "exp9",
    "memory-mps-budget":               "exp9",
    "memory-mps-interferers-4sm":               "exp9",
    "memory-two-threads":               "exp9",
    "workqueue-aux-streams": "exp10",
    "repro":               "exp6",
    "concurrent-mps":      "exp3",
}

def key_map():
    """The family -> aggregate key map."""
    return dict(EXP_DIR)


def groups():
    """The named family groups, from sweep.sh.

    One level of $VAR reference is resolved, because ALL_FAMILIES is defined in terms of the
    other two: reading it literally would yield the two variable names as if they were families.
    """
    src = (UTILITY / "sweep.sh").read_text()
    var_of = {v: k for k, v in GROUP_VARS.items()}
    raw = {}
    for name, var in GROUP_VARS.items():
        m = re.search(var + r'="([^"]*)"', src)
        raw[name] = m.group(1) if m else ""

    def resolve(text, depth=4):
        for _ in range(depth):
            new = re.sub(r"\$([A-Z_]+)",
                         lambda m: raw.get(var_of.get(m.group(1), ""), ""), text)
            if new == text:
                break
            text = new
        return text

    return {name: resolve(text).split() for name, text in raw.items()}


def gpu_devfreq():
    """The GPU governor directory, taken from the sweep so the two check the same thing."""
    src = (UTILITY / "sweep.sh").read_text()
    m = re.search(r'^GPU_DEVFREQ=(.+)$', src, re.M)
    if not m:
        return None
    value = m.group(1).strip()
    inner = re.search(r":-([^}]+)}", value)     # ${GPU_DEVFREQ:-/sys/...}
    if inner:
        value = inner.group(1)
    value = value.strip().strip('"').strip("'")
    return Path(value) if value.startswith("/") else None


def expand(selection):
    """A group name, a family name, or a comma-separated list of either."""
    known = groups()
    out = []
    for token in str(selection).split(","):
        token = token.strip()
        if token in known:
            out.extend(known[token])
        else:
            out.append(token)
    return out


def families_on_disk(bench=None):
    bench = bench or bench_root()
    if not bench.is_dir():
        return []
    return sorted(p.name for p in bench.iterdir()
                  if p.is_dir() and p.name not in ("aggregate", "trace"))


def state(bench, family):
    """Configurations, the replicate count they hold, and the dates they were measured.

    Counted by run index, not by stats file: a two-network arm writes one file per network per
    run, so counting files would double it.
    """
    d = Path(bench) / family
    if not d.is_dir():
        return None
    cfgs, per_config, dates = [], [], set()
    for cfg in sorted(p for p in d.iterdir() if p.is_dir()):
        files = sorted(cfg.glob("r*_stats.csv"))
        if not files:
            continue
        cfgs.append(cfg.name)
        per_config.append(len({f.name.split("_")[0] for f in files}))
        for f in files:
            dates.add(datetime.date.fromtimestamp(f.stat().st_mtime))
    if not cfgs:
        return None
    return dict(family=family, key=key_map().get(family, "?"), configs=len(cfgs),
                each=max(per_config), stored=sum(per_config),
                first=min(dates).isoformat(), last=max(dates).isoformat())


def group_state(selection, bench=None):
    """The same, aggregated over a selection."""
    bench = Path(bench or bench_root())
    rows = [state(bench, fam) for fam in expand(selection)]
    rows = [r for r in rows if r]
    if not rows:
        return None
    return dict(families=len(rows), configs=sum(r["configs"] for r in rows),
                stored=sum(r["stored"] for r in rows),
                first=min(r["first"] for r in rows), last=max(r["last"] for r in rows),
                rows=rows)
