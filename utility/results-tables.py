#!/usr/bin/env python3
"""results-tables.py — the tables of the Results section, from the measurements.

ONE SHAPE EVERYWHERE. First column the NETWORK, second the CONFIGURATION, then the
metrics. (The comparison tables add the mechanism as a third column before the
metrics, so one network can be read down all technologies.) The network's name is
printed on every row where it CHANGES, so it heads a block when a block is one
network, and repeats when the rows alternate.

    Network | Configuration | SMs | Median (ms) | Thr. (inf/s)
    Network | Configuration | std | std xiso | p95 | p99 | tail | tail xiso | obs-max

Submission model, frozen by the submission-model section: one submitting thread.
Isolation and MPS have one process per network, so one submitter by construction;
green context is measured as exp2-1t.

Run from anywhere: it imports aggregate.py from its own directory and reads the
results tree under $DATA. No number is typed by hand.
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as A

DATA = A.load(os.path.join(A.DATA_ROOT, "bench"))
if not DATA:
    sys.exit("results-tables.py: no measurements found under $DATA/bench")

OUT = []


def emit(line=""):
    OUT.append(line)


def ms(v):
    return "---" if v is None else "{:.4f}".format(v)


def xr(v, ref):
    return "---" if (v is None or not ref) else "{:.3f}".format(v / ref)


def val(exp, cfg, net, col):
    r = A.aggr(DATA, exp, cfg, net, "gpu_ms", col)
    return r[0] if isinstance(r, tuple) else r


def thr(exp, cfg, net):
    """Measured inferences per second: count over the window the runner timed.

    Not 1000/median, which is the GPU-only figure and reads higher, because it
    ignores the gap the host leaves between inferences.
    """
    vals = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        rec = DATA[(exp, cfg, r)].get(net)
        if rec and "gpu_ms" in rec and rec["gpu_ms"].get("duration_s", 0) > 0:
            vals.append(rec["gpu_ms"]["count"] / rec["gpu_ms"]["duration_s"])
    return st.median(vals) if vals else None


def tails(exp, cfg, net):
    vals = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        rec = DATA[(exp, cfg, r)].get(net)
        if rec and "gpu_ms" in rec and "p99_minus_median" in rec["gpu_ms"]:
            vals.append(rec["gpu_ms"]["p99_minus_median"])
    return vals


def family(exp, cfg, desc):
    """The family a row belongs to. A \\midrule marks a change of family, so this is
    what the rules mean; a change of network means nothing unless the network's rows
    form a block."""
    if exp == "exp1":
        return "isolated"
    if exp == "exp2-1t":
        return "unpartitioned" if cfg == "baseline" else "partitioned"
    if exp == "exp3":
        if cfg in ("mps_single", "mps_off", "mps_default"):
            return "references"
        return "atp" if cfg.startswith("mps_atp") else "static"
    if exp == "exp7":
        return "mps" if cfg == "mps_owncores" else "in-process"
    if exp == "":
        return "references"
    return desc.split(",")[0]


def ref(net):
    """The isolated reference for xiso: the network alone on the whole device."""
    return (val("exp1", net + "_whole", net, "p99_minus_median"),
            val("exp1", net + "_whole", net, "std"))


# The caption is emitted by close_table, below the tabular. Above a table LaTeX
# separates the caption with \belowcaptionskip, which is 0pt by default, so a caption
# placed there sits against the top rule; below the tabular it is separated by
# \abovecaptionskip, 10pt. It is also the convention the rest of the thesis uses.
_CAPTION = ""


def open_table(cols, spec, caption, label):
    global _CAPTION
    _CAPTION = "\\caption{%s}\n\\label{%s}" % (caption, label)
    emit(r"\begin{table}[H]")
    emit(r"\centering")
    emit(r"\footnotesize")
    emit(r"\setlength{\tabcolsep}{4pt}")
    emit(r"\begin{tabular}{%s}" % spec)
    emit(r"\toprule")
    emit(cols)
    emit(r"\midrule")


def close_table():
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    for line in _CAPTION.split("\n"):
        emit(line)
    emit(r"\end{table}")
    emit()


def separators(rows, by_network):
    """Which row indices should be preceded by a rule. With by_network, a network change
    is the boundary and the network's rows are a block; otherwise the family is, and a
    change of network means nothing."""
    marks = []
    prev_net = prev_fam = None
    for i, (net, desc, exp, cfg, sms) in enumerate(rows):
        fam = family(exp, cfg, desc)
        mark = False
        if prev_net is not None:
            if by_network:
                mark = net != prev_net
            else:
                mark = fam != prev_fam
        marks.append(mark)
        prev_net, prev_fam = net, fam
    return marks


def rates(rows, by_network=False):
    """rows: (net, desc, exp, cfg, sms). Column 1 is the network on every row."""
    marks = separators(rows, by_network)
    for mark, (net, desc, exp, cfg, sms) in zip(marks, rows):
        if mark:
            emit(r"\midrule")
        emit(r"{} & {} & {} & {} & {} \\".format(
            net, desc, "---" if sms is None else str(sms),
            ms(val(exp, cfg, net, "median")), "{:.0f}".format(thr(exp, cfg, net) or 0)))


def dispersion(rows, by_network=False):
    marks = separators(rows, by_network)
    for mark, (net, desc, exp, cfg, sms) in zip(marks, rows):
        if mark:
            emit(r"\midrule")
        cells = [net, desc]
        t, s = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} \\".format(
            " & ".join(cells), ms(std), xr(std, s), ms(val(exp, cfg, net, "p95")),
            ms(val(exp, cfg, net, "p99")), ms(tail), xr(tail, t),
            ms(A.obs_max(DATA, exp, cfg, net))))


# ------------------------------------------------------------------ isolation
# One network alone, varying the SM budget. MobileNet and ResNet have the whole
# range; the batch networks were only ever measured on the whole device.
ISO = []
for net in ("mobilenet", "resnet"):
    ISO.append((net, "whole device, no partitioning", "exp1", net + "_whole", 16))
    for pct, sm in ((100, 16), (75, 12), (50, 8), (25, 4)):
        ISO.append((net, "green context, %d\\,\\%% of the device" % pct, "exp1",
                    "%s_gc%d" % (net, pct), sm))
# Inception, ViT and RetinaNet are deliberately absent: they belong to the
# auxiliary-stream work. This section is about two networks.

open_table(r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\", "llrrr",
           "The five networks in isolation: latency and throughput against the size of "
           "the green context. Percentages are of the device's sixteen SMs. Whole device "
           "and a 100\\,\\% green context are the same reservation, measured twice, so "
           "the pair also shows the run-to-run spread of a single configuration.", "tab:iso-rates")
rates(ISO, by_network=True)
close_table()

open_table(r"Network & Configuration & std & std $\times$iso & p95 & p99 & tail & "
           r"tail $\times$iso & obs-max \\", "llrrrrrrr",
           "The same builds by dispersion. \\emph{tail} is $p99-\\mathrm{median}$; "
           "$\\times$iso is against the network's own whole-device run.",
           "tab:iso-dispersion")
dispersion(ISO, by_network=True)
close_table()

# ------------------------------------------ two networks on green contexts
PAIR_EXP = "exp2-1t"
PAIRS = [("baseline", "no partitioning, one context, whole device", 16, 16),
         ("50_50", "two green contexts, 50\\,\\% / 50\\,\\%", 8, 8),
         ("75_25", "two green contexts, 75\\,\\% / 25\\,\\%", 12, 4),
         ("25_75", "two green contexts, 25\\,\\% / 75\\,\\%", 4, 12)]
PAIR = []
for cfg, desc, mob_sm, res_sm in PAIRS:
    for net, sm in (("mobilenet", mob_sm), ("resnet", res_sm)):
        PAIR.append((net, desc, PAIR_EXP, cfg, sm))

open_table(r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\", "llrrr",
           "Two networks on disjoint green contexts, one submitting thread.",
           "tab:two-rates")
rates(PAIR)
close_table()

open_table(r"Network & Configuration & std & std $\times$iso & p95 & p99 & tail & "
           r"tail $\times$iso & obs-max \\", "llrrrrrrr",
           "The pair by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:two-dispersion")
dispersion(PAIR)
close_table()

# ------------------------------------------------------------------------ MPS
MPS = [("mps_single", "one process under MPS, whole device", None, None, True),
       ("mps_off", "two plain processes, no MPS (isolation)", 16, 16, False),
       ("mps_default", "one context, both clients (MPS default)", 16, 16, False),
       ("mps_atp_50_50", "MPS ATP, 50\\,\\% thread occupancy each", None, None, False),
       ("mps_atp_75_25", "MPS ATP, 75\\,\\% / 25\\,\\%", None, None, False),
       ("mps_atp_25_75", "MPS ATP, 25\\,\\% / 75\\,\\%", None, None, False),
       ("mps_static_50_50", "MPS static partitions, 50\\,\\% / 50\\,\\%", 8, 8, False),
       ("mps_static_75_25", "MPS static partitions, 75\\,\\% / 25\\,\\%", 12, 4, False),
       ("mps_static_25_75", "MPS static partitions, 25\\,\\% / 75\\,\\%", 4, 12, False)]
MPSROWS = []
for cfg, desc, mob_sm, res_sm, solo in MPS:
    nets = (("mobilenet", mob_sm),) if solo else (("mobilenet", mob_sm), ("resnet", res_sm))
    for net, sm in nets:
        MPSROWS.append((net, desc, "exp3", cfg, sm))

open_table(r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\", "llrrr",
           "Two processes under MPS, one submitter per process. \\texttt{mps\\_single} is "
           "one client with nothing to share the device with. ATP caps thread occupancy, "
           "which is not an SM reservation, so it has no SM count.",
           "tab:mps-rates")
rates(MPSROWS)
close_table()

open_table(r"Network & Configuration & std & std $\times$iso & p95 & p99 & tail & "
           r"tail $\times$iso & obs-max \\", "llrrrrrrr",
           "The MPS arms by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:mps-dispersion")
dispersion(MPSROWS)
close_table()

# --------------------------------------------------------- green context vs MPS
# Grouped BY NETWORK, so one network's whole comparison is one block: the mechanism
# varies down the block and the ratio is in the configuration column. The isolated
# row is the floor for the same SM budget, which is the honest reference: what that
# network costs when its share is its own.
FLOOR = {("50_50", "mobilenet"): ("mobilenet_gc50", 8),
         ("50_50", "resnet"): ("resnet_gc50", 8),
         ("75_25", "mobilenet"): ("mobilenet_gc75", 12),
         ("75_25", "resnet"): ("resnet_gc25", 4),
         ("25_75", "mobilenet"): ("mobilenet_gc25", 4),
         ("25_75", "resnet"): ("resnet_gc75", 12)}
RATIOS = [("50_50", "50\\,\\% / 50\\,\\%"), ("75_25", "75\\,\\% / 25\\,\\%"),
          ("25_75", "25\\,\\% / 75\\,\\%")]
# The share each network gets, by ratio. Not symmetric: ResNet takes the complement,
# and getting this wrong is silent.
SHARE = {("50_50", "mobilenet"): 8, ("50_50", "resnet"): 8,
         ("75_25", "mobilenet"): 12, ("75_25", "resnet"): 4,
         ("25_75", "mobilenet"): 4, ("25_75", "resnet"): 12}

CMP = []
for net in ("mobilenet", "resnet"):
    for cfg, ratio in RATIOS:
        CMP.append((net, "green context, " + ratio, PAIR_EXP, cfg, SHARE[(cfg, net)]))
    for cfg, ratio in RATIOS:
        CMP.append((net, "MPS static, " + ratio, "exp3", "mps_static_" + cfg, SHARE[(cfg, net)]))
    for cfg, ratio in RATIOS:
        CMP.append((net, "MPS ATP, " + ratio, "exp3", "mps_atp_" + cfg, None))
    CMP.append((net, "MPS default, one context", "exp3", "mps_default", 16))
    CMP.append((net, "no MPS, two plain processes", "exp3", "mps_off", 16))
    for cfg, ratio in RATIOS:
        iso_cfg, sm = FLOOR[(cfg, net)]
        CMP.append((net, "alone on %d SMs (%s)" % (sm, ratio), "exp1", iso_cfg, sm))

open_table(r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\",
           "llrrr",
           "Green context against MPS static, MPS ATP and MPS default, at matched "
           "ratios, with one submitting thread. The isolation rows are the floor: the "
           "same network alone on the same share of the device.",
           "tab:compare-rates")
rates(CMP)
close_table()

open_table(r"Network & Configuration & std & std $\times$iso & p95 & p99 & "
           r"tail & tail $\times$iso & obs-max \\", "llrrrrrrr",
           "The comparison by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:compare-dispersion")
dispersion(CMP)
close_table()

# --------------------------------------------------------------- the CPU side
CPU = [("t1_unpinned", "one submitter, no affinity"),
       ("t1_pinfifo99", "one submitter, core 6, SCHED\\_FIFO 99"),
       ("t2_unpinned", "two submitters, no affinity (one thread per network)"),
       ("t2_core67", "two submitters, allowed cores 6 and 7"),
       ("mps_owncores", "two processes, one on core 6 and one on core 7")]

open_table(r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\", "llrrr",
           "Two networks at 75\\,\\% / 25\\,\\% with the submission model as the only "
           "variable: green context, and the last row under MPS static.",
           "tab:cpu-rates")
for cfg, desc in CPU:
    for net in ("mobilenet", "resnet"):
        if cfg == "mps_owncores" and net == "mobilenet":
            emit(r"\midrule")
        emit(r"{} & {} & {} & {} & {:.0f} \\".format(
            net, desc, 12 if net == "mobilenet" else 4,
            ms(val("exp7", cfg, net, "median")), thr("exp7", cfg, net) or 0))
close_table()

open_table(r"Network & Configuration & tail (ms) & the five run-level tails & "
           r"tail $\times$iso \\", "llrrr",
           "The same arms by tail, with every run's tail listed. The two-submitter arms "
           "are not stable, which is what the column is there to show.",
           "tab:cpu-dispersion")
for cfg, desc in CPU:
    for net in ("mobilenet", "resnet"):
        if cfg == "mps_owncores" and net == "mobilenet":
            emit(r"\midrule")
        t, _ = ref(net)
        tail = val("exp7", cfg, net, "p99_minus_median")
        runs = tails("exp7", cfg, net)
        emit(r"{} & {} & {} & {} & {} \\".format(
            net, desc,
            ms(tail), " / ".join("{:.4f}".format(v) for v in runs) or "---",
            xr(tail, t)))
close_table()

sys.stdout.write("\n".join(OUT))
