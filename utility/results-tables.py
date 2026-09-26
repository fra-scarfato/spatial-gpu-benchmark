#!/usr/bin/env python3
"""results-tables.py — the tables of the Results section, from the measurements.

Two layouts, each used where it reads best.

NETWORK FIRST, for the tables that describe one technology:

    Network | Configuration | SMs | Median (ms) | Thr. (inf/s)
    Network | Configuration | std | std xiso | p95 | p99 | tail | tail xiso | obs-max

Four-way, for the comparison tables, because there the question is what one mechanism
does to one network at one split, and that is a row:

    Ratio | Mechanism | MobileNet (median, thr) | ResNet (median, thr) | Aggregate
    Ratio | Mechanism | Network | std | xiso | p95 | p99 | tail | xiso | obs-max

The Configuration column always names the mechanism — GC, MPS static, MPS ATP, MPS
default, no MPS, or whole/alone for the unpartitioned and isolated references — so no
row has to be read against a caption to know what it is.

Submission model, frozen by the submission-model section: one submitting thread.
isolation and MPS have one process per network and therefore one submitter by
construction; green context is measured as exp2-1t.

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
_SMS = 16
_CAPTION = ""


def emit(line=""):
    OUT.append(line)


def ms(v):
    return "---" if v is None else "{:.4f}".format(v)


def xr(v, ref):
    return "---" if (v is None or not ref) else "{:.3f}".format(v / ref)


def val(exp, cfg, net, col):
    r = A.aggr(DATA, exp, cfg, net, "gpu_ms", col)
    return r[0] if isinstance(r, tuple) else r


def runs_of(exp, cfg, net):
    out = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        rec = DATA[(exp, cfg, r)].get(net)
        if rec and "gpu_ms" in rec:
            out.append(rec["gpu_ms"])
    return out


def thr(exp, cfg, net):
    """Measured inferences per second: count over the window the runner timed.

    Not 1000/median, which is the GPU-only figure and reads higher because it ignores
    the gap the host leaves between inferences.
    """
    vals = [m["count"] / m["duration_s"] for m in runs_of(exp, cfg, net)
            if m.get("duration_s", 0) > 0]
    return st.median(vals) if vals else None


def agg_thr(exp, cfg, nets):
    """Aggregate throughput: both networks' inferences over the window they shared.

    max(duration) is that window, because the processes start together. Dividing by
    the sum, or by the shorter duration, would invent throughput that never happened.
    """
    vals = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        total, longest = 0, 0.0
        for net in nets:
            rec = DATA[(exp, cfg, r)].get(net)
            if not rec or "gpu_ms" not in rec or rec["gpu_ms"].get("duration_s", 0) <= 0:
                total = 0
                break
            total += rec["gpu_ms"]["count"]
            longest = max(longest, rec["gpu_ms"]["duration_s"])
        if total and longest:
            vals.append(total / longest)
    return st.median(vals) if vals else None


def ref(net):
    """The isolated reference for xiso: the network alone on the whole device."""
    return (val("exp1", net + "_whole", net, "p99_minus_median"),
            val("exp1", net + "_whole", net, "std"))


def family(exp, cfg, desc):
    """The family a row belongs to. A rule marks a change of family, so this is what
    the rules mean: a change of network means nothing unless the network's rows form a
    block."""
    if exp == "exp1":
        return "isolated"
    if exp == "exp2-1t":
        return "unpartitioned" if cfg == "baseline" else "greenctx"
    if exp == "exp3":
        if cfg in ("mps_single", "mps_off", "mps_default"):
            return "references"
        return "atp" if cfg.startswith("mps_atp") else "static"
    if exp == "exp7":
        return "mps" if cfg == "mps_owncores" else "in-process"
    return desc


def open_table(cols, spec, caption, label):
    """The caption goes BELOW the tabular. Above a table LaTeX separates it with
    \\belowcaptionskip, which is 0pt by default, so it sits against the top rule;
    below, it is separated by \\abovecaptionskip, 10pt. It is also what the rest of
    the thesis does."""
    global _CAPTION
    _CAPTION = "\\caption{%s}\n\\label{%s}" % (caption, label)
    emit(r"\begin{table}[H]")
    emit(r"\centering")
    emit(r"\footnotesize")
    emit(r"\setlength{\tabcolsep}{4pt}")
    emit(r"\begin{tabular}{%s}" % spec)
    emit(r"\toprule")
    for c in cols:
        emit(c)
    emit(r"\midrule")


def close_table():
    emit(r"\bottomrule")
    emit(r"\end{tabular}")
    for line in _CAPTION.split("\n"):
        emit(line)
    emit(r"\end{table}")
    emit()


def separators(rows, by_network):
    marks, prev_net, prev_fam = [], None, None
    for net, desc, exp, cfg, sms in rows:
        fam = family(exp, cfg, desc)
        mark = False
        if prev_net is not None:
            mark = (net != prev_net) if by_network else (fam != prev_fam)
        marks.append(mark)
        prev_net, prev_fam = net, fam
    return marks


def rates(rows, by_network=False):
    for mark, (net, desc, exp, cfg, sms) in zip(separators(rows, by_network), rows):
        if mark:
            emit(r"\midrule")
        emit(r"{} & {} & {} & {} & {} \\".format(
            net, desc, "---" if sms is None else str(sms),
            ms(val(exp, cfg, net, "median")), "{:.0f}".format(thr(exp, cfg, net) or 0)))


def dispersion(rows, by_network=False):
    for mark, (net, desc, exp, cfg, sms) in zip(separators(rows, by_network), rows):
        if mark:
            emit(r"\midrule")
        t, s = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(
            net, desc, ms(std), xr(std, s), ms(val(exp, cfg, net, "p95")),
            ms(val(exp, cfg, net, "p99")), ms(tail), xr(tail, t),
            ms(A.obs_max(DATA, exp, cfg, net))))


# ------------------------------------------------------------------ isolation
# One network alone, the SMs it may use restricted with a green context. gc100 is a
# green context over the whole device and separates the cost of the mechanism from the
# cost of the budget. The other three networks are not in this section: they belong to
# the auxiliary-stream work.
ISO = []
for net in ("mobilenet", "resnet"):
    ISO.append((net, "whole device", "exp1", net + "_whole", 16))
    for pct, sm in ((100, 16), (75, 12), (50, 8), (25, 4)):
        ISO.append((net, "GC %d\\,\\%%" % pct, "exp1", "%s_gc%d" % (net, pct), sm))

RATES_SPEC = "llrrr"
RATES_COLS = [r"Network & Configuration & SMs & Median (ms) & Thr.\ (inf/s) \\"]
DISP_SPEC = "llrrrrrrr"
DISP_COLS = [r"Network & Configuration & std & std $\times$iso & p95 & p99 & tail & "
             r"tail $\times$iso & obs-max \\"]

open_table(RATES_COLS, RATES_SPEC,
           "The two networks in isolation: latency and throughput against the size of "
           "the green context. \\texttt{GC} is a green context, and the percentage is "
           "the share of the device's sixteen SMs it reserves; \\texttt{whole device} "
           "is no reservation at all. The pair \\texttt{whole device} and \\texttt{GC "
           "100\\,\\%} is the same reservation measured twice, so it also shows the "
           "run-to-run spread of a single configuration.",
           "tab:iso-rates")
rates(ISO, by_network=True)
close_table()

open_table(DISP_COLS, DISP_SPEC,
           "The same builds by dispersion. \\emph{tail} is $p99-\\mathrm{median}$; "
           "$\\times$iso is against the network's own whole-device run.",
           "tab:iso-dispersion")
dispersion(ISO, by_network=True)
close_table()

# ------------------------------------------- two networks on green contexts
PAIR_EXP = "exp2-1t"
PAIRS = [("baseline", "baseline", 16, 16),
         ("50_50", "GC 50/50", 8, 8),
         ("75_25", "GC 75/25", 12, 4),
         ("25_75", "GC 25/75", 4, 12)]
PAIR = []
for cfg, desc, mob_sm, res_sm in PAIRS:
    for net, sm in (("mobilenet", mob_sm), ("resnet", res_sm)):
        PAIR.append((net, desc, PAIR_EXP, cfg, sm))

open_table(RATES_COLS, RATES_SPEC,
           "Two networks on disjoint green contexts, one submitting thread. A ratio "
           "is MobileNet's share of the device then ResNet's: \texttt{GC 75/25} is a "
           r"green context of 12\,SMs for MobileNet and 4 for ResNet.",
           "tab:two-rates")
rates(PAIR)
close_table()

open_table(DISP_COLS, DISP_SPEC,
           "The pair by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:two-dispersion")
dispersion(PAIR)
close_table()

# ------------------------------------------------------------------------ MPS
MPS = [("mps_single", "MPS alone", None, None, True),
       ("mps_off", "no MPS", 16, 16, False),
       ("mps_default", "MPS default", 16, 16, False),
       ("mps_atp_50_50", "MPS ATP 50/50", None, None, False),
       ("mps_atp_75_25", "MPS ATP 75/25", None, None, False),
       ("mps_atp_25_75", "MPS ATP 25/75", None, None, False),
       ("mps_static_50_50", "MPS static 50/50", 8, 8, False),
       ("mps_static_75_25", "MPS static 75/25", 12, 4, False),
       ("mps_static_25_75", "MPS static 25/75", 4, 12, False)]
MPSROWS = []
for cfg, desc, mob_sm, res_sm, solo in MPS:
    nets = (("mobilenet", mob_sm),) if solo else (("mobilenet", mob_sm), ("resnet", res_sm))
    for net, sm in nets:
        MPSROWS.append((net, desc, "exp3", cfg, sm))

open_table(RATES_COLS, RATES_SPEC,
           "Two processes under MPS, one submitter per process. \\texttt{MPS ATP} caps "
           "the fraction of the SM's thread slots a client may occupy, which reserves "
           "no particular SMs, so no SM count applies to those rows and the ratio is "
           "of occupancy rather than of SMs. \\texttt{MPS static} assigns exclusive "
           "partitions and is the spatial analogue of a green context. \\texttt{MPS "
           "alone} is one client with nothing to share the device with, and "
           "\\texttt{no MPS} is the same two processes with MPS disabled.",
           "tab:mps-rates")
rates(MPSROWS)
close_table()

open_table(DISP_COLS, DISP_SPEC,
           "The MPS arms by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:mps-dispersion")
dispersion(MPSROWS)
close_table()

# --------------------------------------------------------- green context vs MPS
# Four-way, because the question is what each mechanism does to one network at one
# split, and here that is a row: the mechanisms run down the block for a fixed ratio,
# and the two networks sit side by side.
SHARE = {("50_50", "mobilenet"): 8, ("50_50", "resnet"): 8,
         ("75_25", "mobilenet"): 12, ("75_25", "resnet"): 4,
         ("25_75", "mobilenet"): 4, ("25_75", "resnet"): 12}
FLOOR = {("50_50", "mobilenet"): "mobilenet_gc50", ("50_50", "resnet"): "resnet_gc50",
         ("75_25", "mobilenet"): "mobilenet_gc75", ("75_25", "resnet"): "resnet_gc25",
         ("25_75", "mobilenet"): "mobilenet_gc25", ("25_75", "resnet"): "resnet_gc75"}
RATIOS = [("50_50", "50\\,\\% / 50\\,\\%"), ("75_25", "75\\,\\% / 25\\,\\%"),
          ("25_75", "25\\,\\% / 75\\,\\%")]

FOUR_COLS = [
    r"Ratio & Mechanism & \multicolumn{2}{c}{MobileNet V2-12} & "
    r"\multicolumn{2}{c}{ResNet-50 v2-7} & Aggregate \\",
    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
    r" & & Median (ms) & Thr.\ (inf/s) & Median (ms) & Thr.\ (inf/s) & (inf/s) \\",
]
FOUR_SPEC = "llrrrrr"

open_table(FOUR_COLS, FOUR_SPEC,
           "Green context against MPS static, MPS ATP and MPS default, at matched "
           "ratios, with one submitting thread. Within a ratio block the rows differ "
           "only in the mechanism, so a column can be read down them directly. "
           "\\texttt{MPS default} and \\texttt{no MPS} have no ratio: the first lets "
           "MPS co-schedule the two clients on the whole device, the second is two "
           "plain processes. \\texttt{alone} is the floor: each network on the same "
           "share of the device with nothing else running, which is what its own "
           "partition costs.",
           "tab:compare-rates")

for ci, (cfg, ratio) in enumerate(RATIOS):
    if ci:
        emit(r"\midrule")
    for mech, exp, mcfg in (("GC", PAIR_EXP, cfg),
                            ("MPS static", "exp3", "mps_static_" + cfg),
                            ("MPS ATP", "exp3", "mps_atp_" + cfg)):
        nets = ("mobilenet", "resnet")
        emit(r"{} & {} & {} & {:.0f} & {} & {:.0f} & {} \\".format(
            ratio, mech,
            ms(val(exp, mcfg, "mobilenet", "median")), thr(exp, mcfg, "mobilenet") or 0,
            ms(val(exp, mcfg, "resnet", "median")), thr(exp, mcfg, "resnet") or 0,
            "{:.0f}".format(agg_thr(exp, mcfg, nets) or 0)))

emit(r"\midrule")
for mech, exp, mcfg in (("MPS default", "exp3", "mps_default"),
                        ("no MPS", "exp3", "mps_off")):
    emit(r"--- & {} & {} & {:.0f} & {} & {:.0f} & {} \\".format(
        mech,
        ms(val(exp, mcfg, "mobilenet", "median")), thr(exp, mcfg, "mobilenet") or 0,
        ms(val(exp, mcfg, "resnet", "median")), thr(exp, mcfg, "resnet") or 0,
        "{:.0f}".format(agg_thr(exp, mcfg, ("mobilenet", "resnet")) or 0)))
emit(r"\midrule")
for ci, (cfg, ratio) in enumerate(RATIOS):
    emit(r"{} & alone & {} & {:.0f} & {} & {:.0f} & --- \\".format(
        ratio,
        ms(val("exp1", FLOOR[(cfg, "mobilenet")], "mobilenet", "median")),
        thr("exp1", FLOOR[(cfg, "mobilenet")], "mobilenet") or 0,
        ms(val("exp1", FLOOR[(cfg, "resnet")], "resnet", "median")),
        thr("exp1", FLOOR[(cfg, "resnet")], "resnet") or 0))
close_table()

FOUR_DISP_COLS = [
    r"Ratio & Mechanism & Network & std & std $\times$iso & p95 & p99 & tail & "
    r"tail $\times$iso & obs-max \\"]
open_table(FOUR_DISP_COLS, "lllrrrrrrr",
           "The comparison by dispersion, as in Table~\\ref{tab:iso-dispersion}. The "
           "two rows of a mechanism are its two networks, so the mechanisms can also "
           "be compared within one network by reading every second row.",
           "tab:compare-dispersion")


def four_disp(ratio, mech, pairs):
    for net, exp, cfg in pairs:
        t, s = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(
            ratio, mech, net, ms(std), xr(std, s), ms(val(exp, cfg, net, "p95")),
            ms(val(exp, cfg, net, "p99")), ms(tail), xr(tail, t),
            ms(A.obs_max(DATA, exp, cfg, net))))


for ci, (cfg, ratio) in enumerate(RATIOS):
    if ci:
        emit(r"\midrule")
    four_disp(ratio, "GC", [(n, PAIR_EXP, cfg) for n in ("mobilenet", "resnet")])
    four_disp(ratio, "MPS static", [(n, "exp3", "mps_static_" + cfg) for n in ("mobilenet", "resnet")])
    four_disp(ratio, "MPS ATP", [(n, "exp3", "mps_atp_" + cfg) for n in ("mobilenet", "resnet")])
emit(r"\midrule")
for mech, mcfg in (("MPS default", "mps_default"), ("no MPS", "mps_off")):
    four_disp("---", mech, [(n, "exp3", mcfg) for n in ("mobilenet", "resnet")])
emit(r"\midrule")
for cfg, ratio in RATIOS:
    four_disp(ratio, "alone", [(n, "exp1", FLOOR[(cfg, n)]) for n in ("mobilenet", "resnet")])
close_table()

# --------------------------------------------------------------- the CPU side
CPU = [("t1_unpinned", "GC", "1 thread"),
       ("t1_pinfifo99", "GC", "1 thread, FIFO 99"),
       ("t2_unpinned", "GC", "2 threads"),
       ("t2_core67", "GC", "2 threads, cores 6-7"),
       ("mps_owncores", "MPS static", "1 core each")]
CPUROWS = []
for cfg, mech, desc in CPU:
    for net in ("mobilenet", "resnet"):
        CPUROWS.append((net, mech, desc, "exp7", cfg, 12 if net == "mobilenet" else 4))


def cpu_rates(rows):
    """Six columns: the mechanism is its own column, because this table holds both."""
    prev = None
    for net, mech, desc, exp, cfg, sms in rows:
        if prev is not None and mech != prev:
            emit(r"\midrule")
        prev = mech
        emit(r"{} & {} & {} & {} & {} & {} \\".format(
            net, mech, desc, sms, ms(val(exp, cfg, net, "median")),
            "{:.0f}".format(thr(exp, cfg, net) or 0)))


def cpu_dispersion(rows):
    """As the other dispersion tables, except p95, which is dropped so that the extra
    mechanism column still fits the text width."""
    prev = None
    for net, mech, desc, exp, cfg, sms in rows:
        if prev is not None and mech != prev:
            emit(r"\midrule")
        prev = mech
        t, sd = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(
            net, mech, desc, ms(std), xr(std, sd), ms(val(exp, cfg, net, "p99")),
            ms(tail), xr(tail, t), ms(A.obs_max(DATA, exp, cfg, net))))

open_table([r"Network & Mechanism & Configuration & SMs & Median (ms) & "
                r"Thr.\ (inf/s) \\"], "lllrrr",
           "Two networks at 75\\,\\% / 25\\,\\% with the submission model as the only "
           "variable. \\texttt{1 thread} drives both networks from one thread while "
           "keeping one inference in flight per network; \\texttt{2 threads} gives "
           "each network its own thread, which is what an application does by "
           "accident, and \\texttt{FIFO 99} additionally pins the submitter to one "
           "core at the highest real-time priority the kernel offers. The last row is "
           "the same split under MPS static, one process per network and one core "
           "each, for comparison with the green-context rows above it.",
           "tab:cpu-rates")
cpu_rates(CPUROWS)
close_table()

open_table([r"Network & Mechanism & Configuration & std & std $\times$iso & p99 & "
                r"tail & tail $\times$iso & obs-max \\"], "lllrrrrrr",
           "The same arms by dispersion, as in Table~\\ref{tab:iso-dispersion}, without "
           "p95 so that the mechanism column fits the text width.",
           "tab:cpu-dispersion")
cpu_dispersion(CPUROWS)
close_table()

sys.stdout.write("\n".join(OUT))
