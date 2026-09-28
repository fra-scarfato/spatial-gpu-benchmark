#!/usr/bin/env python3
"""results-tables.py — the tables of the Results section, from the measurements.

PERFORMANCE tables put the two networks on top, side by side, with the aggregate at
the right; each network's columns are its own SM count, median and throughput. That is
the layout of the comparison tables already in the thesis, and it is the one that shows
a difference between two networks at the same configuration as a row of numbers rather
than as two rows to be compared in the head.

    Configuration | SMs | MobileNet (median, thr) | SMs | ResNet (median, thr) | Aggregate

DISPERSION tables are read one network at a time instead, because there each row is a
statistic of a single distribution:

    Network | Configuration | std | xiso | p95 | p99 | tail | xiso | obs-max

The comparison tables keep the four-way layout (ratio x mechanism), for the same reason
the performance tables are wide: what a mechanism does to one network at one split is
a row.

Every configuration cell names its mechanism — GC, MPS static, MPS ATP, MPS default,
no MPS, whole or alone — so no row needs the caption to be identified.

Submission model, frozen earlier: one submitting thread. Isolation and MPS have one
process per network, so one submitter by construction; green context is exp2-1t.

Run from anywhere: it imports aggregate.py from its own directory and reads the results
tree under $DATA. No number is typed by hand.
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
_CAPTION = ""
NETS = ("mobilenet", "resnet")


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
    """Measured inferences per second: count over the window the runner timed. Not
    1000/median, which is the GPU-only figure and reads higher because it ignores the
    gap the host leaves between inferences."""
    vals = [m["count"] / m["duration_s"] for m in runs_of(exp, cfg, net)
            if m.get("duration_s", 0) > 0]
    return st.median(vals) if vals else None


def agg_thr(exp, cfg, nets):
    """Both networks' inferences over the window they shared; max(duration) is that
    window, because the processes start together. Dividing by the sum, or by the
    shorter duration, would invent throughput that never happened."""
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


def must(v, what):
    """Refuse to emit a missing measurement.

    A table cell that is None because a file was not parsed looks exactly like a cell
    that is legitimately "not applicable", and one of these reached a table before it
    was caught: the aggregate was skipping every row labelled with a third network, and
    the table printed dashes. Where a value is required, say so.
    """
    if v is None:
        raise SystemExit("results-tables.py: no measurement for " + what)
    return v


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
    """The caption goes BELOW the tabular: above a table LaTeX separates it with
    \\belowcaptionskip, which is 0pt by default, so it sits against the top rule."""
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


# ---------------------------------------------------------------- performance
def wide_head(lead, one_sm, agg):
    """Two header rows: the networks as column groups, then the statistic names.

    one_sm: the two networks share one SM column (isolation, where both receive the
    same share). Otherwise each network carries its own, which the pair needs because
    75/25 gives MobileNet twelve SMs and ResNet four.

    The row terminator ends the LAST cell rather than being a cell of its own: with no
    aggregate column, a separate one produced seven cells in a six-column table.
    """
    nlead = len(lead)
    if one_sm:
        g1, g2 = "%d-%d" % (nlead + 2, nlead + 3), "%d-%d" % (nlead + 4, nlead + 5)
        row1 = lead + [r"SMs", r"\multicolumn{2}{c}{MobileNet V2-12}",
                       r"\multicolumn{2}{c}{ResNet-50 v2-7}"]
        row2 = [""] * nlead + ["", r"Median (ms)", r"Thr.\ (inf/s)", r"Median (ms)",
                               r"Thr.\ (inf/s)"]
    else:
        g1, g2 = "%d-%d" % (nlead + 1, nlead + 3), "%d-%d" % (nlead + 4, nlead + 6)
        row1 = lead + [r"\multicolumn{3}{c}{MobileNet V2-12}",
                       r"\multicolumn{3}{c}{ResNet-50 v2-7}"]
        row2 = [""] * nlead + [r"SMs", r"Median (ms)", r"Thr.\ (inf/s)", r"SMs",
                               r"Median (ms)", r"Thr.\ (inf/s)"]
    if agg:
        row1.append(r"Aggregate \\")
        row2.append(r"(inf/s) \\")
    else:
        row1[-1] += r" \\"
        row2[-1] += r" \\"
    cols = [" & ".join(row1), r"\cmidrule(lr){%s} \cmidrule(lr){%s}" % (g1, g2),
            " & ".join(row2)]
    spec = "l" * nlead + ("rrrrr" if one_sm else "rrrrrr") + ("r" if agg else "")
    return cols, spec


def wide_row(lead, rows):
    """rows: (which network, sms, exp, cfg) in order, MobileNet then ResNet."""
    cells = list(lead)
    for _, sms, exp, cfg in rows:
        cells += ["---" if sms is None else str(sms),
                  ms(val(exp, cfg, _, "median")), "{:.0f}".format(thr(exp, cfg, _) or 0)]
    return cells


def wide_table(cols, spec, caption, label, body, agg=False, agg_exp=None, agg_cfg=None):
    open_table(cols, spec, caption, label)
    prev = None
    for lead, rows, fam in body:
        if prev is not None and fam != prev:
            emit(r"\midrule")
        prev = fam
        cells = wide_row(lead, rows)
        if agg:
            cells.append("{:.0f}".format(agg_thr(agg_exp, agg_cfg or "", NETS) or 0)
                         if agg_cfg else "---")
        emit(" & ".join(cells) + r" \\")
    close_table()


# ------------------------------------------------------------------ isolation
ISO = []
for net in NETS:
    pass
ISO_ROWS = []
for pct, sm in ((100, 16), (75, 12), (50, 8), (25, 4)):
    ISO_ROWS.append(("GC %d\,\%%" % pct, sm, "exp1", "%s_gc%d" % ("mobilenet", pct),
                     "exp1", "%s_gc%d" % ("resnet", pct)))
cols, spec = wide_head([r"Configuration"], True, False)
open_table(cols, spec,
           "The two networks in isolation: latency and throughput against the size of the "
           "green context, which both networks receive in full. \\texttt{GC} is a green "
           "context and the percentage is the share of the device's sixteen SMs it "
           "reserves. \\texttt{whole device} is no reservation at all, and the pair "
           "\\texttt{whole device} and \\texttt{GC 100\,\%} is the same reservation "
           "measured twice, so it also shows the run-to-run spread of one configuration.",
           "tab:iso-rates")
emit(r"whole device & 16 & {} & {:.0f} & {} & {:.0f} \\".format(
    ms(val("exp1", "mobilenet_whole", "mobilenet", "median")),
    thr("exp1", "mobilenet_whole", "mobilenet") or 0,
    ms(val("exp1", "resnet_whole", "resnet", "median")),
    thr("exp1", "resnet_whole", "resnet") or 0))
for desc, sm, mo_e, mo_c, re_e, re_c in ISO_ROWS:
    emit(r"{} & {} & {} & {:.0f} & {} & {:.0f} \\".format(
        desc, sm, ms(val(mo_e, mo_c, "mobilenet", "median")),
        thr(mo_e, mo_c, "mobilenet") or 0,
        ms(val(re_e, re_c, "resnet", "median")), thr(re_e, re_c, "resnet") or 0))
close_table()

DISP_COLS = [r"Network & Configuration & std & std $\times$iso & p95 & p99 & tail & "
             r"tail $\times$iso & obs-max \\"]
DISP_SPEC = "llrrrrrrr"


def dispersion(rows):
    prev = None
    for net, desc, exp, cfg, sms in rows:
        fam = family(exp, cfg, desc)
        if prev is not None and fam != prev:
            emit(r"\midrule")
        prev = fam
        t, s = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(
            net, desc, ms(std), xr(std, s), ms(val(exp, cfg, net, "p95")),
            ms(val(exp, cfg, net, "p99")), ms(tail), xr(tail, t),
            ms(A.obs_max(DATA, exp, cfg, net))))


disp_rows = []
for net in NETS:
    disp_rows.append((net, "whole device", "exp1", net + "_whole", 16))
    for pct, sm in ((100, 16), (75, 12), (50, 8), (25, 4)):
        disp_rows.append((net, "GC %d\,\%%" % pct, "exp1", "%s_gc%d" % (net, pct), sm))
open_table(DISP_COLS, DISP_SPEC,
           "The same builds by dispersion. \\emph{tail} is $p99-\\mathrm{median}$; "
           "$\\times$iso is against the network's own whole-device run.",
           "tab:iso-dispersion")
dispersion(disp_rows)
close_table()

# ------------------------------------------- two networks on green contexts
PAIR_EXP = "exp2-1t"
PAIRS = [("baseline", "baseline", 16, 16),
         ("50_50", "GC 50/50", 8, 8),
         ("75_25", "GC 75/25", 12, 4),
         ("25_75", "GC 25/75", 4, 12)]
cols, spec = wide_head([r"Configuration"], False, True)
open_table(cols, spec,
           "Two networks on disjoint green contexts, one submitting thread. A ratio is "
           "MobileNet's share of the device then ResNet's: \\texttt{GC 75/25} is a green "
           "context of 12\\,SMs for MobileNet and 4 for ResNet. Aggregate throughput is "
           "both networks' inferences over the window they shared, so it is close to the "
           "sum of the two throughput columns and a check that nothing is counted twice.",
           "tab:two-rates")
for cfg, desc, mob_sm, res_sm in PAIRS:
    emit(r"{} & {} & {} & {:.0f} & {} & {} & {:.0f} & {} \\".format(
        desc, mob_sm, ms(val(PAIR_EXP, cfg, "mobilenet", "median")),
        thr(PAIR_EXP, cfg, "mobilenet") or 0, res_sm,
        ms(val(PAIR_EXP, cfg, "resnet", "median")), thr(PAIR_EXP, cfg, "resnet") or 0,
        "{:.0f}".format(agg_thr(PAIR_EXP, cfg, NETS) or 0)))
close_table()

pairdisp = []
for cfg, desc, mob_sm, res_sm in PAIRS:
    pairdisp += [(n, desc, PAIR_EXP, cfg, mob_sm if n == "mobilenet" else res_sm)
                 for n in NETS]
open_table(DISP_COLS, DISP_SPEC,
           "The pair by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:two-dispersion")
dispersion(pairdisp)
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
cols, spec = wide_head([r"Configuration"], False, True)
open_table(cols, spec,
           "Two processes under MPS, one submitter per process. \\texttt{MPS ATP} caps the "
           "fraction of the SM's thread slots a client may occupy, which reserves no "
           "particular SMs, so the ratio is of occupancy rather than of SMs and no SM "
           "count applies. \\texttt{MPS static} assigns exclusive partitions and is the "
           "spatial analogue of a green context. \\texttt{MPS alone} is one client with "
           "nothing to share the device with.",
           "tab:mps-rates")
prev = None
for cfg, desc, mob_sm, res_sm, solo in MPS:
    fam = family("exp3", cfg, desc)
    if prev is not None and fam != prev:
        emit(r"\midrule")
    prev = fam
    if solo:
        emit(r"{} & --- & {} & {:.0f} & --- & --- & --- & --- \\".format(
            desc, ms(val("exp3", cfg, "mobilenet", "median")),
            thr("exp3", cfg, "mobilenet") or 0))
    else:
        emit(r"{} & {} & {} & {:.0f} & {} & {} & {:.0f} & {} \\".format(
            desc, "---" if mob_sm is None else mob_sm,
            ms(val("exp3", cfg, "mobilenet", "median")), thr("exp3", cfg, "mobilenet") or 0,
            "---" if res_sm is None else res_sm,
            ms(val("exp3", cfg, "resnet", "median")), thr("exp3", cfg, "resnet") or 0,
            "{:.0f}".format(agg_thr("exp3", cfg, NETS) or 0)))
close_table()

mpsdisp = []
for cfg, desc, mob_sm, res_sm, solo in MPS:
    for n in (("mobilenet",) if solo else NETS):
        mpsdisp.append((n, desc, "exp3", cfg, mob_sm if n == "mobilenet" else res_sm))
open_table(DISP_COLS, DISP_SPEC,
           "The MPS arms by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:mps-dispersion")
dispersion(mpsdisp)
close_table()

# --------------------------------------------------------- green context vs MPS
SHARE = {("50_50", "mobilenet"): 8, ("50_50", "resnet"): 8,
         ("75_25", "mobilenet"): 12, ("75_25", "resnet"): 4,
         ("25_75", "mobilenet"): 4, ("25_75", "resnet"): 12}
FLOOR = {("50_50", "mobilenet"): "mobilenet_gc50", ("50_50", "resnet"): "resnet_gc50",
         ("75_25", "mobilenet"): "mobilenet_gc75", ("75_25", "resnet"): "resnet_gc25",
         ("25_75", "mobilenet"): "mobilenet_gc25", ("25_75", "resnet"): "resnet_gc75"}
RATIOS = [("50_50", "50\,\% / 50\,\%"), ("75_25", "75\,\% / 25\,\%"),
          ("25_75", "25\,\% / 75\,\%")]
FOUR_COLS = [
    r"Ratio & Mechanism & \multicolumn{2}{c}{MobileNet V2-12} & "
    r"\multicolumn{2}{c}{ResNet-50 v2-7} & Aggregate \\",
    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
    r" & & Median (ms) & Thr.\ (inf/s) & Median (ms) & Thr.\ (inf/s) & (inf/s) \\",
]
open_table(FOUR_COLS, "llrrrrr",
           "Green context against MPS static, MPS ATP and MPS default, at matched ratios, "
           "with one submitting thread. Within a ratio block the rows differ only in the "
           "mechanism, so a column can be read straight down them. \\texttt{MPS default} "
           "and \\texttt{no MPS} have no ratio: the first lets MPS co-schedule the two "
           "clients on the whole device, the second is two plain processes. \\texttt{alone} "
           "is the floor, each network on the same share of the device with nothing else "
           "running.",
           "tab:compare-rates")
for ci, (cfg, ratio) in enumerate(RATIOS):
    if ci:
        emit(r"\midrule")
    for mech, exp, mcfg in (("GC", PAIR_EXP, cfg), ("MPS static", "exp3", "mps_static_" + cfg),
                            ("MPS ATP", "exp3", "mps_atp_" + cfg)):
        emit(r"{} & {} & {} & {:.0f} & {} & {:.0f} & {} \\".format(
            ratio, mech, ms(val(exp, mcfg, "mobilenet", "median")),
            thr(exp, mcfg, "mobilenet") or 0, ms(val(exp, mcfg, "resnet", "median")),
            thr(exp, mcfg, "resnet") or 0, "{:.0f}".format(agg_thr(exp, mcfg, NETS) or 0)))
emit(r"\midrule")
for mech, mcfg in (("MPS default", "mps_default"), ("no MPS", "mps_off")):
    emit(r"--- & {} & {} & {:.0f} & {} & {:.0f} & {} \\".format(
        mech, ms(val("exp3", mcfg, "mobilenet", "median")), thr("exp3", mcfg, "mobilenet") or 0,
        ms(val("exp3", mcfg, "resnet", "median")), thr("exp3", mcfg, "resnet") or 0,
        "{:.0f}".format(agg_thr("exp3", mcfg, NETS) or 0)))
emit(r"\midrule")
for cfg, ratio in RATIOS:
    emit(r"{} & alone & {} & {:.0f} & {} & {:.0f} & --- \\".format(
        ratio, ms(val("exp1", FLOOR[(cfg, "mobilenet")], "mobilenet", "median")),
        thr("exp1", FLOOR[(cfg, "mobilenet")], "mobilenet") or 0,
        ms(val("exp1", FLOOR[(cfg, "resnet")], "resnet", "median")),
        thr("exp1", FLOOR[(cfg, "resnet")], "resnet") or 0))
close_table()

open_table([r"Ratio & Mechanism & Network & std & std $\times$iso & p95 & p99 & tail & "
            r"tail $\times$iso & obs-max \\"], "lllrrrrrrr",
           "The comparison by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
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
    four_disp(ratio, "GC", [(n, PAIR_EXP, cfg) for n in NETS])
    four_disp(ratio, "MPS static", [(n, "exp3", "mps_static_" + cfg) for n in NETS])
    four_disp(ratio, "MPS ATP", [(n, "exp3", "mps_atp_" + cfg) for n in NETS])
emit(r"\midrule")
for mech, mcfg in (("MPS default", "mps_default"), ("no MPS", "mps_off")):
    four_disp("---", mech, [(n, "exp3", mcfg) for n in NETS])
emit(r"\midrule")
for cfg, ratio in RATIOS:
    four_disp(ratio, "alone", [(n, "exp1", FLOOR[(cfg, n)]) for n in NETS])
close_table()

# --------------------------------------------------------------- the CPU side
CPU = [("t1_unpinned", "GC", "1 thread"),
       ("t1_pinfifo99", "GC", "1 thread, FIFO 99"),
       ("t2_unpinned", "GC", "2 threads"),
       ("t2_core67", "GC", "2 threads, cores 6-7"),
       ("mps_owncores", "MPS static", "1 core each")]
cols, spec = wide_head([r"Mechanism", r"Configuration"], False, True)
open_table(cols, spec,
           "Two networks at 75\,\% / 25\,\% with the submission model as the only "
           "variable. \\texttt{1 thread} drives both networks from one thread while "
           "keeping one inference in flight per network; \\texttt{2 threads} gives each "
           "network its own thread, which is what an application does by accident, and "
           "\\texttt{FIFO 99} additionally pins the submitter to one core at the highest "
           "real-time priority the kernel offers. The last row is the same split under MPS "
           "static, one process per network and one core each.",
           "tab:cpu-rates")
prev = None
for cfg, mech, desc in CPU:
    fam = family("exp7", cfg, desc)
    if prev is not None and fam != prev:
        emit(r"\midrule")
    prev = fam
    emit(r"{} & {} & 12 & {} & {:.0f} & 4 & {} & {:.0f} & {} \\".format(
        mech, desc, ms(val("exp7", cfg, "mobilenet", "median")),
        thr("exp7", cfg, "mobilenet") or 0, ms(val("exp7", cfg, "resnet", "median")),
        thr("exp7", cfg, "resnet") or 0, "{:.0f}".format(agg_thr("exp7", cfg, NETS) or 0)))
close_table()

cpudisp = []
for cfg, mech, desc in CPU:
    for n in NETS:
        cpudisp.append((n, "{} {}".format(mech, desc), "exp7", cfg,
                        12 if n == "mobilenet" else 4))
open_table(DISP_COLS, DISP_SPEC,
           "The same arms by dispersion, as in Table~\\ref{tab:iso-dispersion}.",
           "tab:cpu-dispersion")
dispersion(cpudisp)
close_table()


# ------------------------------------------------- the work-queue configuration
# The subject is RetinaNet's aux8 build, the only engine in the corpus whose stream
# count (9) exceeds the platform's default declaration (8), with MobileNet as the
# co-runner at 50/50: one stream, so it never competes for the declaration.
WQ = [("pair_smsonly", "SMs only (descriptor carries no work queues, declaration 8)"),
      ("pair_dev10", "declared 10, \\texttt{device\\_ctx}"),
      ("pair_bal10", "declared 10, \\texttt{green\\_ctx\\_balanced}")]
open_table([r"Network & Configuration & Median (ms) & Thr.\ (inf/s) & p99 (ms) & "
            r"tail (ms) \\"], "llrrrr",
           "The work-queue half of a green context descriptor varied on the pair whose "
           "client has more streams than the platform declares by default: RetinaNet's "
           "\\texttt{aux8} build with nine stream-ordered workloads against a default "
           "declaration of eight, and MobileNet as the co-runner, 50\,\% each, one "
           "submitting thread. The first row is the descriptor every earlier measurement "
           "used.",
           "tab:wq-rows")
for cfg, desc in WQ:
    for net in ("retinanet", "mobilenet"):
        what = "exp8 {} {}".format(cfg, net)
        emit(r"{} & {} & {} & {:.0f} & {} & {} \\".format(
            net, desc, ms(must(val("exp8", cfg, net, "median"), what)),
            must(thr("exp8", cfg, net), what) or 0,
            ms(must(val("exp8", cfg, net, "p99"), what)),
            ms(must(val("exp8", cfg, net, "p99_minus_median"), what))))
close_table()

AUX = [("inception", "inception_whole", "default build, no auxiliary stream"),
       ("inception", "inception_aux1_whole", "1 auxiliary stream (\\texttt{aux1})"),
       ("inception", "inception_aux2_whole", "2 auxiliary streams (\\texttt{aux2})"),
       ("inception", "inception_aux4_whole", "4 auxiliary streams (\\texttt{aux4})"),
       ("inception", "inception_aux5_whole", "5 auxiliary streams (\\texttt{aux5})"),
       ("retinanet", "retinanet_whole", "default build, no auxiliary stream"),
       ("retinanet", "retinanet_aux1_whole", "1 auxiliary stream (\\texttt{aux1})"),
       ("retinanet", "retinanet_aux8_whole", "8 auxiliary streams (\\texttt{aux8})"),
       ("retinanet", "retinanet_aux16_whole", "9 auxiliary streams (\\texttt{aux16})")]
open_table([r"Network & Build & Median (ms) & Thr.\ (inf/s) & tail (ms) \\"], "llrrr",
           "What the auxiliary streams themselves buy, per engine build, each measured "
           "alone on the whole device. The builds differ only in how many auxiliary "
           "streams the builder was allowed to use, and TensorRT then creates and uses "
           "them itself.",
           "tab:aux-builds")
for net, cfg, desc in AUX:
    emit(r"{} & {} & {} & {:.0f} & {} \\".format(
        net, desc, ms(val("exp1", cfg, net, "median")), thr("exp1", cfg, net) or 0,
        ms(val("exp1", cfg, net, "p99_minus_median"))))
close_table()


# Three clients in one split: 1 + 3 + 9 = 13 stream-ordered workloads against a default
# declaration of 8. The gain attempt, since a gain is only possible where the default
# budget is exceeded.
MULTI = [("multi_default", "default declaration, inherited 8"),
         ("multi_bal13", "declared 13, \\texttt{green\\_ctx\\_balanced}"),
         ("multi_dev13", "declared 13, \\texttt{device\\_ctx}")]
open_table([r"Network & Configuration & Median (ms) & p99 (ms) & tail (ms) \\"], "llrrr",
           "Three clients in one green-context split, one submitting thread: MobileNet "
           "6\\,SMs, Inception\\,aux2 6, RetinaNet\\,aux8 4, so the three together ask for "
           "thirteen stream-ordered workloads against a platform default declaration of "
           "eight. Aggregate throughput is 129 inferences per second in every row.",
           "tab:wq-multi")
for cfg, desc in MULTI:
    for net in ("mobilenet", "inception", "retinanet"):
        what = "exp8 {} {}".format(cfg, net)
        emit(r"{} & {} & {} & {} & {} \\".format(
            net, desc, ms(must(val("exp8", cfg, net, "median"), what)),
            ms(must(val("exp8", cfg, net, "p99"), what)),
            ms(must(val("exp8", cfg, net, "p99_minus_median"), what))))
close_table()










# ------------------------------------------- memory contention: the target's budget
# The target holds 8 SMs in every row, so the rows differ in the co-runner and the mechanism
# alone. The pair of tables follows the shape the two mechanisms are compared with at matched
# ratios: a rates table with the two networks grouped under multicolumn headers, and a
# dispersion table with the network in its own column and the multipliers against the isolated
# whole-device run. Under MPS static the same split needs three partitions - 8 SMs to the
# target, 4 to the co-runner and 4 left unassigned - because a static partition set covers the
# device in two parts.
MB_ROWS = [("none", "GC", "exp1", "mobilenet_gc50", False),
           ("none", "MPS static", "exp9", "mps_alone", False),
           ("resnet 4\,SMs", "GC", "exp9", "target8_co4", True),
           ("resnet 4\,SMs", "MPS static", "exp9", "mps_target8_co4", True),
           ("resnet 8\,SMs", "GC", "exp2-1t", "50_50", True),
           ("resnet 8\,SMs", "MPS static", "exp9", "mps_target8_co8", True)]
MB_COLS = [
    r"Configuration & Mechanism & \multicolumn{2}{c}{MobileNet V2-12} & "
    r"\multicolumn{2}{c}{ResNet-50 v2-7} & Aggregate \\",
    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
    r" & & Median (ms) & Thr.\ (inf/s) & Median (ms) & Thr.\ (inf/s) & (inf/s) \\",
]

open_table(MB_COLS, "llrrrrr",
           "Memory contention with the target's budget fixed, under each mechanism, with one "
           "submitting thread per process. MobileNet holds the same 8\,SMs in every row, so "
           "each configuration appears twice, under green contexts and under MPS static "
           "partitions, and the two rows of a pair differ in the mechanism alone. An MPS "
           "static partition set covers the device in two parts, so the 4\,SM co-runner is "
           "given its own partition and the last quarter of the device is left unassigned, as "
           "the green-context arm leaves it, so the 4\,SM arms use half the device. The aggregate is the two clients' rates together.",
           "tab:mem-budget")
_mb_cfg = None
for _desc, _mech, _exp, _cfg, _co in MB_ROWS:
    if _mb_cfg is not None and _desc != _mb_cfg:
        emit(r"\midrule")
    _mb_cfg = _desc
    _mob_m = ms(val(_exp, _cfg, "mobilenet", "median"))
    _mob_t = thr(_exp, _cfg, "mobilenet") or 0
    _agg = agg_thr(_exp, _cfg, NETS) if _co else agg_thr(_exp, _cfg, ("mobilenet",))
    if _co:
        _res_m = ms(val(_exp, _cfg, "resnet", "median"))
        _res_t = "{:.0f}".format(thr(_exp, _cfg, "resnet") or 0)
    else:
        _res_m = _res_t = "---"
    emit(r"{} & {} & {} & {:.0f} & {} & {} & {} \\".format(
        _desc, _mech, _mob_m, _mob_t, _res_m, _res_t,
        "{:.0f}".format(_agg) if _agg else "---"))
close_table()


def mem_budget_disp(desc, mech, exp, cfg, nets):
    for net in nets:
        t, s = ref(net)
        tail = val(exp, cfg, net, "p99_minus_median")
        std = val(exp, cfg, net, "std")
        emit(r"{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\".format(
            desc, mech, net, ms(std), xr(std, s), ms(val(exp, cfg, net, "p95")),
            ms(val(exp, cfg, net, "p99")), ms(tail), xr(tail, t),
            ms(A.obs_max(DATA, exp, cfg, net))))


open_table([r"Configuration & Mechanism & Network & std & std $\times$iso & p95 & p99 & tail & "
            r"tail $\times$iso & obs-max \\"], "lllrrrrrrr",
           "The same arms by dispersion, as in Table~\\ref{tab:compare-dispersion}. The "
           "multipliers are against the isolated reference of the network, which is the "
           "network alone on the whole device, not alone at the target's SM count.",
           "tab:mem-budget-dispersion")
_mb_cfg = None
for _desc, _mech, _exp, _cfg, _co in MB_ROWS:
    if _mb_cfg is not None and _desc != _mb_cfg:
        emit(r"\midrule")
    _mb_cfg = _desc
    mem_budget_disp(_desc, _mech, _exp, _cfg, NETS if _co else ("mobilenet",))
close_table()

# ------------------------------------------------- memory contention: the matrix
# The target's partition is fixed at 8 SMs; what varies is the participant's kind, its budget,
# and the mechanism that confines it. The two mechanisms cannot be compared at a participant
# budget other than 8 SMs, because an MPS static partition divides the device in two parts, so
# the mechanism rows appear at 8 SMs and the budget rows under green contexts. The participant
# writes no stats file, so its own load is read from the run's log, where it reports the bytes
# or the FLOPs it moved. Each mechanism's multipliers are against its own reference row.
_BENCH = os.environ.get("BENCH", "/mnt/nvme/f.scarfato/bench")


def itf_load(family, cfg, unit):
    """Median of the participant's own reported rate across runs, or None."""
    import glob
    import re
    import statistics
    found = []
    for pat in ("r*.log", "r*_itf.log"):
        for f in sorted(glob.glob("{}/{}/{}/{}".format(_BENCH, family, cfg, pat))):
            text = open(f, errors="replace").read()
            if unit == "GB/s":
                m = re.findall(r"bytes moved:\s*[\d.]+ GB \(([\d.]+) GB/s\)", text)
            else:
                m = re.findall(r"arithmetic:\s*[\d.]+ GFLOP \(([\d.]+) TFLOP/s\)", text)
            if m:
                found.append(float(m[-1]))
    return statistics.median(found) if found else None


_mx_ref = {"GC": ("exp1", "mobilenet_gc50"), "MPS static": ("exp9", "mps_alone")}
_mx_base = {k: must(val(e, c, "mobilenet", "median"), c) for k, (e, c) in _mx_ref.items()}
_mx_base_tail = {k: must(val(e, c, "mobilenet", "p99_minus_median"), c)
                 for k, (e, c) in _mx_ref.items()}
MX_ROWS = [("participant none", "GC", "exp1", "mobilenet_gc50", None, None),
           (r"\texttt{mem\_saxpy} at 4\,SMs", "GC", "exp9", "mem4", "GB/s", "mem-s2"),
           (r"\texttt{mem\_saxpy} at 8\,SMs", "GC", "exp9", "mem8", "GB/s", "mem-s2"),
           (r"\texttt{compute\_gemm} at 4\,SMs", "GC", "exp9", "compute4", "TFLOP/s", "mem-s2"),
           (r"\texttt{compute\_gemm} at 8\,SMs", "GC", "exp9", "compute8", "TFLOP/s", "mem-s2"),
           ("participant none", "MPS static", "exp9", "mps_alone", None, None),
           (r"\texttt{mem\_saxpy} at 8\,SMs", "MPS static", "exp9", "mps_mem8", "GB/s", "mem-s3"),
           (r"\texttt{compute\_gemm} at 8\,SMs", "MPS static", "exp9", "mps_compute8",
            "TFLOP/s", "mem-s3")]

open_table([r"Co-runner & Mechanism & Median (ms) & Median $\times$iso & Thr.\ (inf/s) & "
            r"tail (ms) & tail $\times$iso \\"], "llrrrrr",
           "What a participant's kind and budget cost a target holding its own 8\,SMs, under "
           "each mechanism. The configuration carries the load the participant reported for "
           "itself, the bytes the memory kind moved or the FLOPs the compute kind performed. "
           "The budget is varied under green contexts, since an MPS static partition divides "
           "the device in two parts and fixes the participant at 8\,SMs; each mechanism's "
           "multipliers are against its own target alone.",
           "tab:mem-matrix")
_mx_mech = None
for _label, _mech, _exp, _cfg, _unit, _fam in MX_ROWS:
    if _mx_mech is not None and _mech != _mx_mech:
        emit(r"\midrule")
    _mx_mech = _mech
    _m = must(val(_exp, _cfg, "mobilenet", "median"), _cfg + " median")
    _t = must(val(_exp, _cfg, "mobilenet", "p99_minus_median"), _cfg + " tail")
    _thr = must(thr(_exp, _cfg, "mobilenet"), _cfg + " rate") or 0
    _load = itf_load(_fam, _cfg, _unit) if _fam else None
    _desc = _label + (" ({:.1f} {})".format(_load, _unit) if _load else "")
    emit(r"{} & {} & {} & {:.3f} & {:.0f} & {} & {:.3f} \\".format(
        _desc, _mech, ms(_m), _m / _mx_base[_mech], _thr, ms(_t), _t / _mx_base_tail[_mech]))
close_table()

# --------------------------------------------------- memory contention: the trace
# Three nsys kernel traces, two seconds each, of the same target alone and with each
# participant at 4 SMs. One inference is one pass of the kernel cycle, whose length the trace
# states itself; the span of an inference is its kernels' time plus the gaps between them,
# and the last column is the part of the gap that elapsed before the launch call for the next
# kernel returned. Diagnostic, not a campaign measurement: profiling perturbs.
_TRACE_CSV = os.environ.get("TRACE_CSV", "/mnt/nvme/f.scarfato/bench/trace/mem-trace.csv")
if os.path.exists(_TRACE_CSV):
    import csv as _csv
    _conf = {"alone": "participant none", "mem4": r"\texttt{mem\_saxpy} at 4\,SMs",
             "compute4": r"\texttt{compute\_gemm} at 4\,SMs"}
    open_table([r"Network & Configuration & Inference & Span (ms) & Kernels "
                r"(ms) & Gaps (ms) & Host-late gaps (ms) \\"], "lllrrrr",
               "One inference, decomposed, from a trace of the target alone and of the "
               "target with each participant. The trace is sliced into inferences by the "
               "period of its own kernel sequence, fifty-five kernels in all three, so the "
               "inference here is the same unit the campaign measures. The last column is "
               "the part of the gap between kernels that elapsed before the launch call for "
               "the next kernel had returned, which the host owes and the device does not.",
               "tab:mem-trace")
    _last = None
    for _row in _csv.DictReader(open(_TRACE_CSV)):
        if _last is not None and _row["trace"] != _last:
            emit(r"\midrule")
        _last = _row["trace"]
        emit(r"mobilenet & {} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\".format(
            _conf.get(_row["trace"], _row["trace"]),
            "slowest 1\,\%" if _row["quantile"] != "median inference" else "median",
            float(_row["span_ms"]), float(_row["kernels_ms"]), float(_row["gaps_ms"]),
            float(_row["host_gaps_ms"])))
    close_table()

sys.stdout.write("\n".join(OUT))
