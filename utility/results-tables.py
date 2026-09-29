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

from lib.tables import (NETS, OUT, agg_thr, close_table, emit, family, ms, must,  # noqa: E402
                       open_table, ref, runs_of, thr, val, xr)


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
WQ = [("pair-sms-only", "SMs only (descriptor carries no work queues, declaration 8)"),
      ("pair-device-ctx-10", "declared 10, \\texttt{device\\_ctx}"),
      ("pair-balanced-10", "declared 10, \\texttt{green\\_ctx\\_balanced}")]
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
MULTI = [("three-clients-default", "default declaration, inherited 8"),
         ("three-clients-balanced-13", "declared 13, \\texttt{green\\_ctx\\_balanced}"),
         ("three-clients-device-ctx-13", "declared 13, \\texttt{device\\_ctx}")]
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
           ("none", "MPS static", "exp9", "mps-alone", False),
           ("resnet 4\,SMs", "GC", "exp9", "mobilenet8-resnet4", True),
           ("resnet 4\,SMs", "MPS static", "exp9", "mps-mobilenet8-resnet4", True),
           ("resnet 8\,SMs", "GC", "exp2-1t", "50_50", True),
           ("resnet 8\,SMs", "MPS static", "exp9", "mps-mobilenet8-resnet8", True)]
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
# and the mechanism that confines it. Every configuration is measured under BOTH mechanisms and
# the two rows of a pair stand together, so the comparison is between adjacent rows. A static
# partition set covers the device in two parts, so the 4 SM participants use the colon ratio,
# which places three partitions and leaves the last quarter of the device unassigned. The
# participant writes no stats file, so its own load is read from the run's log.
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


MX_ROWS = [("none", "GC", "exp1", "mobilenet_gc50", None, None),
           ("none", "MPS static", "exp9", "mps-alone", None, None),
           (r"\texttt{mem\_saxpy} at 4\,SMs", "GC", "exp9", "memsaxpy-4sm", "GB/s", "memory-interferers"),
           (r"\texttt{mem\_saxpy} at 4\,SMs", "MPS static", "exp9", "mps-memsaxpy-4sm", "GB/s", "memory-mps-interferers-4sm"),
           (r"\texttt{mem\_saxpy} at 8\,SMs", "GC", "exp9", "memsaxpy-8sm", "GB/s", "memory-interferers"),
           (r"\texttt{mem\_saxpy} at 8\,SMs", "MPS static", "exp9", "mps-memsaxpy-8sm", "GB/s", "memory-mps-interferers"),
           (r"\texttt{compute\_gemm} at 4\,SMs", "GC", "exp9", "computegemm-4sm", "TFLOP/s", "memory-interferers"),
           (r"\texttt{compute\_gemm} at 4\,SMs", "MPS static", "exp9", "mps-computegemm-4sm", "TFLOP/s", "memory-mps-interferers-4sm"),
           (r"\texttt{compute\_gemm} at 8\,SMs", "GC", "exp9", "computegemm-8sm", "TFLOP/s", "memory-interferers"),
           (r"\texttt{compute\_gemm} at 8\,SMs", "MPS static", "exp9", "mps-computegemm-8sm", "TFLOP/s", "memory-mps-interferers")]
_mx_ref = {"GC": ("exp1", "mobilenet_gc50"), "MPS static": ("exp9", "mps-alone")}
_mx_base = {k: must(val(e, c, "mobilenet", "median"), c) for k, (e, c) in _mx_ref.items()}
_mx_base_tail = {k: must(val(e, c, "mobilenet", "p99_minus_median"), c)
                 for k, (e, c) in _mx_ref.items()}

open_table([r"Configuration & Mechanism & Median (ms) & Median $\times$iso & Thr.\ (inf/s) & "
            r"tail (ms) & tail $\times$iso \\"], "llrrrrr",
           "What a participant's kind and budget cost a target holding its own 8\,SMs, under "
           "each mechanism. Each configuration appears twice, under green contexts and under "
           "MPS static partitions, and the two rows of a pair differ in the mechanism alone; "
           "the participant reaches the same rate under both. The configuration carries the "
           "load the participant reported for itself, and every multiplier is against the "
           "target alone under the same mechanism, the first pair of rows.",
           "tab:mem-matrix")
_mx_cfg = None
for _desc, _mech, _exp, _cfg, _unit, _fam in MX_ROWS:
    if _mx_cfg is not None and _desc != _mx_cfg:
        emit(r"\midrule")
    _mx_cfg = _desc
    _m = must(val(_exp, _cfg, "mobilenet", "median"), _cfg + " median")
    _t = must(val(_exp, _cfg, "mobilenet", "p99_minus_median"), _cfg + " tail")
    _thr = must(thr(_exp, _cfg, "mobilenet"), _cfg + " rate") or 0
    _load = itf_load(_fam, _cfg, _unit) if _fam else None
    _show = _desc + (" ({:.1f} {})".format(_load, _unit) if _load else "")
    emit(r"{} & {} & {} & {:.3f} & {:.0f} & {} & {:.3f} \\".format(
        _show, _mech, ms(_m), _m / _mx_base[_mech], _thr, ms(_t), _t / _mx_base_tail[_mech]))
close_table()

# ------------------------------------- the work-queue descriptor, with streams to declare
# The census found the earlier pair had almost nothing to declare - MobileNet adopts no auxiliary
# stream and ResNet one - which left scope and limit looking inert for want of a demand. These arms
# repeat the question on engines that do run streams: RetinaNet built for eight auxiliary streams
# and Inception for two, in three client configurations, with the declaration at the heaviest
# client's stream count and at the sum, under both scopes. The two RetinaNets of a pair are the
# same engine on the same share, so one column carries them and the caption records how far the
# other is from it.
WQAUX_ROWS = [
    ("RetinaNet + RetinaNet", "SMs only", "retinanet-retinanet-sms-only", ("retinanet_a", "retinanet_b")),
    ("RetinaNet + RetinaNet", "device-ctx 9", "retinanet-retinanet-device-ctx-9", ("retinanet_a", "retinanet_b")),
    ("RetinaNet + RetinaNet", "balanced 9", "retinanet-retinanet-balanced-9", ("retinanet_a", "retinanet_b")),
    ("RetinaNet + RetinaNet", "device-ctx 18", "retinanet-retinanet-device-ctx-18", ("retinanet_a", "retinanet_b")),
    ("RetinaNet + RetinaNet", "balanced 18", "retinanet-retinanet-balanced-18", ("retinanet_a", "retinanet_b")),
    ("RetinaNet + Inception", "SMs only", "retinanet-inception-sms-only", ("retinanet", "inception")),
    ("RetinaNet + Inception", "device-ctx 9", "retinanet-inception-device-ctx-9", ("retinanet", "inception")),
    ("RetinaNet + Inception", "balanced 9", "retinanet-inception-balanced-9", ("retinanet", "inception")),
    ("RetinaNet + Inception", "device-ctx 12", "retinanet-inception-device-ctx-12", ("retinanet", "inception")),
    ("RetinaNet + Inception", "balanced 12", "retinanet-inception-balanced-12", ("retinanet", "inception")),
    ("RetinaNet + RetinaNet + Inception", "SMs only", "retinanet-retinanet-inception-sms-only", ("retinanet_a", "retinanet_b", "inception")),
    ("RetinaNet + RetinaNet + Inception", "device-ctx 9", "retinanet-retinanet-inception-device-ctx-9", ("retinanet_a", "retinanet_b", "inception")),
    ("RetinaNet + RetinaNet + Inception", "balanced 9", "retinanet-retinanet-inception-balanced-9", ("retinanet_a", "retinanet_b", "inception")),
    ("RetinaNet + RetinaNet + Inception", "device-ctx 21", "retinanet-retinanet-inception-device-ctx-21", ("retinanet_a", "retinanet_b", "inception")),
    ("RetinaNet + RetinaNet + Inception", "balanced 21", "retinanet-retinanet-inception-balanced-21", ("retinanet_a", "retinanet_b", "inception")),
]
WQAUX_NETS = {"retinanet-retinanet-sms-only": ("retinanet_a", "retinanet_b"),
              "retinanet-retinanet-device-ctx-9": ("retinanet_a", "retinanet_b"),
              "retinanet-retinanet-balanced-9": ("retinanet_a", "retinanet_b"),
              "retinanet-retinanet-device-ctx-18": ("retinanet_a", "retinanet_b"),
              "retinanet-retinanet-balanced-18": ("retinanet_a", "retinanet_b"),
              "retinanet-inception-sms-only": ("retinanet", "inception"),
              "retinanet-inception-device-ctx-9": ("retinanet", "inception"),
              "retinanet-inception-balanced-9": ("retinanet", "inception"),
              "retinanet-inception-device-ctx-12": ("retinanet", "inception"),
              "retinanet-inception-balanced-12": ("retinanet", "inception"),
              "retinanet-retinanet-inception-sms-only": ("retinanet_a", "retinanet_b", "inception"),
              "retinanet-retinanet-inception-device-ctx-9": ("retinanet_a", "retinanet_b", "inception"),
              "retinanet-retinanet-inception-balanced-9": ("retinanet_a", "retinanet_b", "inception"),
              "retinanet-retinanet-inception-device-ctx-21": ("retinanet_a", "retinanet_b", "inception"),
              "retinanet-retinanet-inception-balanced-21": ("retinanet_a", "retinanet_b", "inception")}

open_table([
    r"Clients & Descriptor & \multicolumn{2}{c}{RetinaNet, 9 streams} & "
    r"\multicolumn{2}{c}{Inception, 3 streams} & Aggregate \\",
    r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
    r" & & Median (ms) & tail (ms) & Median (ms) & tail (ms) & (inf/s) \\",
], "llrrrrr",
   "The work-queue declaration and scope on engines that run streams. RetinaNet is built for "
   "eight auxiliary streams and runs nine, Inception for two and runs three, so unlike the "
   "earlier pair there is a demand for the descriptor's queue half to describe. Five arms per "
   "client configuration, one submitting thread, the same shares in all five. The two RetinaNets "
   "of a configuration are the same engine on the same share and their medians differ by at most "
   "0.6\\,\\%, so one column carries them. The aggregate is the sum of the clients' "
   "measured rates. The declaration is a hint to the driver and not a cap.",
   "tab:wq-aux")
_wqaux_clients = None
for _clients, _desc, _cfg, _nets in WQAUX_ROWS:
    if _wqaux_clients is not None and _clients != _wqaux_clients:
        emit(r"\midrule")
    _wqaux_clients = _clients
    _rn_net = _nets[0]                     # retinanet_a in the pairs, retinanet with Inception
    _rn = must(val("exp10", _cfg, _rn_net, "median"), _cfg + " " + _rn_net)
    _rnt = must(val("exp10", _cfg, _rn_net, "p99_minus_median"), _cfg + " " + _rn_net + " tail")
    if "inception" in _nets:
        _inc = ms(val("exp10", _cfg, "inception", "median"))
        _inct = ms(val("exp10", _cfg, "inception", "p99_minus_median"))
    else:
        _inc = _inct = "---"
    _agg = sum(thr("exp10", _cfg, _net) or 0 for _net in _nets)
    emit(r"{} & {} & {} & {} & {} & {} & {:.0f} \\".format(
        _clients, _desc, ms(_rn), ms(_rnt), _inc, _inct, _agg))
close_table()


# ------------------------------------------- memory contention: two co-runners at once
# One target at 8 SMs and two co-runners at 4 SMs each, so the device is fully used. One row per
# observation, in the shape the rest of the section uses: the target, the co-runners, the target's
# median and rate, the multiplier against its own isolated run, and its tail. The question - whether
# two co-runners cost the sum of what each costs alone, or the product - is arithmetic on those
# multipliers and belongs in the prose, not in a column invented for it.
ADDT_TARGETS = [("mobilenet", "retinanet", "inception"),
                ("resnet", "retinanet", "inception"),
                ("inception", "mobilenet", "retinanet"),
                ("retinanet", "mobilenet", "inception")]

open_table([r"Target & Co-runners & Median (ms) & Thr.\ (inf/s) & Median $\times$iso & "
            r"tail (ms) \\"], "llrrrr",
           "One target at 8\,SMs and two co-runners at 4\,SMs each, the device fully used. Each "
           "co-runner is measured alone before both are measured together, so the first row of a "
           "block is the reference the multipliers divide by. Every column is measured; whether "
           "the two co-runners cost the sum of their separate costs or the product is arithmetic "
           "on the multipliers and is stated in the text.",
           "tab:mem-additivity")
for _t, _a, _b in ADDT_TARGETS:
    _base = must(val("exp11", _t + "-alone", _t, "median"), _t + " alone")
    for _co, _desc in ((None, "none"),
                       (_a, _a),
                       (_b, _b),
                       ("%s and %s" % (_a, _b), "%s and %s" % (_a, _b))):
        _cfg = _t + "-alone" if _co is None else _t + "+" + _co.replace(" and ", "+")
        _m = must(val("exp11", _cfg, _t, "median"), _cfg + " median")
        _thr = thr("exp11", _cfg, _t) or 0
        _tail = must(val("exp11", _cfg, _t, "p99_minus_median"), _cfg + " tail")
        emit(r"{} & {} & {} & {:.0f} & {:.3f} & {} \\".format(
            _t, _desc, ms(_m), _thr, _m / _base, ms(_tail)))
    if _t != ADDT_TARGETS[-1][0]:
        emit(r"\midrule")
close_table()


sys.stdout.write("\n".join(OUT))
