#!/usr/bin/env python3
"""Emit the Experiment 5 tables as LaTeX, from the same data the aggregator uses.

Run it from anywhere: it imports aggregate.py from its own directory and reads the
results tree under $DATA. Every
number is read out of the measurements, so nothing here can drift from the
aggregate's own tables.
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as A
from lib import tables as T          # the access layer, so that only one module reads the tree

# The data root comes from the aggregate module, so the two tools cannot
# disagree about where the results are.
DATA = A.load(os.path.join(A.DATA_ROOT, "bench"))
OUT = []
if not DATA:
    sys.exit("no data: nothing was read from the results tree")


def ms(v):
    return T.ms(v)


def xr(a, b):
    # this file calls it xr(numerator, denominator); the shared layer takes (value, reference)
    return T.xr(b, a)


def med(exp, cfg, net):
    return T.val(exp, cfg, net, "median")


def tail(exp, cfg, net):
    return T.val(exp, cfg, net, "p99_minus_median")


def col2(exp, cfg, net, c):
    return T.val(exp, cfg, net, c)


def obs(exp, cfg, net):
    try:
        return A.obs_max(DATA, exp, cfg, net)
    except Exception:
        return None


def iso(net):
    return T.ref(net)[0]          # the network alone on the whole device: its tail


def agg_tp(exp, cfg):
    vals = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        t = A._run_throughput(DATA, exp, cfg, r)
        if t is not None:
            vals.append(t)
    return st.median(vals) if vals else None


def net_tp(exp, cfg, net):
    return T.thr(exp, cfg, net)


def emit(s):
    OUT.append(s)


# --------------------------------------------------------------- 1. comparison
def t_compare():
    emit(r"""\begin{table}[H]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{l l r r r r r r r r}
    \toprule
    Configuration & Network & \multicolumn{3}{c}{Median (ms)} & \multicolumn{3}{c}{tail (ms)} & \multicolumn{2}{c}{obs-max (ms)} \\
    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}
     & & 2 threads & 1 thread & $\times$ & 2 threads & 1 thread & $\times$ & 2 threads & 1 thread \\
    \midrule""")
    for cfg in ("baseline", "50_50", "75_25", "25_75"):
        for net in ("mobilenet", "resnet"):
            emit("    \\texttt{{{}}} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                cfg.replace("_", "\\_"), net,
                ms(med("exp2", cfg, net)), ms(med("exp2-1t", cfg, net)), xr(med("exp2", cfg, net), med("exp2-1t", cfg, net)),
                ms(tail("exp2", cfg, net)), ms(tail("exp2-1t", cfg, net)), xr(tail("exp2", cfg, net), tail("exp2-1t", cfg, net)),
                ms(obs("exp2", cfg, net)), ms(obs("exp2-1t", cfg, net))))
        a, b = agg_tp("exp2", cfg), agg_tp("exp2-1t", cfg)
        emit("    \\texttt{{{}}} & aggregate & \\multicolumn{{2}}{{c}}{{{}}} & {} & \\multicolumn{{2}}{{c}}{{{}}} & & \\multicolumn{{2}}{{c}}{{inf/s}} \\\\".format(
            cfg.replace("_", "\\_"), "---" if a is None else "{:.0f}".format(a), xr(a, b),
            "---" if b is None else "{:.0f}".format(b)))
        emit("    \\midrule" if cfg != "25_75" else "    \\bottomrule")
    emit(r"""  \end{tabular}
  \caption{Experiment 5: Experiment 2's pair, two submitting threads against one.
  The medians and the aggregate throughput are unmoved; the tail is not. Every
  value is the median across the five runs of that run-level statistic.}
  \label{tab:exp5-compare}
\end{table}
""")


# ------------------------------------------------------------------ 2. the pair
def t_pair():
    emit(r"""\begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{l c r r c r r r}
    \toprule
    & \multicolumn{3}{c}{MobileNet V2-12} & \multicolumn{3}{c}{ResNet-50 v2-7} & \\
    \cmidrule(lr){2-4} \cmidrule(lr){5-7}
    Configuration & SMs & Median (ms) & Thr.\ (inf/s) & SMs & Median (ms) & Thr.\ (inf/s) & Aggregate \\
    \midrule""")
    for cfg, m, r in (("baseline", 16, 16), ("50_50", 8, 8), ("75_25", 12, 4), ("25_75", 4, 12)):
        mv, rv = med("exp2-1t", cfg, "mobilenet"), med("exp2-1t", cfg, "resnet")
        emit("    \\texttt{{{}}} & {} & {} & {:.0f} & {} & {} & {:.0f} & {:.0f} \\\\".format(
            cfg.replace("_", "\\_"), "{} shared".format(m) if cfg == "baseline" else str(m), ms(mv),
            net_tp("exp2-1t", cfg, "mobilenet") or 0,
            "{} shared".format(r) if cfg == "baseline" else str(r), ms(rv),
            net_tp("exp2-1t", cfg, "resnet") or 0,
            agg_tp("exp2-1t", cfg) or 0))
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{Experiment 5: Experiment 2's four configurations measured with a
  single submitting thread, otherwise unchanged. Read against
  Table~\ref{tab:exp2}, the medians and throughputs agree; the difference is in
  the dispersion, which Table~\ref{tab:exp5-compare} and
  Table~\ref{tab:exp5-pair-dispersion} give.}
  \label{tab:exp5-pair}
\end{table}
""")


# --------------------------------------------------------------- 3. residual
def t_residual():
    emit(r"""\begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{l l c r r r r r}
    \toprule
    Network & Configuration & SMs & Isolated (ms) & \multicolumn{2}{c}{2 threads} & \multicolumn{2}{c}{1 thread} \\
    \cmidrule(lr){5-6} \cmidrule(lr){7-8}
     & & & & Concurrent & Overhead & Concurrent & Overhead \\
    \midrule""")
    for cfg, m, r in (("baseline", 16, 16), ("50_50", 8, 8), ("75_25", 12, 4), ("25_75", 4, 12)):
        for net, sms in (("mobilenet", m), ("resnet", r)):
            # The isolated reference is the run at the SAME SM count, not the
            # whole-device one: that is the whole point of the table. The exp1
            # configurations are named by the percentage each network asked for.
            iso_cfg = "{}_{}".format(net, "whole" if sms == 16 else "gc{}".format(sms * 100 // 16))
            i = med("exp1", iso_cfg, net)
            a, b = med("exp2", cfg, net), med("exp2-1t", cfg, net)
            emit("    {} & \\texttt{{{}}} & {} & {} & {} & {}{:+.1f}\\,\\% & {} & {}{:+.1f}\\,\\% \\\\".format(
                net.capitalize(), cfg.replace("_", "\\_"),
                "{} shared".format(sms) if cfg == "baseline" else str(sms),
                ms(i), ms(a), "", 100.0 * (a / i - 1) if a and i else 0,
                ms(b), "", 100.0 * (b / i - 1) if b and i else 0))
        emit("    \\midrule" if cfg != "25_75" else "    \\bottomrule")
    emit(r"""  \end{tabular}
  \caption{Experiment 5: residual interference recomputed with a single
  submitting thread, against the same isolated runs at the same SM count.
  Every value is the median across the five runs of that run-level statistic. Part of
  what Table~\ref{tab:exp2-residual} attributed to interference that spatial
  separation could not remove was the submission handoff.}
  \label{tab:exp5-residual}
\end{table}
""")


# ---------------------------------------------------- 4. dispersion of the pair
def t_dispersion():
    emit(r"""\begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{l l l r r r r r r r}
    \toprule
    Configuration & Threads & Network & std & std $\times$iso & p95 & p99 & tail & tail $\times$iso & obs-max \\
    \midrule""")
    for cfg in ("baseline", "50_50", "75_25", "25_75"):
        for exp, thr in (("exp2", "2"), ("exp2-1t", "1")):
            for net in ("mobilenet", "resnet"):
                emit("    \\texttt{{{}}} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                    cfg.replace("_", "\\_"), thr, net,
                    ms(col2(exp, cfg, net, "std")),
                    xr(col2("exp1", net + "_whole", net, "std"), col2(exp, cfg, net, "std")),
                    ms(col2(exp, cfg, net, "p95")), ms(col2(exp, cfg, net, "p99")),
                    ms(tail(exp, cfg, net)), xr(iso(net), tail(exp, cfg, net)),
                    ms(obs(exp, cfg, net))))
        emit("    \\midrule" if cfg != "25_75" else "    \\bottomrule")
    emit(r"""  \end{tabular}
  \caption{Experiment 5: dispersion of Experiment 2's pair under both submission
  models, in the form of Table~\ref{tab:exp1-dispersion}. The pair alternates so
  the two models can be read against each other line by line.}
  \label{tab:exp5-pair-dispersion}
\end{table}
""")


# ------------------------------------------------ 5/6. the two mechanism families
def family(title, exp, one, pairs, label, caption):
    emit(r"""\begin{table}[H]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{l l r r r r r r r r}
    \toprule
    Configuration & Declared & \multicolumn{3}{c}{MobileNet V2-12 (median, ms)} & \multicolumn{3}{c}{ResNet-50 v2-7 (median, ms)} & \multicolumn{2}{c}{Aggregate (median, inf/s)} \\
    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-10}
     & & 2 th. & 1 th. & $\times$ & 2 th. & 1 th. & $\times$ & 2 th. & 1 th. \\
    \midrule""".replace("2 th.", "2 thr").replace("1 th.", "1 thr"))
    for cfg in pairs:
        limit = cfg.split("_")[1]
        mv, rv = med(exp, cfg, "mobilenet"), med(one, cfg, "mobilenet")
        mr, rr = med(exp, cfg, "resnet"), med(one, cfg, "resnet")
        emit("    \\texttt{{{}}} & {} & {} & {} & {} & {} & {} & {} & {:.0f} & {:.0f} \\\\".format(
            cfg.replace("_", "\\_"), limit, ms(mv), ms(rv), xr(mv, rv),
            ms(mr), ms(rr), xr(mr, rr), agg_tp(exp, cfg) or 0, agg_tp(one, cfg) or 0))
    emit("    \\bottomrule\n  \\end{tabular}\n  \\caption{" + caption +
         "}\n  \\label{" + label + "}\n\\end{table}\n")


# ------------------------------------------------------------------- 7. the frame
def t_frame():
    emit(r"""\begin{table}[H]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{l l l l r r r r r r}
    \toprule
    Configuration & Threads & Declared & Sharing & \multicolumn{3}{c}{MobileNet V2-12} & \multicolumn{3}{c}{ResNet-50 v2-7} \\
    \cmidrule(lr){5-7} \cmidrule(lr){8-10}
     & & concurrency & scope & tail (ms) & tail $\times$iso & obs-max & tail (ms) & tail $\times$iso & obs-max \\
    \midrule""")
    rows = [
        (r"\texttt{devctx\_8\_75\_25}",  "exp2", "exp4-1t", "8"),
        (r"\texttt{balanced\_8\_75\_25}", "exp2", "exp4-1t", "8"),
        (r"\texttt{balanced\_3\_75\_25}", "exp5", "exp5-1t", "3"),
        (r"\texttt{balanced\_2\_75\_25}", "exp5", "exp5-1t", "2"),
        (r"\texttt{balanced\_1\_75\_25}", "exp5", "exp5-1t", "1"),
    ]
    for name, exp, one, limit in rows:
        cfg = name.replace(r"\texttt{", "").replace("\\_", "_").replace("}", "")
        for thr, e in (("2", exp), ("1", one)):
            emit("    {} & {} & {} & \\texttt{{device\\_ctx}} & {} & {} & {} & {} & {} & {} \\\\".format(
                name if thr == "2" else "", thr, limit,
                ms(tail(e, cfg, "mobilenet")), xr(iso("mobilenet"), tail(e, cfg, "mobilenet")), ms(obs(e, cfg, "mobilenet")),
                ms(tail(e, cfg, "resnet")), xr(iso("resnet"), tail(e, cfg, "resnet")), ms(obs(e, cfg, "resnet"))))
        emit("    \\midrule")
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{Experiment 5: the tail at the 75/25 split with the submitting thread
  varied and nothing else, each value the median across the five runs of that
  run-level statistic, beside the mechanisms of Experiments 2 and 3 in
  Table~\ref{tab:exp4-tail}.}
  \label{tab:exp5-tail}
\end{table}
""")


t_compare()
t_pair()
t_residual()
t_dispersion()
family("work queues", "exp2", "exp4-1t",
       ["devctx_8_50_50", "balanced_8_50_50", "devctx_2_50_50", "balanced_2_50_50", "devctx_1_50_50",
        "devctx_8_75_25", "balanced_8_75_25", "devctx_2_75_25", "balanced_2_75_25", "devctx_1_75_25"],
       "tab:exp5-wq",
       "Experiment 5: the work queue arms of Experiment 4 with a single submitting thread. "
       "The scope remains inert, and the limit-1 arms no longer collapse: with one submitter, "
       "declaring fewer concurrent workloads than a run produces does not serialise it.")
family("auxiliary stream", "exp5", "exp5-1t",
       ["devctx_3_50_50", "balanced_3_50_50", "devctx_2_50_50", "balanced_2_50_50",
        "balanced_8_50_50", "balanced_32_50_50", "devctx_8_50_50",
        "devctx_3_75_25", "balanced_3_75_25", "devctx_2_75_25", "balanced_2_75_25",
        "balanced_8_75_25", "balanced_32_75_25", "devctx_8_75_25"],
       "tab:exp5-aux",
       "Experiment 5: the auxiliary-stream family with a single submitting thread. "
       "ResNet is built to allow one auxiliary stream, so a pair has three stream-ordered "
       "workloads; the declared concurrency that matches it is three.")
t_frame()

# ------------------------------------------------- the mechanism comparison
# Experiment 3's four-way table sets the mechanisms side by side at matched
# ratios. Experiment 5 repeats it with the green context submitted by a single
# thread, which is the only row that differs between the two — so both use these.
def fourway_rows(gc_exp):
    yield ("--- (none)", "MPS default", "exp3", "mps_default")
    for ratio, cfg in (("50/50", "50_50"), ("75/25", "75_25"), ("25/75", "25_75")):
        yield (ratio, "green context", gc_exp, cfg)
        yield (ratio, "MPS static", "exp3", "mps_static_" + cfg)
        yield (ratio, "MPS ATP", "exp3", "mps_atp_" + cfg)


def t_fourway(label, gc_exp, caption):
    """The four mechanisms at matched ratios, with the green context from gc_exp."""
    emit(r"""\begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{5pt}
  \begin{tabular}{l l r r r r r}
    \toprule
    Ratio & Mechanism & \multicolumn{2}{c}{MobileNet V2-12} & \multicolumn{2}{c}{ResNet-50 v2-7} & Aggregate \\
    \cmidrule(lr){3-4} \cmidrule(lr){5-6}
     & & Median (ms) & Thr.\ (inf/s) & Median (ms) & Thr.\ (inf/s) & (inf/s) \\
    \midrule""")
    for ratio, mech, exp, cfg in fourway_rows(gc_exp):
        if mech == "green context":
            emit("    \\midrule")
        mv, rv = med(exp, cfg, "mobilenet"), med(exp, cfg, "resnet")
        emit("    {} & {} & {} & {:.0f} & {} & {:.0f} & {:.0f} \\\\".format(
            ratio, mech, ms(mv), net_tp(exp, cfg, "mobilenet") or 0,
            ms(rv), net_tp(exp, cfg, "resnet") or 0, agg_tp(exp, cfg) or 0))
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{""" + caption + r"""}
  \label{""" + label + r"""}
\end{table}
""")


def t_fourway_dispersion(label, gc_exp, caption):
    """The dispersion of the same comparison."""
    emit(r"""\begin{table}[H]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{l l l r r r r r r r}
    \toprule
    Ratio & Mechanism & Network & std & std $\times$iso & p95 & p99 & tail & tail $\times$iso & obs-max \\
    \midrule""")
    for ratio, mech, exp, cfg in fourway_rows(gc_exp):
        if mech == "green context":
            emit("    \\midrule")
        for net in ("mobilenet", "resnet"):
            emit("    {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                ratio, mech, net,
                ms(col2(exp, cfg, net, "std")),
                xr(col2("exp1", net + "_whole", net, "std"), col2(exp, cfg, net, "std")),
                ms(col2(exp, cfg, net, "p95")), ms(col2(exp, cfg, net, "p99")),
                ms(tail(exp, cfg, net)), xr(iso(net), tail(exp, cfg, net)),
                ms(obs(exp, cfg, net))))
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{""" + caption + r"""}
  \label{""" + label + r"""}
\end{table}
""")

# The comparison tables. The last one belongs in Experiment 3, not here: it is a
# dispersion twin of that subsection's four-way table, kept in this script so every
# LaTeX table in the chapter is produced by one place.
t_fourway("tab:exp5-fourway", "exp2-1t",
          "Experiment 5: the mechanisms at matched ratios with the green context submitted by a single thread. The MPS rows are Experiment 3's and do not change, so only the green-context rows differ from Table~\\ref{tab:exp3-fourway}. Each value is the median across the five runs of that run-level statistic.")
t_fourway_dispersion("tab:exp5-fourway-dispersion", "exp2-1t",
                     'Experiment 5: the dispersion of the comparison above, with the green context submitted by a single thread, in the form of Table~\\ref{tab:exp1-dispersion}. Each value is the median across the five runs of that run-level statistic.')
t_fourway_dispersion("tab:exp3-fourway-dispersion", "exp2",
                     'Experiment 3: the dispersion of the comparison above, in the form of Table~\\ref{tab:exp1-dispersion}. Each value is the median across the five runs of that run-level statistic.')


# --------------------------------------------------------------- the census
# One row per network and engine build, so the table reads as a progression in the
# number of streams rather than as a single number per network. `streams` is what
# the engine reports per client: one main stream plus the auxiliary streams it
# adopted. The census takes the numbers from the same source as everything else --
# the measurements -- with the stream counts, which come from the engines, as the
# only literals here.
#
# (network, input, build label, streams per client, experiment key, config, network key)
CENSUS_ROWS = (
    ("MobileNet V2-12", "3$\\times$224$\\times$224", "default, 0 under every cap",
     1, "exp1", "mobilenet_whole", "mobilenet"),
    ("ResNet-50 v2-7", "3$\\times$224$\\times$224", "default",
     1, "exp1", "resnet_whole", "resnet"),
    ("", "", "cap 1", 2, "exp5", "aux1_whole", "resnet"),
    ("Inception-v3", "3$\\times$299$\\times$299", "default",
     1, "exp1", "inception_whole", "inception"),
    ("", "", "cap 1", 2, "exp1", "inception_aux1_whole", "inception"),
    ("", "", "cap 2", 3, "exp1", "inception_aux2_whole", "inception"),
    ("", "", "cap 4", 5, "exp1", "inception_aux4_whole", "inception"),
    ("", "", "cap 5, saturation", 6, "exp1", "inception_aux5_whole", "inception"),
    ("ViT-B/16-224", "3$\\times$224$\\times$224", "default, 0 under every cap",
     1, "exp1", "vit_whole", "vit"),
    ("RetinaNet-9", "3$\\times$480$\\times$640", "default",
     1, "exp1", "retinanet_whole", "retinanet"),
    ("", "", "cap 1", 2, "exp1", "retinanet_aux1_whole", "retinanet"),
    ("", "", "cap 8", 9, "exp1", "retinanet_aux8_whole", "retinanet"),
    ("", "", "cap 16, saturation", 10, "exp1", "retinanet_aux16_whole", "retinanet"),
)


def t_census_rates():
    """What each build costs and delivers, alone on the whole device."""
    emit(r"""\begin{table}[H]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{l l l r r r}
    \toprule
    Network & Input & Build & Streams & Median (ms) & Thr.\ (inf/s) \\
    \midrule""")
    for net, inp, build, streams, exp, cfg, key in CENSUS_ROWS:
        m = med(exp, cfg, key)
        t = net_tp(exp, cfg, key)
        emit("    {} & {} & {} & {} & {} & {:.0f} \\\\".format(
            net, inp, build, streams, ms(m), t if t else 0))
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{The auxiliary-stream census, as cost and throughput. One row per build,
  measured alone on the whole device, five runs each; the values are medians across
  the five runs of that run-level statistic. \emph{Streams} is what the built engine
  reports per client --- one main stream plus the auxiliary streams it adopted --- so
  the row above and the row below it differ only in how many streams the engine was
  allowed and chose to use. MobileNet and the ViT adopt none under every cap; ResNet
  one; Inception up to five and RetinaNet up to nine.}
  \label{tab:census}
\end{table}
""")


def t_census_dispersion():
    """The same builds, by dispersion: what the streams cost in predictability."""
    emit(r"""\begin{table}[H]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \begin{tabular}{l r r r r r r r r}
    \toprule
    Network & Streams & std & std $\times$iso & p95 & p99 & tail & tail $\times$iso & obs-max \\
    \midrule""")
    for net, inp, build, streams, exp, cfg, key in CENSUS_ROWS:
        base = med("exp1", key + "_whole", key)
        emit("    {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
            net if net else "\\quad " + build, streams,
            ms(col2(exp, cfg, key, "std")),
            xr(col2("exp1", key + "_whole", key, "std"), col2(exp, cfg, key, "std")),
            ms(col2(exp, cfg, key, "p95")), ms(col2(exp, cfg, key, "p99")),
            ms(tail(exp, cfg, key)), xr(iso(key), tail(exp, cfg, key)),
            ms(obs(exp, cfg, key))))
    emit(r"""    \bottomrule
  \end{tabular}
  \caption{The same builds, by dispersion, in the form of
  Table~\ref{tab:exp1-dispersion}. Each row is the build named in the row above it
  in Table~\ref{tab:census}, and $\times$iso is against that network's own
  whole-device run. Reading down a network's block shows what the streams cost:
  the serial build has the thinnest tail of the block, and it grows with the streams
  while the median barely moves.}
  \label{tab:census-dispersion}
\end{table}
""")

# The census, as rates and as dispersion. Both read the same measurements as
# everything else; only the stream counts are literals, because they come from
# the built engines rather than from a run.
t_census_rates()
t_census_dispersion()

sys.stdout.write("\n".join(OUT))
