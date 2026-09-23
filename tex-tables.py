#!/usr/bin/env python3
"""Emit the Experiment 5 tables as LaTeX, from the same data the aggregator uses.

Run from the runner directory (it imports aggregate.py and reads ../bench). Every
number is read out of the measurements, so nothing here can drift from the
aggregate's own tables.
"""
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate as A

# The data lives beside the RUNNER, not beside this script: locate it from the
# aggregate module, which always sits in the runner directory.
RUNNER = os.path.dirname(os.path.abspath(A.__file__))
DATA = A.load(os.path.join(RUNNER, "..", "bench"))
OUT = []
if not DATA:
    sys.exit("no data: nothing was read from the results tree")


def ms(v):
    return "---" if v is None else "{:.4f}".format(v)


def xr(a, b):
    return "---" if not a or not b else "{:.3f}".format(b / a)


def med(exp, cfg, net):
    r = A.aggr(DATA, exp, cfg, net, "gpu_ms", "median")
    return r[0] if r else None


def tail(exp, cfg, net):
    r = A.aggr(DATA, exp, cfg, net, "gpu_ms", "p99_minus_median")
    return r[0] if r else None


def col2(exp, cfg, net, c):
    r = A.aggr(DATA, exp, cfg, net, "gpu_ms", c)
    return r[0] if r else None


def obs(exp, cfg, net):
    try:
        return A.obs_max(DATA, exp, cfg, net)
    except Exception:
        return None


def iso(net):
    r = A.aggr(DATA, "exp1", net + "_whole", net, "gpu_ms", "p99_minus_median")
    return r[0] if r else None


def agg_tp(exp, cfg):
    vals = []
    for r in sorted(A.runs_of(DATA, exp, cfg)):
        t = A._run_throughput(DATA, exp, cfg, r)
        if t is not None:
            vals.append(t)
    return st.median(vals) if vals else None


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
            1000.0 / mv if mv else 0, r if cfg == "baseline" else r, ms(rv),
            1000.0 / rv if rv else 0,
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
            i = med("exp1", "{}_{}".format(net, "whole"), net)
            a, b = med("exp2", cfg, net), med("exp2-1t", cfg, net)
            emit("    {} & \\texttt{{{}}} & {} & {} & {} & {}{:+.1f}\\,\\% & {} & {}{:+.1f}\\,\\% \\\\".format(
                net.capitalize(), cfg.replace("_", "\\_"),
                "{} shared".format(sms) if cfg == "baseline" else str(sms),
                ms(i), ms(a), "", 100.0 * (a / i - 1) if a and i else 0,
                ms(b), "", 100.0 * (b / i - 1) if b and i else 0))
        emit("    \\midrule" if cfg != "25_75" else "    \\bottomrule")
    emit(r"""  \end{tabular}
  \caption{Experiment 5: residual interference recomputed with a single
  submitting thread, against the same isolated runs at the same SM count. Part of
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
    Configuration & Declared & \multicolumn{3}{c}{MobileNet V2-12} & \multicolumn{3}{c}{ResNet-50 v2-7} & \multicolumn{2}{c}{Aggregate} \\
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
  varied and nothing else, beside the mechanisms of Experiments 2 and 3 in
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

sys.stdout.write("\n".join(OUT))
