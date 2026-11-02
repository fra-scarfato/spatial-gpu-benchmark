"""The table writer and the accessors the table definitions share.

Moved out of results-tables.py unchanged, so the generated LaTeX is identical. `open_table`
takes a list of header lines, so a table with grouped columns can carry the multicolumn row,
its cmidrules and the second header row as separate entries.
"""
import os
import statistics as st

import aggregate as A

DATA = A.load(os.path.join(A.DATA_ROOT, "bench"))

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
