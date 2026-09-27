#!/usr/bin/env python3
"""Turn the interferer kernel traces into the numbers the memory subsection states.

Three traces of the same target, alone and with each interferer at 4 SMs, taken with nsys
kernel tracing (no root needed; the SoC counters are a separate instrument that does need
it). Each is sliced into inferences using the trace's own periodicity, since the kernel
sequence repeats once per inference, and each inference is split into the time its kernels
were running and the gaps between them. The gaps are further split by whether the next
kernel had already been enqueued when the previous one ended: if it had, the device had
work and had not started it; if it had not, the host still owed it.

Writes $DATA/bench/trace/mem-trace.csv, which is what results-tables.py renders. Profiling
perturbs, so these are diagnostics and not campaign measurements: they are quoted for the
shape of the split, which reproduces the campaign, not for absolute tails.

    utility/itf-trace.py [<dir with the .sqlite exports>]
"""
import collections
import csv
import os
import sqlite3
import statistics
import sys

PARTICIPANT = ("sgemm", "saxpy", "axpy")
TRACES = (("alone", "target alone"),
          ("mem4", "with mem_saxpy at 4 SMs"),
          ("compute4", "with compute_gemm at 4 SMs"))


def load(path):
    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
    namecol = next(c for c in ("demangledName", "shortName", "name", "kernelName")
                   if c in cols)
    first = con.execute("SELECT {} FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1".format(namecol)).fetchone()
    strings = dict(con.execute("SELECT id, value FROM StringIds")) if isinstance(first[0], int) else {}
    kernels = []
    for name, start, end, stream, corr in con.execute(
            "SELECT {}, start, end, streamId, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL "
            "WHERE end > start".format(namecol)):
        kernels.append((strings.get(name, str(name)) if isinstance(name, int) else name,
                        start, end, stream, corr))
    kernels.sort(key=lambda k: k[1])
    runtime = {}
    for start, end, corr in con.execute(
            "SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE end > start"):
        runtime.setdefault(corr, (start, end))
    return kernels, runtime


def period(names, hi=160):
    """Kernels per inference, from the shift at which the name sequence agrees with itself.

    The spacing between occurrences of the most common kernel does not work: MobileNet's
    depthwise convolution appears at many layers, so that spacing is the distance between
    layers and not between inferences.
    """
    n = min(len(names), 8000)
    best, kpi = -1.0, 1
    for p in range(16, hi):
        agree = sum(1 for i in range(n - p) if names[i] == names[i + p]) / (n - p)
        if agree > best:
            best, kpi = agree, p
    return kpi, best


def analyse(path):
    kernels, runtime = load(path)
    low = lambda n: any(k in (n or "").lower() for k in PARTICIPANT)
    target = [k for k in kernels if not low(k[0])]
    main = collections.Counter(k[3] for k in target).most_common(1)[0][0]
    target = [k for k in target if k[3] == main]

    kpi, agree = period([k[0] for k in target])

    inferences = []
    i = 0
    while i + kpi <= len(target):
        cycle = target[i:i + kpi]
        span = (cycle[-1][2] - cycle[0][1]) / 1000.0
        kernels_us = sum(k[2] - k[1] for k in cycle) / 1000.0
        queued = late = 0.0
        for j in range(len(cycle) - 1):
            gap = (cycle[j + 1][1] - cycle[j][2]) / 1000.0
            if gap <= 0:
                continue
            r = runtime.get(cycle[j + 1][4])
            if r and r[1] <= cycle[j][2]:
                queued += gap
            else:
                late += gap
        inferences.append((span, kernels_us, queued + late, late))
        i += kpi

    inferences.sort(key=lambda x: -x[0])
    slow = inferences[:max(1, len(inferences) // 100)]
    mean = lambda rows, c: statistics.mean(r[c] for r in rows)
    return dict(kpi=kpi, agree=agree, n=len(inferences), slow=len(slow),
                median=tuple(statistics.median(r[c] for r in inferences) for c in range(4)),
                tail=tuple(mean(slow, c) for c in range(4)),
                participant=sum(1 for k in kernels if low(k[0])))


def main():
    trace_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/nvme/f.scarfato/scratch/trc"
    out_dir = os.environ.get("DATA", "/mnt/nvme/f.scarfato") + "/bench/trace"
    rows = []
    for name, label in TRACES:
        path = os.path.join(trace_dir, name + ".sqlite")
        if not os.path.exists(path):
            sys.exit("[itf-trace] missing {}".format(path))
        a = analyse(path)
        for quantile, values in (("median inference", a["median"]),
                                 ("slowest 1%", a["tail"])):
            span, kernels_us, gaps, late = values
            rows.append(dict(trace=name, label=label, quantile=quantile,
                             inferences=a["n"], slow=a["slow"], kpi=a["kpi"],
                             agree=a["agree"], participant=a["participant"],
                             span=span, kernels=kernels_us, gaps=gaps, host_gaps=late))
        print("  {:<10} {:<6} inferences {:>5}  kernels/inference {} ({:.0f}% agree)  "
              "participant kernels {}".format(label, "", a["n"], a["kpi"],
                                              100 * a["agree"], a["participant"]))
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "mem-trace.csv")
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote {}".format(dest))


if __name__ == "__main__":
    main()
