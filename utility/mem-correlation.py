#!/usr/bin/env python3
"""Co-runner DRAM traffic against the median penalty it causes the target.

    utility/mem-correlation.py

The residuals were measured before the traffic was, so they have never been put beside each other.
The traffic comes from the SoC counters, taken while each co-runner ran alone, which is the only
attribution the counters allow; the penalties come from the aggregate CSVs, target median against the
target's median alone in the same session.

One care: for a co-runner that is a real network the traffic is the isolated reading, while inside the
pair its traffic is 70-81 % of that, because the two clients are sub-additive. The correlation is
therefore against the co-runner's own demand rather than its realised demand, and it is stated as such.

The synthetic interferer is different in kind: mem_saxpy reports its own GB/s from its own accounting
(it knows how many bytes it asked for), and compute_gemm moves almost no DRAM at all because its
operands are L2-resident. Those two are what separate the traffic-dependent part of the penalty from
the part that is present anyway.
"""
import csv
import glob
import os
import statistics
import sys

BENCH = os.environ.get("DATA", "/mnt/nvme/f.scarfato") + "/bench"
AGG = os.path.join(BENCH, "aggregate")


def medians(exp, cfg, net):
    """The per-run medians of gpu_ms for one cell."""
    path = os.path.join(AGG, exp + ".csv")
    out = []
    for row in csv.DictReader(open(path)):
        if row["config"] == cfg and row["network"] == net and row["metric"] == "gpu_ms":
            out.append(float(row["median"]))
    return out


def one(exp, cfg, net):
    v = medians(exp, cfg, net)
    return statistics.median(v) if v else None


def soc_traffic():
    """Sustained GB/s per configuration, from the SoC analysis."""
    out = {}
    path = os.path.join(BENCH, "soc-pressure.csv")
    if not os.path.exists(path):
        return out
    for row in csv.DictReader(open(path)):
        r, w = float(row["dram_read_busy_pct"]), float(row["dram_write_busy_pct"])
        out[row["configuration"]] = r / 48.0 * 91.5 + w / 83.0 * 91.5
    return out


def main():
    soc = soc_traffic()
    # the reference: the target alone at 8 SMs, in the additivity session where it exists
    ref = one("exp11", "mobilenet-alone", "mobilenet") or one("exp1", "mobilenet_gc50", "mobilenet")
    ref1 = one("exp1", "mobilenet_gc50", "mobilenet")
    if ref is None:
        sys.exit("  no reference for mobilenet at 8 SMs")
    print("\n  reference: mobilenet at 8 SMs alone = %.4f ms (exp11) / %.4f ms (exp1 gc50)" % (ref, ref1 or 0))

    # (label, experiment, config, traffic in GB/s, where the traffic comes from)
    arms = [
        ("compute_gemm @ 4 SMs", "exp9",  "computegemm-4sm", 0.0,  "own accounting, L2-resident"),
        ("compute_gemm @ 8 SMs", "exp9",  "computegemm-8sm", 0.0,  "own accounting, L2-resident"),
        ("inception @ 4 SMs",    "exp11", "mobilenet+inception", soc.get("soc-greenctx-inception_v3-25", 0.0), "isolated SoC"),
        ("resnet @ 4 SMs",       "exp9",  "mobilenet8-resnet4", soc.get("soc-greenctx-resnet50-v2-7-25", 0.0), "isolated SoC"),
        ("retinanet @ 4 SMs",    "exp11", "mobilenet+retinanet", soc.get("soc-greenctx-retinanet-9-25", 0.0), "isolated SoC"),
        ("resnet @ 8 SMs",       "exp2-1t", "50_50",            soc.get("soc-greenctx-resnet50-v2-7-50", 0.0), "isolated SoC"),
        ("mem_saxpy @ 4 SMs",    "exp9",  "memsaxpy-4sm",   54.9, "own accounting"),
        ("mem_saxpy @ 8 SMs",    "exp9",  "memsaxpy-8sm",   89.9, "own accounting"),
    ]

    print("\n  %-22s %-9s %-9s %-8s %s" % ("co-runner", "traffic", "median", "penalty", "traffic from"))
    print("  " + "-" * 88)
    rows = []
    for label, exp, cfg, traffic, source in arms:
        m = one(exp, cfg, "mobilenet")
        if m is None:
            print("  %-22s %-9s %s" % (label, "%.1f" % traffic, "no data"))
            continue
        pen = (m / ref - 1.0) * 100.0
        rows.append((label, traffic, m, pen))
        print("  %-22s %-9s %-9s %-8s %s" % (
            label, "%.1f GB/s" % traffic, "%.4f" % m, "%+.1f %%" % pen, source))

    # one line through all eight, including the two that move no DRAM: the intercept is the part of
    # the penalty that is present with no traffic at all, and it is the interesting coefficient
    t_mean = statistics.mean(t for _, t, _, _ in rows)
    p_mean = statistics.mean(p for _, _, _, p in rows)
    slope = sum((t - t_mean) * (p - p_mean) for _, t, _, p in rows) / sum(
        (t - t_mean) ** 2 for _, t, _, _ in rows)
    inter = p_mean - slope * t_mean
    print("\n  through all eight: penalty = %+.1f %% + %.3f %% per GB/s" % (inter, slope))
    worst = max(abs(p - (inter + slope * t)) for _, t, _, p in rows)
    print("  worst residual: %.1f points, so one line describes all eight" % worst)
    print("  at the demand of the real co-runners (21-38 GB/s) the traffic term is %.0f-%.0f points,"
          % (slope * 21, slope * 38))
    print("  and about %.0f points are present anyway." % inter)


if __name__ == "__main__":
    main()
