#!/usr/bin/env python3
"""The co-runner's DRAM demand against the median penalty it causes, and the split of that penalty.

    utility/mem-correlation.py

Eight co-runners have a measured penalty and a known demand, but only seven have a *measured*
demand: compute_gemm at 8 SMs is quoted nowhere, and its demand is not the same as at 4 SMs, so it
is left out rather than estimated into a coefficient.

The split in Table~\\ref{tab:mem-demand} is this:

    demand_share_i = slope * demand_i
    the_rest_i     = penalty_i - demand_share_i

with the slope and intercept from a least-squares fit of penalty against demand over the seven rows.
The two columns are therefore a model's decomposition, not two measurements, and the rest column
absorbs the fit's error: that is why ResNet at 8 SMs shows 15.3 where the other rows cluster near
10.6.

Each penalty is against the target alone at 8 SMs in the same family, which is what every other
table in this section does: the mechanism family reads 0.6752 ms and the additivity family 0.6718,
so the same arm differs by half a point between families.

The demand of the four networks is their own SoC reading while alone, converted to GB/s with the
copy calibration; that of the two synthetic participants is their own accounting. Inside a pair a
network realises 69 to 81 per cent of the demand it shows alone, so the x axis is demand rather than
realised traffic and the slope is softer than it would be against realised traffic.
"""
import os
import statistics
import sys

# (label, demand in GB/s, measured penalty in per cent, where the numbers come from)
ARMS = [
    ("compute_gemm at 4 SMs", 13.9, 14.5, "ncu: 4.98% of peak read, 4.0% write, copy calibration"),
    ("inception at 4 SMs",    20.8, 16.1, "SoC: inception alone at 25% of the SMs"),
    ("resnet at 4 SMs",       26.6, 17.4, "SoC: resnet alone at 25%"),
    ("retinanet at 4 SMs",    29.3, 19.0, "SoC: retinanet alone at 25%"),
    ("resnet at 8 SMs",       38.4, 26.6, "SoC: resnet alone at 50%"),
    ("mem_saxpy at 4 SMs",    54.9, 24.2, "the interferer's own accounting"),
    ("mem_saxpy at 8 SMs",    89.9, 37.3, "the interferer's own accounting"),
]


def fit(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return slope, my - slope * mx


def main():
    xs = [a[1] for a in ARMS]
    ys = [a[2] for a in ARMS]
    slope, inter = fit(xs, ys)
    print("\n  least squares over %d rows: penalty = %.2f %% + %.4f %% per GB/s"
          % (len(ARMS), inter, slope))
    print("  slope  = %s" % ("sum((x-xbar)(y-ybar)) / sum((x-xbar)^2)"))
    print("  xbar = %.3f GB/s, ybar = %.3f %%, slope = %.4f, intercept = %.3f"
          % (statistics.mean(xs), statistics.mean(ys), slope, inter))

    print("\n  %-24s %8s %9s %11s %10s %8s  %s"
          % ("co-runner", "demand", "penalty", "share", "the rest", "residual", "demand from"))
    print("  " + "-" * 108)
    worst = 0.0
    for label, d, pen, basis in ARMS:
        share = slope * d
        rest = pen - share
        resid = pen - (inter + share)
        worst = max(worst, abs(resid))
        print("  %-24s %7.1f  %+8.1f %+10.1f %+9.1f %+8.2f  %s"
              % (label, d, pen, share, rest, resid, basis))
    print("\n  the rest column clusters near the intercept, %.1f, and the widest miss is %.1f points"
          % (inter, worst))
    print("  the share column is slope * demand, so it is the fit's term and not a second measurement")

    dest = os.environ.get("DATA", "/mnt/nvme/f.scarfato") + "/bench/mem-demand.csv"
    with open(dest, "w") as fh:
        fh.write("co_runner,demand_gb_s,penalty_pct,demand_share_pct,the_rest_pct,residual_pts\n")
        for label, d, pen, _ in ARMS:
            fh.write("%s,%.1f,%.1f,%.2f,%.2f,%.2f\n"
                     % (label, d, pen, slope * d, pen - slope * d, pen - (inter + slope * d)))
    print("  wrote %s" % dest)


if __name__ == "__main__":
    main()
