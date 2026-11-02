# Offline diagnosis of the smooth model's residual: are the networks bursty at the probe's scale?
#   python3 analysis/explore/burst-diag.py
# Reads every window of the probe (stress-reporter-onoff) and the smooth88 outputs.
import csv, glob, statistics as st
import os
import numpy as np
os.chdir(os.path.join(os.environ.get("DATA", "/mnt/nvme/f.scarfato"), "bench"))

REP = "stress-reporter-onoff"
AGG = "aggregate"
FROZEN = "frozen"

def windows(cfg):
    runs = []
    for p in sorted(glob.glob(f"{REP}/{cfg}/r*_latency.txt")):
        v = [float(l.split()[3]) for l in open(p)
             if not l.startswith("#") and len(l.split()) == 4 and l.split()[0] == "65536"]
        runs.append(np.array(v))
    return runs

CFGS = ["spin", "smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100",
        "mobilenet-8sm", "inception-8sm", "resnet-8sm", "retinanet-8sm"]
W = {c: windows(c) for c in CFGS}

# 1. Window distribution per configuration: median over the five runs of each statistic, and the
#    range of the p99 - median gap across runs.
print("1. window distribution (ns): median p90 p99 gap=p99-med [gap min-max over runs]")
for c in CFGS:
    s = [(np.median(v), np.percentile(v, 90), np.percentile(v, 99)) for v in W[c]]
    g = [b - a for a, _, b in s]
    print(f"  {c:15} {st.median(x[0] for x in s):6.1f} {st.median(x[1] for x in s):6.1f} "
          f"{st.median(x[2] for x in s):6.1f}  gap {st.median(g):5.1f} [{min(g):5.1f}-{max(g):5.1f}]  runs {len(s)}")

# 2. The smooth scale for single windows: window median per level -> GB/s. Each window of a network
#    is placed on it; the time average of these is the stress the network applies on average.
cal = {r["config"]: r for r in csv.DictReader(open(f"{FROZEN}/stress-calibration-smooth88.csv"))}
lv = [float(cal[c]["level"]) for c in ["spin", "smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100"]]
med = [float(cal[c]["median"]) for c in ["spin", "smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100"]]
p99 = [float(cal[c]["p99"]) for c in ["spin", "smooth-D025", "smooth-D050", "smooth-D075", "smooth-D100"]]
print("\n2. equivalent stress (GB/s): from probe p99 | time-average of windows | share of windows above smooth 100% median | peak runs")
top = med[-1]
for c in CFGS[5:]:
    allw = np.concatenate(W[c])
    per_window = np.interp(allw, med, lv)            # clamps at both ends
    leq_p99 = np.interp(st.median(np.percentile(v, 99) for v in W[c]), p99, lv)
    above = allw > top
    # lengths of consecutive runs of windows above the top, within each run
    lens = []
    for v in W[c]:
        k = 0
        for x in v > top:
            if x: k += 1
            elif k: lens.append(k); k = 0
        if k: lens.append(k)
    print(f"  {c:15} p99 {leq_p99:6.1f} | mean {per_window.mean():5.1f} | above {100*above.mean():4.1f} % | "
          f"peak length median {st.median(lens) if lens else 0} windows, 90th pct {np.percentile(lens,90) if lens else 0:.0f}")

# 3. The stress each target actually felt: its measured p99 slowdown inverted on its own smooth
#    sensitivity curve, against what the probe said.
sens = {}
for r in csv.DictReader(open(f"{FROZEN}/stress-sensitivity-smooth88.csv")):
    sens.setdefault(r["target"], []).append((float(r["level"]), float(r["slowdown_p99"]), float(r["slowdown_median"])))
print("\n3. felt stress (GB/s): measured slowdown inverted on the target's smooth curve")
for r in csv.DictReader(open(f"{AGG}/stress-comparison-smooth88.csv")):
    pts = sorted(sens[r["target"]])
    L = [p[0] for p in pts]
    f99 = np.interp(float(r["measured_p99"]), [p[1] for p in pts], L)
    fmed = np.interp(float(r["measured_median"]), [p[2] for p in pts], L)
    over = " (above curve)" if float(r["measured_p99"]) > pts[-1][1] else ""
    print(f"  {r['target']:9} <- {r['corunner']:9} felt by p99 {f99:6.1f}{over} | felt by median {fmed:6.1f} | probe p99 says {float(r['L_eq_p99']):6.1f}")
