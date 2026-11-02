#!/usr/bin/env python3
"""profiles.py - every number the memory subsection takes from a profiled run.

    python3 analysis/profiles.py            read $DATA/profiles, write $DATA/profiles/derived/*.csv

It reads what measure/profile.sh wrote and nothing else, and every quantity it prints is defined in
docs/MEASUREMENTS.md. In outline:

  extract   each <arm>.nsys-rep is exported to sqlite ONCE, and three compact files are kept beside
            it: <arm>.kernels.csv.gz (every kernel: start, end, stream, correlation id, blocks,
            name), <arm>.launches.csv.gz (every launch call: correlation id, host start and end,
            thread) and, for a counter run, <arm>.dram.csv.gz (the DRAM read and write series).
            The sqlite is deleted afterwards: it is hundreds of MB and everything is in the three.
  dram      bytes per inference and the run-average GB/s of each network alone on each budget,
            and the traffic of a pair against the sum of its two networks alone
  traces    per inference of the target: span, kernel time, gaps the device owed (queued) and gaps
            the host owed (host-late); the same target kernel with and without a co-runner kernel
            overlapping it; the fixed and proportional parts of the slowdown; launch-call times
  demand    each co-runner's DRAM demand beside the median penalty the campaign measured for it
"""
import glob
import gzip
import os
import re
import sqlite3
import statistics
import subprocess
import sys

import numpy as np
import pandas as pd

DATA = os.environ.get("DATA", "/mnt/nvme/f.scarfato")
PROF = os.path.join(DATA, "profiles")
BENCH = os.path.join(DATA, "bench")
DERIVED = os.path.join(PROF, "derived")
NSYS = os.environ.get("NSYS", "/usr/local/bin/nsys")

# The calibration behind every GB/s is measured, not assumed: calibrate() reads it from the
# profiled device-probe run in profiles/calibration (see MEASUREMENTS.md). These are filled there.
COPY_GB_S = COPY_READ_PCT = COPY_WRITE_PCT = None
SOC_HZ = 10000.0
WARMUP = 200                      # the runner's untimed warm-up inferences, per network
TARGET_KPI = 55                   # MobileNet's kernels per inference, found from the traces


# ------------------------------------------------------------------ extract

def extract(rep, parts=("kernels", "launches")):
    """The compact files of a report that are asked for, exporting the report at most once."""
    stem = rep[:-len(".nsys-rep")]
    missing = [p for p in parts if not os.path.exists("%s.%s.csv.gz" % (stem, p))]
    # a counter file written before the GPU's own counters were kept is extracted again
    if "dram" in parts and "dram" not in missing and \
            "gpu_read_pct" not in pd.read_csv(stem + ".dram.csv.gz", nrows=1).columns:
        missing.append("dram")
    if not missing:
        return
    db = stem + ".sqlite"
    if not os.path.exists(db):
        print("  exporting %s (minutes)" % os.path.basename(rep), flush=True)
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true",
                        "--output", db, rep], check=True, capture_output=True)
    con = sqlite3.connect(db)
    # The kernels and the launch calls always, since the export is the expensive part.
    pd.read_sql("""SELECT k.start, k.end, k.streamId AS stream, k.correlationId AS corr,
                          k.gridX*k.gridY*k.gridZ AS blocks, s.value AS name
                   FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName""",
                con).to_csv(stem + ".kernels.csv.gz", index=False)
    pd.read_sql("""SELECT r.correlationId AS corr, r.start, r.end, r.globalTid AS tid, s.value AS api
                   FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId
                   WHERE s.value LIKE '%LaunchKernel%'""", con).to_csv(
                       stem + ".launches.csv.gz", index=False)
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "SOC_METRICS" in tables:
        pd.DataFrame(soc_series(con)).to_csv(stem + ".dram.csv.gz", index=False)
    con.close()
    os.remove(db)


def soc_series(con):
    """The DRAM counters of the whole SoC and the GPU's own, one row per 100 us sample.

    The export names no columns, so SOC_METRICS is read by position: (start, end, domain, index,
    value), and TARGET_INFO_SOC_METRICS maps each index to its name, at positions 3 and 4. The DRAM
    pair counts every client of the memory controller, the CPU among them; the GPU pair counts the
    GPU alone, and is the one every quantity in GB is built from."""
    names = {row[4].split(" [")[0]: row[3] for row in
             con.execute("SELECT * FROM TARGET_INFO_SOC_METRICS")}
    cols = {"read_pct": "DRAM Read Throughput", "write_pct": "DRAM Write Throughput",
            "gpu_read_pct": "GPU Read Throughput", "gpu_write_pct": "GPU Write Throughput"}
    by_idx = {}
    for row in con.execute("SELECT * FROM SOC_METRICS"):
        by_idx.setdefault(row[3], []).append(row[4])
    n = min(len(by_idx[names[c]]) for c in cols.values())
    return {k: by_idx[names[c]][:n] for k, c in cols.items()}


# ------------------------------------------------------------------ calibration

def calibrate():
    """GB/s per counter percent, from device-probe's device-to-device copies profiled with the counters.

    The copies' own rate is their bytes over their summed durations, from the trace. The counters'
    reading is the median over the copy plateau, the samples where read exceeds 20 per cent: the
    copy reads and writes every byte once, so both counters sit on a plateau while it runs. Write-
    only samples outside the copies (device-probe's memsets, near 80 per cent write and 1 per cent read)
    are excluded by the same condition, and they are what an earlier reading took for the copy.
    """
    global COPY_GB_S, COPY_READ_PCT, COPY_WRITE_PCT
    d = os.path.join(PROF, "calibration")
    db = os.path.join(d, "devprobe.sqlite")       # data names: the probe was called devprobe then
    if not os.path.exists(db):
        subprocess.run([NSYS, "export", "--type", "sqlite", "--force-overwrite=true", "--output",
                        db, os.path.join(d, "devprobe.nsys-rep")], check=True, capture_output=True)
    con = sqlite3.connect(db)
    n, nbytes, ns = con.execute("SELECT count(*), sum(bytes), sum(end - start) FROM "
                                "CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind = 8").fetchone()
    d = pd.DataFrame(soc_series(con), dtype=float)
    con.close()
    r, w = d.gpu_read_pct, d.gpu_write_pct
    plateau = r > 20
    COPY_GB_S = nbytes / ns
    COPY_READ_PCT = float(r[plateau].median())
    COPY_WRITE_PCT = float(w[plateau].median())
    return pd.DataFrame([dict(copies=n, gb_copied=round(nbytes / 1e9, 2),
                              copy_seconds=round(ns / 1e9, 4), copy_gb_s=round(COPY_GB_S, 1),
                              plateau_samples=int(plateau.sum()), read_pct=COPY_READ_PCT,
                              write_pct=COPY_WRITE_PCT,
                              peak_read_gb_s=round(COPY_GB_S * 100 / COPY_READ_PCT, 1),
                              peak_write_gb_s=round(COPY_GB_S * 100 / COPY_WRITE_PCT, 1),
                              write_samples_above_60_pct=int((w > 60).sum()),
                              their_read_pct=float(r[w > 60].median()))])


# ------------------------------------------------------------------ the runner's own report

def report(log):
    """Per network: timed inferences and loop throughput; and the co-runner's own figures."""
    txt = open(log, errors="replace").read()
    nets = {}
    for block in re.split(r"\n--- network '", txt)[1:]:
        name = block.split("'", 1)[0]
        n = re.search(r"inferences: (\d+)\s+duration", block)
        rate = re.search(r"Throughput \(whole loop\): ([0-9.]+)", block)
        if n:
            nets[name] = (int(n.group(1)), float(rate.group(1)) if rate else float("nan"))
    co = {}
    for key, pat in (("gb_s", r"bytes moved:\s+[0-9.]+ GB \(([0-9.]+) GB/s\)"),
                     ("tflop_s", r"arithmetic:\s+[0-9.]+ GFLOP \(([0-9.]+) TFLOP/s\)"),
                     ("seconds", r"units:\s+\d+ in ([0-9.]+) s")):
        m = re.search(pat, txt)
        if m:
            co[key] = float(m.group(1))
    return nets, co


def stats_median(path):
    for line in open(path):
        f = line.split(",")
        if f[0] == "gpu_ms":
            return float(f[3])
    return float("nan")


# ------------------------------------------------------------------ dram

def dram_gb(stem):
    """Gigabytes the GPU moved to and from DRAM over the run: the integral of its two counters.

    The GPU's own counters, not the SoC's DRAM pair, which also counts the CPU and never reads
    below 1 per cent. The GPU pair reads 0 when the GPU is idle, so no floor is subtracted."""
    d = pd.read_csv(stem + ".dram.csv.gz")
    r, w = d.gpu_read_pct, d.gpu_write_pct
    gb = (r.sum() / COPY_READ_PCT + w.sum() / COPY_WRITE_PCT) * COPY_GB_S / SOC_HZ
    busy = lambda s: s[s > 1].mean() if (s > 1).any() else 0.0
    return gb, r.max(), busy(r), busy(w)


def dram_tables():
    rows = []
    # soc-new and soc-mps: single networks alone at 50 % (green context, MPS static); soc-mps-sizes:
    # the same under MPS static at 25, 37, 63 and 75 % (the geometry test)
    for rep in sorted(glob.glob(os.path.join(PROF, "soc-budget", "*.nsys-rep")) +
                      glob.glob(os.path.join(PROF, "soc-new", "*.nsys-rep")) +
                      glob.glob(os.path.join(PROF, "soc-mps", "*.nsys-rep")) +
                      glob.glob(os.path.join(PROF, "soc-mps-sizes", "*.nsys-rep"))):
        stem = rep[:-len(".nsys-rep")]
        extract(rep, ("dram",))
        nets, _ = report(stem + ".log")
        gb, peak, busy_r, busy_w = dram_gb(stem)
        n = sum(v[0] + WARMUP for v in nets.values())
        rate = sum(v[1] for v in nets.values())
        mb = gb * 1e3 / n
        rows.append(dict(arm=os.path.basename(stem), networks=len(nets), inferences=n,
                         inf_per_s=round(rate, 1), gb_total=round(gb, 2),
                         mb_per_inference=round(mb, 2) if len(nets) == 1 else "",
                         gb_s_run_average=round(mb * rate / 1e3, 1) if len(nets) == 1 else "",
                         read_peak_pct=peak, read_busy_pct=round(busy_r, 4),
                         write_busy_pct=round(busy_w, 4)))
    dram = pd.DataFrame(rows)
    single = {r.arm.replace("soc-greenctx-", ""): r.mb_per_inference
              for r in dram.itertuples() if r.networks == 1}
    pairs = []
    for r in dram.itertuples():
        if r.networks < 2:
            continue
        nets, _ = report(os.path.join(PROF, "soc-budget", r.arm + ".log"))
        parts = r.arm.replace("soc-greenctx-", "").split("_")
        predicted = 0.0
        for part, (name, (n, _)) in zip(parts, nets.items()):
            predicted += (n + WARMUP) * single[part] / 1e3
        pairs.append(dict(arm=r.arm, gb_measured=round(r.gb_total, 1),
                          gb_sum_of_alone=round(predicted, 1),
                          ratio=round(r.gb_total / predicted, 3)))
    return dram, pd.DataFrame(pairs)


def l2_traffic():
    """The soc-l2 arms (tab:l2-evidence): the GPU's DRAM GB/s over each profiled session. Each
    co-runner of the L2 experiments ran alone (alone-<co-runner>) and next to MobileNet on 8 SMs
    (mobilenet-<co-runner>); the table takes each against the spin arm of its kind, whose kernel
    touches no memory. The session, not the co-runner's own run time, is the denominator, so the
    target's traffic and the co-runner's are averaged over the same window."""
    rows = []
    for rep in sorted(glob.glob(os.path.join(PROF, "soc-l2", "*.nsys-rep"))):
        stem = rep[:-len(".nsys-rep")]
        extract(rep, ("dram",))
        gb = dram_gb(stem)[0]
        seconds = len(pd.read_csv(stem + ".dram.csv.gz")) / SOC_HZ
        rows.append(dict(arm=os.path.basename(stem), gb=round(gb, 3), seconds=round(seconds, 2),
                         gb_s=round(gb / seconds, 3)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ traces

def target_stream(k, kpi=TARGET_KPI):
    """The stream whose inferences are kpi kernels long, counted between its first kernels."""
    first = k[k.name.str.contains("copyPacked")]
    for stream, g in first.groupby("stream"):
        idx = k.index[k.stream == stream]
        pos = np.searchsorted(idx, g.index)
        if len(pos) > 10 and statistics.median(np.diff(pos)) == kpi:
            return stream
    raise ValueError("no stream with %d kernels per inference" % kpi)


def covered(a, b, os_, oe):
    """Fraction of each interval [a, b] covered by the co-runner's kernels (os_, oe sorted)."""
    i0 = np.searchsorted(oe, a)
    i1 = np.searchsorted(os_, b)
    out = np.zeros(len(a))
    for j in range(len(a)):
        s = 0
        for m in range(i0[j], i1[j]):
            s += max(0, min(b[j], oe[m]) - max(a[j], os_[m]))
        out[j] = s / max(b[j] - a[j], 1)
    return out


def trace_tables():
    split, overlap, launch, perkernel = [], [], [], []
    ref = None
    # The five traces of one session, and the side-64 footprint arm, whose report has a CUDA trace
    # too: the co-runner that launches tens of thousands of small kernels a second.
    for arm, rep in [(a, os.path.join(PROF, "traces", a + ".nsys-rep"))
                     for a in ("alone", "mem4", "compute4", "spin4", "resnet4")] + \
                    [("gemm64", os.path.join(PROF, "footprint", "A1-l1.nsys-rep"))]:
        if not os.path.exists(rep):
            continue
        stem = rep[:-len(".nsys-rep")]
        extract(rep)
        k = pd.read_csv(stem + ".kernels.csv.gz").sort_values("start").reset_index(drop=True)
        la = pd.read_csv(stem + ".launches.csv.gz")
        k = k.merge(la[["corr", "end", "tid"]].rename(columns={"end": "launch_end"}),
                    on="corr", how="left")
        ts = target_stream(k)
        t = k[k.stream == ts].reset_index(drop=True)
        o = k[k.stream != ts].sort_values("start")
        # An inference begins at each copyPackedKernel; the warm-up (the first 200 inferences) and the
        # last, possibly cut, inference are dropped, and only inferences of exactly 55 kernels are
        # kept, so a missing kernel can never shift the positions of the ones after it.
        t = t.iloc[WARMUP * TARGET_KPI:].reset_index(drop=True)
        t["inf"] = t.name.str.contains("copyPacked").cumsum()
        t = t[(t.inf > 0) & (t.inf < t.inf.max())]
        size = t.groupby("inf").inf.transform("size")
        whole = (size == TARGET_KPI)
        kept = whole.groupby(t.inf).first().mean()
        t = t[whole].reset_index(drop=True)
        t["pos"] = t.groupby("inf").cumcount()
        t["dur"] = t.end - t.start
        gap = t.start - t.end.shift(1)
        late = (t.launch_end - t.end.shift(1)).clip(lower=0)
        t["gap"] = gap.where(t.pos > 0, 0).clip(lower=0)
        t["host_late"] = np.minimum(t.gap, late.where(t.pos > 0, 0))
        per = t.groupby("inf").agg(first=("start", "min"), last=("end", "max"),
                                   kernels=("dur", "sum"), gaps=("gap", "sum"),
                                   host_late=("host_late", "sum"))
        per["span"] = per["last"] - per["first"]
        per["queued"] = per.gaps - per.host_late
        med = per[["span", "kernels", "gaps", "queued", "host_late"]].median() / 1e3
        split.append(dict(arm=arm, inferences=len(per), **{c + "_us": round(v, 1)
                                                         for c, v in med.items()},
                          gap_median_us=round(t.gap[t.pos > 0].median() / 1e3, 2),
                          # the check on the slicing: the share of inferences with all 55 kernels
                          whole_inferences=round(kept, 4)))
        free = t.groupby("pos").dur.median()
        if arm == "alone":
            ref = free
        tid = t.tid.mode()[0]
        for who, g in la.groupby("tid"):
            launch.append(dict(arm=arm, thread="target" if who == tid else "co-runner",
                               calls=len(g), mean_us=round((g.end - g.start).mean() / 1e3, 2)))
        if len(o) == 0:
            continue
        os_, oe = o.start.values, o.end.values
        t["cov"] = covered(t.start.values, t.end.values, os_, oe)
        g = t[(t.pos > 0) & (t.gap > 0)]
        gcov = covered((g.start - g.gap).values, g.start.values, os_, oe)
        # The co-runner's launches already returned to the host and not yet started on the device,
        # at the moment each gap opens: zero means its queue was empty.
        returned = np.sort(o.launch_end.dropna().values)
        pending = (np.searchsorted(returned, (g.start - g.gap).values)
                   - np.searchsorted(os_, (g.start - g.gap).values))
        on = t[t["cov"] > 0.99].groupby("pos").dur.agg(["median", "size"])
        on = on[on["size"] >= 30]["median"]
        if len(on) < TARGET_KPI:
            continue          # the co-runner overlapped too few kernels for a per-position reading
        off = t[t["cov"] < 0.01].groupby("pos").dur.agg(["median", "size"])
        within = off[off["size"] >= 30]["median"]
        # The free reference: the same run when the co-runner was idle often enough, otherwise the
        # alone trace taken in the same session. The column says which.
        basis, base = ("same run", within) if len(within) == TARGET_KPI else ("alone trace", ref)
        both = pd.concat([on.rename("on"), base.rename("free"),
                          t.groupby("pos").blocks.first()], axis=1).dropna()
        d = (both.on - both.free) / 1e3
        b, a = np.polyfit(both.free / 1e3, d, 1)
        overlap.append(dict(arm=arm, co_runner_covers=round((t["cov"] * t.dur).sum() / t.dur.sum(), 3),
                            free_basis=basis, kernels_free_us=round(both.free.sum() / 1e3, 1),
                            kernels_overlapped_us=round(both.on.sum() / 1e3, 1),
                            fixed_us_per_kernel=round(a, 3), proportional=round(b, 4),
                            gap_us_co_busy=round(g.gap[gcov > 0.99].median() / 1e3, 2),
                            gap_us_co_busy_none_queued=round(
                                g.gap[(gcov > 0.99) & (pending <= 0)].median() / 1e3, 2),
                            gaps_co_busy_none_queued=int(((gcov > 0.99) & (pending <= 0)).sum()),
                            gap_us_co_idle=round(g.gap[gcov < 0.01].median() / 1e3, 2)
                            if (gcov < 0.01).sum() >= 100 else ""))
        for p, r in both.iterrows():
            perkernel.append(dict(arm=arm, pos=p, name=t.name[t.pos == p].iloc[0][:40],
                                  blocks=int(r.blocks), free_us=round(r.free / 1e3, 2),
                                  overlapped_us=round(r.on / 1e3, 2)))
    return (pd.DataFrame(split), pd.DataFrame(overlap), pd.DataFrame(launch),
            pd.DataFrame(perkernel))


# ------------------------------------------------------------------ demand

def campaign_median(family, config, net="mobilenet"):
    """Median of the five run medians, from the raw run files, as aggregate.py computes it."""
    d = os.path.join(BENCH, family, config)
    files = sorted(glob.glob(os.path.join(d, "r*_%s_stats.csv" % net))) or \
        sorted(f for f in glob.glob(os.path.join(d, "r*_stats.csv")) if re.search(r"/r\d+_stats", f))
    return statistics.median(stats_median(f) for f in files) if files else float("nan")


def demand_table(dram):
    gbs = {r.arm.replace("soc-greenctx-", ""): r.gb_s_run_average
           for r in dram.itertuples() if r.networks == 1}
    alone_gc = campaign_median("isolated-scaling", "mobilenet_gc50")
    alone_add = campaign_median("interference-additivity", "mobilenet-alone")

    def saxpy(cfg):
        rates = [report(f)[1].get("gb_s") for f in glob.glob(os.path.join(
            BENCH, "memory-interferers", cfg, "r*.log"))]
        return statistics.median(r for r in rates if r)

    # compute_gemm's demand by subtraction, from the counter pair taken in one session: the device's
    # traffic beside it minus the target's own bytes per inference alone, over the co-runner's time.
    co_gemm = float("nan")
    s = os.path.join(PROF, "soc-itf")
    if os.path.exists(os.path.join(s, "compute4.nsys-rep")):
        for arm in ("alone", "compute4"):
            extract(os.path.join(s, arm + ".nsys-rep"), ("dram",))
        n0 = sum(v[0] + WARMUP for v in report(os.path.join(s, "alone.log"))[0].values())
        mb0 = dram_gb(os.path.join(s, "alone"))[0] / n0
        nets, co = report(os.path.join(s, "compute4.log"))
        n1 = sum(v[0] + WARMUP for v in nets.values())
        co_gemm = (dram_gb(os.path.join(s, "compute4"))[0] - n1 * mb0) / co["seconds"]

    rows = [
        ("spin at 4 SMs", 0.0, "none by construction", "memory-interferers", "spin-4sm", alone_gc),
        ("spin at 8 SMs", 0.0, "none by construction", "memory-interferers", "spin-8sm", alone_gc),
        ("compute_gemm at 4 SMs", co_gemm, "counters, by subtraction", "memory-interferers",
         "computegemm-4sm", alone_gc),
        ("inception at 4 SMs", gbs.get("inception_v3-25"), "counters, alone",
         "interference-additivity", "mobilenet+inception", alone_add),
        ("resnet at 4 SMs", gbs.get("resnet50-v2-7-25"), "counters, alone",
         "memory-budget-greenctx", "mobilenet8-resnet4", alone_gc),
        ("retinanet at 4 SMs", gbs.get("retinanet-9-25"), "counters, alone",
         "interference-additivity", "mobilenet+retinanet", alone_add),
        ("resnet at 8 SMs", gbs.get("resnet50-v2-7-50"), "counters, alone",
         "pair-greenctx-1submit", "50_50", alone_gc),
        ("mem_saxpy at 4 SMs", saxpy("memsaxpy-4sm"), "its own byte count",
         "memory-interferers", "memsaxpy-4sm", alone_gc),
        ("mem_saxpy at 8 SMs", saxpy("memsaxpy-8sm"), "its own byte count",
         "memory-interferers", "memsaxpy-8sm", alone_gc),
    ]
    out = []
    for label, gb, basis, fam, cfg, ref in rows:
        med = campaign_median(fam, cfg)
        out.append(dict(co_runner=label, demand_gb_s=round(gb, 1) if gb == gb else "",
                        demand_basis=basis, target_median_ms=round(med, 4),
                        reference_ms=round(ref, 4), penalty_pct=round((med / ref - 1) * 100, 1)))
    return pd.DataFrame(out)


def fits(demand):
    d = demand[(demand.demand_gb_s != "") & demand.penalty_pct.notna()].copy()
    d["x"] = d.demand_gb_s.astype(float)
    rows = []
    groups = {"all": d,
              "synthetic": d[d.co_runner.str.match(r"(spin|compute_gemm|mem_saxpy)")],
              "networks": d[~d.co_runner.str.match(r"(spin|mem_saxpy)")]}
    for name, g in groups.items():
        if len(g) < 3:
            continue
        b, a = np.polyfit(g.x, g.penalty_pct, 1)
        r2 = np.corrcoef(g.x, g.penalty_pct)[0, 1] ** 2
        rows.append(dict(rows=name, n=len(g), intercept_pct=round(a, 1),
                         pct_per_gb_s=round(b, 3), r2=round(r2, 2)))
    return pd.DataFrame(rows)


def footprint_table():
    """The footprint arms: one profiled run each, read for their medians and the co-runner's rates.
    The GEMM side changes the operand footprint and, with it, the kernel size and the kernels per
    unit, so the arms are confounded and are reported as such. DRAM per target inference is the
    device's total, so it holds the GEMM's own misses as well as the target's."""
    rows, base = [], None
    for log in sorted(glob.glob(os.path.join(PROF, "footprint", "*.log"))):
        arm = os.path.basename(log)[:-4]
        st = os.path.join(PROF, "footprint", arm + "_stats.csv")
        if not os.path.exists(st):
            continue
        txt = open(log, errors="replace").read()
        side = re.search(r"GEMM side:\s+(\d+)", txt)
        mb = re.search(r"working set:\s+([0-9.]+) MB", txt)
        units = re.search(r"units:\s+\d+ in [0-9.]+ s \(([0-9.]+) units/s\)", txt)
        nets, co = report(log)
        med = stats_median(st)
        base = base or med
        # the device's DRAM bytes over the run per target inference (warm-up included, as in
        # dram_tables): the target's own traffic plus the GEMM's, which this run cannot separate
        rep = os.path.join(PROF, "footprint", arm + ".nsys-rep")
        mb_inf = ""
        if os.path.exists(rep) and nets:
            extract(rep, ("dram",))
            gb = dram_gb(rep[:-len(".nsys-rep")])[0]
            mb_inf = round(gb * 1e3 / sum(v[0] + WARMUP for v in nets.values()), 2)
        side = int(side.group(1)) if side else 0
        reps = max(1, int(4300000000.0 / side ** 3)) if side else 0
        rows.append(dict(arm=arm, side=side, footprint_mb=float(mb.group(1)) if mb else 0.0,
                         gemm_kernels_per_s=round(float(units.group(1)) * reps) if units else 0,
                         tflop_s=co.get("tflop_s", 0.0), target_median_ms=med,
                         vs_idle_pct=round((med / base - 1) * 100, 1),
                         dram_mb_per_target_inference=mb_inf))
    return pd.DataFrame(rows)


def main():
    os.makedirs(DERIVED, exist_ok=True)
    pd.set_option("display.width", 200)
    out = {"calibration": calibrate()}
    out["dram"], out["pair-traffic"] = dram_tables()
    out["trace-split"], out["trace-overlap"], out["launch-calls"], out["per-kernel"] = trace_tables()
    out["demand"] = demand_table(out["dram"])
    out["demand-fit"] = fits(out["demand"])
    out["footprint"] = footprint_table()
    out["l2-traffic"] = l2_traffic()
    for name, df in out.items():
        df.to_csv(os.path.join(DERIVED, name + ".csv"), index=False)
        if name != "per-kernel":
            print("\n== %s" % name)
            print(df.to_string(index=False))
    print("\n  written to %s" % DERIVED)


if __name__ == "__main__":
    sys.exit(main())
