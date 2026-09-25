#!/usr/bin/env python3
"""streams.py - per-stream kernel activity and cross-stream overlap from an nsys report.

The question this answers is not "how many streams exist" but "do they actually
overlap". Two streams that alternate are not concurrent work; two streams whose
kernels are simultaneously in flight are. A trace makes that visible by eye, but
a thesis needs a number, and this is where it comes from.

Usage:
    nsys export --type sqlite --output report.sqlite report.nsys-rep
    python3 streams.py report.sqlite [more.sqlite ...]

Exports must be SQLite: the .nsys-rep is a proprietary container, and the SQLite
form is the one that can be queried. nsys export runs anywhere nsys is installed,
including on the machine the report was collected on.

What it prints, per report:
  - one line per CUDA stream that issued kernels: kernel count, total kernel
    time ("busy"), and the span between its first and last kernel;
  - the overlap of the two busiest streams, defined as the time during which
    BOTH have at least one kernel in flight, as a percentage of the time they
    jointly occupy. Same-stream kernels cannot overlap by construction, so this
    percentage measures genuine cross-stream concurrency.

A high busy time with a near-zero overlap is serialisation, whatever the stream
count says. That is the signature worth looking for, and it is the reason a
stream count on its own proves nothing.
"""

import sqlite3
import sys

# The CUPTI table has been stable under this name, but nsys versions differ; the
# fallback below finds it if this one is absent.
KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"


def kernel_rows(db):
    """(streamId, start, end) for every kernel, in ns."""
    try:
        return list(db.execute(
            f"SELECT streamId, start, end FROM {KERNEL_TABLE}"))
    except sqlite3.OperationalError:
        tables = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        candidates = [t for t in tables if "KERNEL" in t.upper()]
        if not candidates:
            raise SystemExit(f"no kernel table in this export; tables are: {tables}")
        print(f"  note: {KERNEL_TABLE} absent, using {candidates[0]}")
        return list(db.execute(
            f"SELECT streamId, start, end FROM {candidates[0]}"))


def merged(intervals):
    """Merge [start,end) pairs into disjoint spans, so overlap is not double-counted."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    """Total time with at least one interval from each list in flight."""
    events = []
    for iv in (a, b):
        for s, e in iv:
            events.append((s, 1))
            events.append((e, -1))
    events.sort()
    total = 0
    live = 0
    prev = None
    for t, delta in events:
        if prev is not None and live >= 2:
            total += t - prev
        live += delta
        prev = t
    return total


def report(path):
    print(f"=== {path}")
    db = sqlite3.connect(path)
    rows = kernel_rows(db)
    if not rows:
        print("  no kernels recorded")
        db.close()
        return

    per = {}
    for sid, s, e in rows:
        per.setdefault(sid, []).append((s, e))

    print(f"  {len(rows)} kernels on {len(per)} stream(s)")
    busiest = sorted(per, key=lambda k: -sum(e - s for s, e in per[k]))
    for sid in busiest:
        iv = per[sid]
        busy = sum(e - s for s, e in iv)
        span = max(e for _, e in iv) - min(s for s, _ in iv)
        print(f"    stream {sid:>5}: {len(iv):>6} kernels"
              f"   busy {busy/1e6:9.2f} ms"
              f"   span {span/1e6:9.2f} ms")

    if len(busiest) >= 2:
        a, b = busiest[0], busiest[1]
        joint = merged(per[a] + per[b])
        joint_ns = sum(e - s for s, e in joint)
        both = overlap(merged(per[a]), merged(per[b]))
        print(f"    overlap of streams {a} and {b}: {both/1e6:.2f} ms"
              f"  = {100.0*both/joint_ns:.1f}% of the {joint_ns/1e6:.2f} ms"
              f" they jointly occupy")
        if len(busiest) > 2:
            print(f"    note: {len(busiest)-2} further stream(s) present;"
                  f" only the two busiest are compared")
    db.close()


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        report(path)
        print()


if __name__ == "__main__":
    main()
