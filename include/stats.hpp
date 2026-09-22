#pragma once
//
// stats.hpp — the statistics, and nothing else.
//
// Pure arithmetic on a vector of measured latencies: no CUDA, no images, no
// configuration. That matters twice over — it can be read and tested without
// the GPU stack, and the reporting convention lives in exactly one place, so
// the numbers printed here and the numbers a Python analysis computes from the
// same CSV agree.
//

#include <cstddef>
#include <vector>

// The battery used for latency reporting:
//
//   mean    average; sensitive to outliers, so never report it alone.
//   median  the middle observation; robust.
//   std     spread around the mean (sample standard deviation, n-1).
//   p95/99  the value below which 95%/99% of observations fall.
//   max     the OBSERVED MAXIMUM of this run: the largest sample it actually
//           saw, not a bound and not a typical value. Its whole purpose is to
//           survive a rare event instead of smoothing it away, so when several
//           runs are combined it has to be reduced with max over the runs --
//           max of maxima is the max of the union, i.e. the maximum over every
//           inference measured. Averaging it, or taking its median, discards
//           the one observation it exists to expose. See obs_max() in
//           aggregate.py, which is the only place allowed to combine it.
//   tail    p99 - median: how fat the worst 1% is. For a real-time system this
//           is the number that decides deadlines.
struct Stats {
    std::size_t n = 0;
    double mean = 0, median = 0, std = 0;
    double p95 = 0, p99 = 0, max = 0, min = 0;
    double tail = 0;
};

// Takes the vector BY VALUE because it sorts it.
Stats computeStats(std::vector<float> values);

// Fixed-width console table, so the columns line up. Values are milliseconds.
void printStatsHeader();
void printStats(const char* label, const Stats& s);
