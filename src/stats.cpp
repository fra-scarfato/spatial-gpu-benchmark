//
// stats.cpp — the statistics implementation.
//

#include "stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// Percentiles interpolate linearly between the two neighbouring sorted samples,
// which is the convention numpy uses. Keep it that way: a mismatch here would
// make the runner and a Python analysis of the same CSV disagree by a few
// microseconds, and nothing would say why.
Stats computeStats(std::vector<float> v) {
    // Taken by value because it is sorted in place. Sorting a copy of a few
    // thousand floats costs microseconds and saves every caller from having to
    // keep the original order.
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n      = v.size();
    s.min    = v.front();
    s.max    = v.back();
    // For an even sample count this is the UPPER of the two middle values, not
    // their average. Deliberate: the median is always an actually observed
    // latency, which is the more honest number for a latency table.
    s.median = v[s.n / 2];

    // Linear interpolation. idx is a fractional position in the sorted array;
    // the result is a weighted blend of the two samples either side of it.
    auto pct = [&](double p) {
        const double idx = p * (s.n - 1);
        const std::size_t lo = std::size_t(idx);
        const std::size_t hi = std::min(lo + 1, s.n - 1);
        const double frac = idx - lo;
        return v[lo] * (1.0 - frac) + v[hi] * frac;
    };
    s.p95  = pct(0.95);
    s.p99  = pct(0.99);
    s.mean = std::accumulate(v.begin(), v.end(), 0.0) / s.n;

    double var = 0;
    for (float x : v) var += (x - s.mean) * (x - s.mean);
    // n-1, not n: this is the SAMPLE standard deviation, because these
    // observations are a sample of the runs we could have done, not the whole
    // population. With thousands of samples the difference is negligible, but
    // it is the correct denominator and it is what numpy uses by default.
    s.std  = std::sqrt(var / (s.n - 1));
    // The headline number for a real-time system: how far the worst 1% sits
    // above the typical case.
    s.tail = s.p99 - s.median;
    return s;
}

// The two printing functions below build a fixed-width table. The widths are
// repeated rather than shared, because the header and the rows have to agree on
// every column width and keeping them adjacent is what makes that visible.
//
// These use std::format rather than the iostream manipulators, because
// setw/setprecision/fixed are STICKY: with the old code every caller had to
// remember to reset the stream afterwards, and forgetting left the next block of
// output full of stray decimals. A format string carries no state to leak, and
// the widths are visible in one line instead of spread over twenty << operators.
void printStatsHeader() {
    // "obs-max" and not "max": the value is the largest sample THIS RUN
    // actually measured, and the label has to carry the word observed so it is
    // never read as a worst-case bound. The CSV column keeps the short name
    // `max`, because every results file already on disk is written that way and
    // the analysis reads that name; only the human-facing label changed.
    std::cout << std::format(
        "{:<24}{:>7}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}{:>12}\n",
        "metric", "n", "mean", "median", "std", "p95", "p99", "obs-max", "min", "p99-med");
    // 24 + 7 + eight 12-wide columns, so the rule matches the table exactly.
    std::cout << std::string(24 + 7 + 12 * 8, '-') << "\n";
}

void printStats(const char* label, const Stats& s) {
    std::cout << std::format(
        "{:<24}{:>7}{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}"
        "{:>12.4f}{:>12.4f}{:>12.4f}{:>12.4f}\n",
        label, s.n, s.mean, s.median, s.std, s.p95, s.p99, s.max, s.min, s.tail);
}
