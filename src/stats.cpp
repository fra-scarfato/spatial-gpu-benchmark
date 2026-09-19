// =====================================================================
// stats.cpp — the statistics implementation.
// =====================================================================

#include "stats.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// Percentiles interpolate linearly between sorted samples: the same
// convention numpy uses, so these numbers match a Python analysis of the
// CSV. Keep it that way — a mismatch here would make the runner and the
// analysis disagree by a few microseconds and nobody would notice why.
// ---------------------------------------------------------------------
Stats computeStats(std::vector<float> v) {
    Stats s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.n      = v.size();
    s.min    = v.front();
    s.max    = v.back();
    s.median = v[s.n / 2];

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
    s.std  = std::sqrt(var / (s.n - 1));   // sample std (n-1)
    s.tail = s.p99 - s.median;
    return s;
}

void printStatsHeader() {
    std::cout << std::left << std::setw(24) << "metric"
              << std::right
              << std::setw(7)  << "n"
              << std::setw(12) << "mean"
              << std::setw(12) << "median"
              << std::setw(12) << "std"
              << std::setw(12) << "p95"
              << std::setw(12) << "p99"
              << std::setw(12) << "max"
              << std::setw(12) << "min"
              << std::setw(12) << "p99-med"
              << "\n";
    std::cout << std::string(24 + 7 + 12 * 8, '-') << "\n";
}

void printStats(const char* label, const Stats& s) {
    std::cout << std::left << std::setw(24) << label
              << std::right
              << std::setw(7) << s.n
              << std::fixed << std::setprecision(4)
              << std::setw(12) << s.mean
              << std::setw(12) << s.median
              << std::setw(12) << s.std
              << std::setw(12) << s.p95
              << std::setw(12) << s.p99
              << std::setw(12) << s.max
              << std::setw(12) << s.min
              << std::setw(12) << s.tail
              << "\n";
    std::cout << std::setprecision(6) << std::defaultfloat;
}
