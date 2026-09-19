// =====================================================================
// run.cpp — a Run: the networks, the protocol and the results.
// =====================================================================

#include "run.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// With one network the output keeps the plain name; with several, each network
// gets its own suffix so the files never collide.
std::string csvPrefixFor(const Options& opts, std::size_t index) {
    if (opts.networks.size() == 1)
        return opts.csvPrefix;
    return opts.csvPrefix + "_" + opts.networks[index].label;
}

} // namespace

// =====================================================================
// Network
// =====================================================================
Network::Network(const NetworkOptions& net,
                 std::unique_ptr<SmAllocation> allocation,
                 const Options& opts,
                 const RunSpec& spec,
                 const std::vector<std::string>& images)
    : label_(net.label),
      allocation_(std::move(allocation)),
      engine_(net.enginePath, *allocation_),
      inputSet_(opts, images, std::size_t(engine_.inputVolume())),
      spec_(spec),
      csv_(spec.csvPrefix + "_latencies.csv")
{
    if (!csv_)
        throw std::runtime_error("cannot open " + spec.csvPrefix + "_latencies.csv");
    csv_ << "phase,index,image,prep_ms,wall_ms,gpu_ms\n";
}

Network::Sample Network::step(std::size_t index) {
    // Fetch the input. In fixed mode this is pointer arithmetic — no disk, no
    // decode, no preprocessing. In end-to-end mode it pays the full CPU cost,
    // which is what prep_ms then measures.
    const auto t0 = Clock::now();
    const float* input = inputSet_.at(index);
    const auto t1 = Clock::now();

    // H2D + kernel + D2H + sync; returns the pure GPU compute time.
    const double gpuMs = engine_.infer(input);
    const auto t2 = Clock::now();

    Sample s;
    s.prepMs = float(std::chrono::duration<double, std::milli>(t1 - t0).count());
    s.wallMs = float(std::chrono::duration<double, std::milli>(t2 - t0).count());
    s.gpuMs  = float(gpuMs);
    return s;
}

void Network::writeRow(const char* phase, std::size_t index, const Sample& s) {
    csv_ << phase << "," << index << "," << inputSet_.idOf(index) << ","
         << s.prepMs << "," << s.wallMs << "," << s.gpuMs << "\n";
}

void Network::warmup() {
    // Warm-up rows are recorded (so the warm-up curve is visible in the CSV)
    // but excluded from the statistics: warm-up is not the steady state, and
    // the steady state is what we characterise.
    std::cout << "[info] warm-up: " << cfg::numWarmup << " inferences over "
              << inputSet_.count() << " distinct inputs\n";

    for (std::size_t i = 0; i < cfg::numWarmup; ++i)
        writeRow("warmup", i, step(i));
}

void Network::measure(StartGate& gate) {
    // The stop rule. A bound of 0 is disabled. With both bounds at 0 the loop
    // makes a single pass over the input set — fast, for development.
    // Otherwise it keeps cycling the set until BOTH bounds are satisfied, so
    // a fast network is measured over the same window as a slow one.
    const bool  enforce   = (spec_.minInferences > 0) || (spec_.minDurationS > 0.0);
    const std::size_t onePass = inputSet_.count();

    if (enforce) {
        std::cout << "[info] timed phase: until at least ";
        if (spec_.minInferences > 0 && spec_.minDurationS > 0.0) {
            std::cout << spec_.minInferences << " inferences AND at least "
                      << spec_.minDurationS << " s";
        } else if (spec_.minInferences > 0) {
            std::cout << spec_.minInferences << " inferences";
        } else {
            std::cout << spec_.minDurationS << " s";
        }
        std::cout << "\n";
    } else {
        std::cout << "[info] timed phase: single pass of " << onePass
                  << " inferences (stop rule disabled)\n";
    }

    // Every network waits here, so the timed phases start together. With one
    // participant this returns immediately.
    gate.arriveAndWait();

    const auto t0 = Clock::now();
    std::size_t n = 0;
    for (;;) {
        const Sample s = step(n);
        writeRow("bench", n, s);
        gpuTimes_.push_back(s.gpuMs);
        wallTimes_.push_back(s.wallMs);
        prepTimes_.push_back(s.prepMs);
        ++n;

        if (!enforce) {
            if (n >= onePass) break;
        } else {
            const double elapsed = secondsSince(t0);
            if (n >= spec_.minInferences && elapsed >= spec_.minDurationS) break;
        }

        if (n % 2000 == 0)
            std::cout << "  ..." << n << " inferences, "
                      << std::fixed << std::setprecision(1)
                      << secondsSince(t0) << " s\n"
                      << std::defaultfloat;
    }

    timedCount_     = n;
    timedDurationS_ = secondsSince(t0);
    csv_.close();
}

void Network::report() const {
    std::cout << "\n=== LATENCY SUMMARY (timed phase) ===\n";
    std::cout << "inferences: " << timedCount_
              << "   duration: " << std::fixed << std::setprecision(2)
              << timedDurationS_ << " s\n";
    // Say it here too: this is where the numbers are read, and prep_ms ~0 is
    // easy to misread as "preprocessing is instantaneous".
    if (inputSet_.isFixed())
        std::cout << "note: fixed-input mode. Preprocessing ran once at startup and is\n"
                     "      NOT part of these numbers: prep_ms is ~0 by construction and\n"
                     "      wall_ms covers exactly H2D + inference + D2H.\n";
    std::cout << "\n" << std::defaultfloat;
    printStatsHeader();
    printStats("gpu compute (ms)",     computeStats(gpuTimes_));
    printStats("end-to-end wall (ms)", computeStats(wallTimes_));
    printStats("preprocess only (ms)", computeStats(prepTimes_));

    std::cout << "\nThroughput (whole loop): "
              << double(timedCount_) / timedDurationS_ << " inferences/s\n";

    const Stats gpu = computeStats(gpuTimes_);
    std::cout << "Throughput (GPU only):   " << 1000.0 / gpu.mean
              << " inferences/s (1/mean)\n";

    std::cout << "\nCSV written to: " << spec_.csvPrefix << "_latencies.csv\n";
}

void Network::writeStats() const {
    // One row per metric; the run metadata is repeated on every row so a
    // single row is self-contained.
    const std::string path = spec_.csvPrefix + "_stats.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path);

    out << "metric,n,mean,median,std,p95,p99,max,min,p99_minus_median,"
        << "count,duration_s\n";

    const std::pair<const char*, const std::vector<float>*> rows[] = {
        {"gpu_ms",  &gpuTimes_},
        {"wall_ms", &wallTimes_},
        {"prep_ms", &prepTimes_},
    };
    out << std::fixed << std::setprecision(6);
    for (const auto& [name, values] : rows) {
        const Stats s = computeStats(*values);
        out << name << "," << s.n << "," << s.mean << "," << s.median << ","
            << s.std << "," << s.p95 << "," << s.p99 << "," << s.max << ","
            << s.min << "," << s.tail << "," << timedCount_ << ","
            << timedDurationS_ << "\n";
    }

    std::cout << "Stats written to: " << path << "\n";
}

// =====================================================================
// Run
// =====================================================================
Run::Run(const Options& opts)
    : opts_(opts)
{
    // 1. The dataset, sorted so runs are deterministic and comparable. It is
    //    listed ONCE and shared by every network, so both networks of a
    //    concurrent run see exactly the same inputs.
    images_ = inputs::listImages(opts_.imageDir);
    if (opts_.maxImages > 0 && images_.size() > opts_.maxImages)
        images_.resize(opts_.maxImages);
    if (images_.empty())
        throw std::runtime_error("no .JPEG files in " + opts_.imageDir);

    // 2. Every SM allocation for the run, in ONE call: this must be the first
    //    CUDA-touching step, and for green contexts it splits the device
    //    jointly so the partitions are disjoint.
    std::vector<int> percents;
    percents.reserve(opts_.networks.size());
    for (const auto& n : opts_.networks)
        percents.push_back(n.smPercent);

    std::vector<std::unique_ptr<SmAllocation>> allocations =
        makeSmAllocations(opts_.allocation, percents);

    // 3. One Network per engine. The allocations are already provisioned, so
    //    building the networks only creates engines, buffers and streams.
    for (std::size_t k = 0; k < opts_.networks.size(); ++k) {
        const RunSpec spec{opts_.protocol.minInferences,
                           opts_.protocol.minDurationS,
                           csvPrefixFor(opts_, k)};
        nets_.push_back(std::make_unique<Network>(opts_.networks[k],
                                                  std::move(allocations[k]),
                                                  opts_, spec, images_));
    }

    // 4. Say what we are about to measure.
    std::cout << "[info] allocation:  " << opts_.allocation << "\n";
    for (const auto& n : nets_)
        std::cout << "[info] network '" << n->label() << "': " << n->allocationName()
                  << ", " << n->inputCount() << " inputs\n";
    std::cout << "[info] dataset:     " << images_.size() << " images available\n"
              << "[info] input mode:  "
              << (nets_[0]->fixedInputs() ? "fixed" : "dataset") << "\n";
    if (nets_.size() > 1 && opts_.allocation == "whole")
        std::cout << "[info] no partitioning: the networks contend for the whole "
                     "GPU (baseline)\n";
    if (nets_[0]->fixedInputs())
        std::cout << "[info] preprocessing: NOT in the timed loop "
                     "(done once, at startup)\n";
}

void Run::execute() {
    if (nets_.size() == 1) {
        // Characterisation: nothing to align with, so the gate is a no-op.
        nets_[0]->warmup();
        StartGate solo(1);
        nets_[0]->measure(solo);
        return;
    }

    // Concurrent: warm-up and the timed phase both happen in parallel, so both
    // networks reach a contended steady state before the gate releases them.
    std::cout << "[info] running " << nets_.size()
              << " networks concurrently, each on its own partition\n";

    StartGate gate(nets_.size());
    std::vector<std::thread> threads;
    threads.reserve(nets_.size());
    for (auto& n : nets_) {
        threads.emplace_back([p = n.get(), &gate] {
            p->warmup();
            p->measure(gate);
        });
    }
    for (auto& t : threads) t.join();
}

void Run::report() const {
    for (const auto& n : nets_) {
        std::cout << "\n--- network '" << n->label() << "' ---\n";
        n->report();
        n->writeStats();
    }

    if (nets_.size() > 1) {
        // Aggregate throughput across the networks: total work over the window
        // they actually shared. max(duration) is the window, because the gate
        // released them together.
        std::size_t total   = 0;
        double      longest = 0.0;
        for (const auto& n : nets_) {
            total  += n->timedCount();
            longest = std::max(longest, n->timedDurationS());
        }
        std::cout << "\n=== AGGREGATE (" << nets_.size() << " networks) ===\n"
                  << "inferences: " << total << " in " << longest << " s\n"
                  << "aggregate throughput: " << double(total) / longest
                  << " inferences/s\n";
    }
}
