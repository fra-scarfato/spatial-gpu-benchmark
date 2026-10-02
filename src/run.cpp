//
// run.cpp — a Run: the networks, the protocol and the results.
//

#include "run.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <sched.h>
#include <sys/resource.h>
#include <cerrno>
#include <cstring>

namespace {

// steady_clock, not system_clock: it is monotonic, so an NTP step during a
// 30-second measurement window cannot make an elapsed time negative or absurd.
// On Linux this maps to CLOCK_MONOTONIC.
using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// With one network the output keeps the plain name; with several, each network
// gets its own suffix so the files never collide. This is the whole reason the
// label exists in the file names.
std::string csvPrefixFor(const Options& opts, std::size_t index) {
    if (opts.networks.size() == 1)
        return opts.csvPrefix;
    return opts.csvPrefix + "_" + opts.networks[index].label;
}

} // namespace


Run::Run(const Options& opts)
    : opts_(opts)
{
    // 1. The dataset. Listed ONCE and shared by every network, so both
    //    networks of a concurrent run see exactly the same inputs. listImages
    //    sorts, so the order is deterministic and two runs are comparable.
    images_ = inputs::listImages(opts_.imageDir);
    if (opts_.maxImages > 0 && images_.size() > opts_.maxImages)
        images_.resize(opts_.maxImages);
    if (images_.empty())
        throw std::runtime_error("no .JPEG files in " + opts_.imageDir);

    // 2. Every SM allocation for the run, in ONE call. This has to be the
    //    first CUDA-touching step, and for green contexts it splits the device
    //    jointly, which is what makes the partitions disjoint.
    std::vector<int> percents;
    percents.reserve(opts_.networks.size());
    for (const auto& n : opts_.networks)
        percents.push_back(n.smPercent);

    // The work queue half of the green context descriptor, translated here so
    // that gpu.cpp never has to know about Options. Unset means the descriptor
    // carries the SM group only, which is what every run before the work queue
    // experiment did — and the reason those results stay comparable.
    //
    // A limit without a scope still needs a descriptor to carry it, and the
    // driver's default sharing behaviour is device_ctx.
    WqConfig wq;
    if (!opts_.wqScope.empty() || opts_.wqLimit != 0) {
        wq.set   = true;
        wq.scope = opts_.wqScope == "green_ctx_balanced"
                       ? cudaDevWorkqueueConfigScopeGreenCtxBalanced
                       : cudaDevWorkqueueConfigScopeDeviceCtx;
        wq.limit = opts_.wqLimit;
    }

    // The interferer, if any, is a participant in the same joint split: appending
    // its percentage makes its partition disjoint from the networks', exactly as
    // theirs are from each other.
    if (!opts_.interferer.kind.empty())
        percents.push_back(opts_.interferer.smPercent);

    std::vector<std::unique_ptr<SmAllocation>> allocations =
        makeSmAllocations(opts_.allocation, percents, wq);

    // 3. One Network per engine. The allocations are already provisioned by
    //    now, so building a network only creates an engine, buffers and a
    //    stream — it never has to provision anything itself.
    for (std::size_t k = 0; k < opts_.networks.size(); ++k) {
        const RunSpec spec{opts_.protocol.minInferences,
                           opts_.protocol.minDurationS,
                           csvPrefixFor(opts_, k)};
        nets_.push_back(std::make_unique<Network>(opts_.networks[k],
                                                  std::move(allocations[k]),
                                                  opts_, spec, images_));
    }

    if (!opts_.interferer.kind.empty()) {
        Interferer::Kind k = Interferer::Kind::MemSaxpy;
        if (opts_.interferer.kind == "compute_gemm")  k = Interferer::Kind::ComputeGemm;
        if (opts_.interferer.kind == "idle")          k = Interferer::Kind::Idle;
        if (opts_.interferer.kind == "spin")          k = Interferer::Kind::Spin;
        const std::size_t megabytes = 256;   // far above L2: the traffic leaves the chip
        interferer_ = std::make_unique<Interferer>(k, std::move(allocations[nets_.size()]),
                                                   megabytes, opts_.interferer.size);
    }

    // 4. Say what is about to be measured. This log is the record of the
    //    configuration: the granted SM counts appear here, and the results
    //    directory does not carry them anywhere else.
    std::cout << "[info] allocation:  " << opts_.allocation << "\n";
    for (const auto& n : nets_)
        std::cout << "[info] network '" << n->label() << "': " << n->allocationName()
                  << ", " << n->inputCount() << " inputs, " << cfg::channels << "x"
                    << n->inputHeight() << "x" << n->inputWidth() << " in\n";
    applyCpuPolicy(opts_);

    // The CPU environment this process actually got. The submission-model arms vary
    // affinity and priority from OUTSIDE (taskset, nice, chrt), and an externally
    // imposed setting that silently failed to take would be indistinguishable from
    // a result. Recorded here so every log answers the question by itself.
    {
        cpu_set_t set;
        std::string cores = "?";
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            cores.clear();
            for (int c = 0; c < CPU_SETSIZE; ++c)
                if (CPU_ISSET(c, &set)) {
                    if (!cores.empty()) cores += ",";
                    cores += std::to_string(c);
                }
        }
        const int pol = sched_getscheduler(0);
        const char* polname = pol == SCHED_FIFO  ? "SCHED_FIFO"
                            : pol == SCHED_RR    ? "SCHED_RR"
                            : pol == SCHED_OTHER ? "SCHED_OTHER" : "?";
        std::cout << "[info] cpu:         cores " << cores
                  << "  nice " << getpriority(PRIO_PROCESS, 0)
                  << "  " << polname;
        if (opts_.cpu >= 0 || !opts_.schedPolicy.empty()) {
            std::cout << "  (asked for";
            if (opts_.cpu >= 0) std::cout << " cpu " << opts_.cpu;
            if (!opts_.schedPolicy.empty())
                std::cout << " " << opts_.schedPolicy << ":" << opts_.schedPriority;
            std::cout << ")";
        }
        std::cout << "\n";
    }
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
        // Characterisation: there is nothing to align with, so the gate is a
        // one-participant no-op.
        nets_[0]->warmup();
        StartGate solo(1);
        if (interferer_) interferer_->start();   // a participant either way
        nets_[0]->measure(solo);
        if (interferer_) interferer_->stop();
        return;
    }

    // Single-submitter variant: the timed phase is driven by ONE thread, so
    // the two networks never enter the launch path at the same time. The
    // warm-up still runs in parallel, so both reach the same contended
    // steady state as in the threaded runs and this differs from them in
    // exactly one thing.
    if (opts_.submitThreads == 1) {
        std::cout << "[info] one submitting thread for " << nets_.size()
                  << " networks, each kept in flight\n";
        std::vector<Network*> ptrs;
        ptrs.reserve(nets_.size());
        for (auto& n : nets_)
            ptrs.push_back(n.get());

        std::vector<std::thread> warm;
        for (auto* p : ptrs)
            warm.emplace_back([p] { p->warmup(); });
        for (auto& t : warm)
            t.join();

        if (interferer_) interferer_->start();
        Network::measureInterleaved(ptrs);
        if (interferer_) interferer_->stop();
        return;
    }

    // Concurrent: each network gets its own thread, and BOTH the warm-up and
    // the timed phase happen in parallel. Warming up in parallel matters — it
    // means both networks reach a contended steady state before the gate
    // releases them, rather than one of them warming up alone on a quiet GPU.
    std::cout << "[info] running " << nets_.size()
              << " networks concurrently, each on its own partition\n";

    StartGate gate(nets_.size());
    std::vector<std::thread> threads;
    threads.reserve(nets_.size());
    for (auto& n : nets_) {
        // The two captures are deliberate: a raw pointer for the network,
        // because a Network is not copyable and so cannot be captured by
        // value, and the gate by reference so both threads share one gate.
        threads.emplace_back([p = n.get(), &gate] {
            p->warmup();
            p->measure(gate);
        });
    }
    for (auto& t : threads) t.join();
}

void Run::report() const {
    if (interferer_) interferer_->report();
    for (const auto& n : nets_) {
        std::cout << "\n--- network '" << n->label() << "' ---\n";
        n->report();
        n->writeStats();
    }

    if (nets_.size() > 1) {
        // Aggregate throughput: the total work over the window the networks
        // actually shared. max(duration) is that window, because the gate
        // released them together and each ran for its own measured duration.
        // Dividing by the sum, or by the shorter duration, would invent
        // throughput that never happened.
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

