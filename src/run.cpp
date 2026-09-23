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

Network::Network(const NetworkOptions& net,
                 std::unique_ptr<SmAllocation> allocation,
                 const Options& opts,
                 const RunSpec& spec,
                 const std::vector<std::string>& images)
    : label_(net.label),
      allocation_(std::move(allocation)),
      // engine_ is built before inputSet_ because declaration order decides,
      // and inputSet_ needs engine_.inputVolume() to size itself.
      engine_(net.enginePath, *allocation_),
      inputSet_(opts, images, std::size_t(engine_.inputVolume()), engine_.inputSide()),
      spec_(spec),
      csv_(spec.csvPrefix + "_latencies.csv")
{
    // ofstream does not throw by default, so an unopened file looks like a
    // working stream that silently discards every row.
    if (!csv_)
        throw std::runtime_error("cannot open " + spec.csvPrefix + "_latencies.csv");
    csv_ << "phase,index,image,prep_ms,wall_ms,gpu_ms\n";
}

Network::Sample Network::step(std::size_t index) {
    // Fetch the input first, on its own, so that its cost can be attributed.
    // In fixed mode this is pointer arithmetic — no disk, no decode, no
    // preprocessing — so prep_ms is ~0 by construction. In end-to-end mode it
    // pays the full CPU cost, which is exactly what prep_ms measures.
    const auto t0 = Clock::now();
    const float* input = inputSet_.at(index);
    const auto t1 = Clock::now();

    // H2D + kernel + D2H + sync; returns the pure GPU compute time.
    const double gpuMs  = engine_.infer(input);
    const double wallMs = engine_.wallMs();   // the same step, taken on the stream

    // prep_ms is the host cost of fetching THIS network's input, and is a
    // disjoint slice of the step rather than extra time. wall_ms is deliberately
    // NOT a host span: it comes from events on this network's stream over
    // H2D + kernel + D2H, so it stays this network's number even when one thread
    // drives several networks at once.
    Sample s;
    s.prepMs = float(std::chrono::duration<double, std::milli>(t1 - t0).count());
    s.wallMs = float(wallMs);
    s.gpuMs  = float(gpuMs);
    return s;
}

void Network::writeRow(const char* phase, std::size_t index, const Sample& s) {
    // "phase" separates warmup from bench so the warm-up curve stays in the
    // file without contaminating the statistics. "image" is the input's id,
    // which in fixed mode is a set index rather than a file name.
    csv_ << phase << "," << index << "," << inputSet_.idOf(index) << ","
         << s.prepMs << "," << s.wallMs << "," << s.gpuMs << "\n";
}

void Network::warmup() {
    // Warm-up rows are written to the CSV so the ramp is visible, but they are
    // never pushed into the statistics vectors: warm-up is not the steady
    // state, and the steady state is what is being characterised. The first
    // inference costs ~37 ms against a ~0.54 ms steady state.
    std::cout << "[info] warm-up: " << cfg::numWarmup << " inferences over "
              << inputSet_.count() << " distinct inputs\n";

    for (std::size_t i = 0; i < cfg::numWarmup; ++i)
        writeRow("warmup", i, step(i));
}

void Network::measure(StartGate& gate) {
    // The stop rule. A bound of 0 disables it, and with both bounds at 0 the
    // loop makes a single pass over the input set, which is the fast
    // development mode.
    //
    // Both bounds are needed and they answer different questions: the count
    // gives the percentiles enough samples, and the duration gives every
    // configuration the same observation window, so a fast network is not
    // measured over a shorter period than a slow one.
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

    // Every network blocks here, so the timed phases begin together. With one
    // participant it returns immediately, which is why the solo path needs no
    // special case.
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

        // Checked on every iteration rather than only when the count is
        // reached: the duration bound can be the first one satisfied.
        if (!enforce) {
            if (n >= onePass) break;
        } else {
            const double elapsed = secondsSince(t0);
            if (n >= spec_.minInferences && elapsed >= spec_.minDurationS) break;
        }

        // Progress only; printing every inference would cost more than the
        // inference itself. std::format, not the sticky setprecision, so this
        // cannot leave the stream in a state the table below inherits.
        if (n % 2000 == 0)
            std::cout << std::format("  ...{} inferences, {:.1f} s\n",
                                     n, secondsSince(t0));
    }

    // The timed window is the loop's own elapsed time, not derived from the
    // per-inference timings, so it stays correct even if a sample is dropped.
    timedCount_     = n;
    timedDurationS_ = secondsSince(t0);
    csv_.close();
}

void Network::report() const {
    std::cout << "\n=== LATENCY SUMMARY (timed phase) ===\n";
    std::cout << std::format("inferences: {}   duration: {:.2f} s\n",
                             timedCount_, timedDurationS_);
    // Said here as well as at startup because this is where the numbers are
    // read, and prep_ms ~0 is easy to misread as "preprocessing is
    // instantaneous".
    if (inputSet_.isFixed())
        std::cout << "note: fixed-input mode. Preprocessing ran once at startup and is\n"
                     "      NOT part of these numbers: prep_ms is ~0 by construction and\n"
                     "      wall_ms covers exactly H2D + inference + D2H.\n";
    std::cout << "\n";
    printStatsHeader();
    printStats("gpu compute (ms)",     computeStats(gpuTimes_));
    printStats("end-to-end wall (ms)", computeStats(wallTimes_));
    printStats("preprocess only (ms)", computeStats(prepTimes_));

    // Two different throughputs, deliberately. The whole-loop figure is the
    // one a scheduler would see; the GPU-only figure is 1/mean and is what the
    // 30-second window would produce if nothing else existed.
    std::cout << "\nThroughput (whole loop): "
              << double(timedCount_) / timedDurationS_ << " inferences/s\n";

    const Stats gpu = computeStats(gpuTimes_);
    std::cout << "Throughput (GPU only):   " << 1000.0 / gpu.mean
              << " inferences/s (1/mean)\n";

    std::cout << "\nCSV written to: " << spec_.csvPrefix << "_latencies.csv\n";
}

void Network::writeStats() const {
    // One row per metric, and the run metadata (count, duration) is repeated
    // on every row so that a single row is self-contained — which is what lets
    // aggregate.py read these files without cross-referencing anything else.
    const std::string path = spec_.csvPrefix + "_stats.csv";
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot open " + path);

    out << "metric,n,mean,median,std,p95,p99,max,min,p99_minus_median,"
        << "count,duration_s\n";

    // An array of pairs rather than three near-identical blocks; the structured
    // binding below unpacks it.
    const std::pair<const char*, const std::vector<float>*> rows[] = {
        {"gpu_ms",  &gpuTimes_},
        {"wall_ms", &wallTimes_},
        {"prep_ms", &prepTimes_},
    };
    // The column count here has to match the header above exactly: this file is
    // the contract with aggregate.py, which parses it positionally.
    for (const auto& [name, values] : rows) {
        const Stats s = computeStats(*values);
        out << std::format("{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},"
                           "{:.6f},{:.6f},{},{:.6f}\n",
                           name, s.n, s.mean, s.median, s.std, s.p95, s.p99,
                           s.max, s.min, s.tail, timedCount_, timedDurationS_);
    }

    std::cout << "Stats written to: " << path << "\n";
}

// One thread drives every network of the run.
//
// Each network keeps its own stream, its own CUDA events and its own statistics:
// the number that matters is stamped on the device between that network's own
// start and stop events, so when the host reads it cannot move it. What changes
// is only that a single thread enters the launch path, and that it never blocks
// on one network while the other has work outstanding.
//
// The shape is: keep exactly one inference in flight per network — the same
// depth the threaded path has — harvest whichever network has drained, record
// its sample, submit its next inference. Each network stops on its own rule, so
// they can finish at different times.
void Network::measureInterleaved(const std::vector<Network*>& nets) {
    const std::size_t n = nets.size();

    std::vector<bool>              enforce(n);
    std::vector<std::size_t>       onePass(n);
    std::vector<std::size_t>       count(n, 0);
    std::vector<bool>              inFlight(n, false);
    std::vector<double>            doneAt(n, 0.0);
    std::vector<Clock::time_point> t0(n), t1(n);

    for (std::size_t i = 0; i < n; ++i) {
        enforce[i] = (nets[i]->spec_.minInferences > 0) ||
                     (nets[i]->spec_.minDurationS > 0.0);
        onePass[i] = nets[i]->inputSet_.count();
        std::cout << "[info] '" << nets[i]->label() << "' timed phase: ";
        if (enforce[i])
            std::cout << "at least " << nets[i]->spec_.minInferences
                      << " inferences AND at least " << nets[i]->spec_.minDurationS
                      << " s\n";
        else
            std::cout << "single pass of " << onePass[i] << " inferences\n";
    }

    const auto start = Clock::now();

    // Prime one inference per network, so the window begins with both
    // partitions busy rather than with one of them idle.
    for (std::size_t i = 0; i < n; ++i) {
        t0[i] = Clock::now();
        const float* input = nets[i]->inputSet_.at(0);
        t1[i] = Clock::now();
        nets[i]->engine_.submit(input);
        inFlight[i] = true;
    }

    std::size_t running = n;
    while (running > 0) {
        for (std::size_t i = 0; i < n; ++i) {
            if (!inFlight[i] || !nets[i]->engine_.ready())
                continue;

            Sample s;
            s.gpuMs  = float(nets[i]->engine_.finish());
            s.wallMs = float(nets[i]->engine_.wallMs());   // per network, not per thread
            s.prepMs = float(std::chrono::duration<double, std::milli>(t1[i] - t0[i]).count());

            nets[i]->writeRow("bench", count[i], s);
            nets[i]->gpuTimes_.push_back(s.gpuMs);
            nets[i]->wallTimes_.push_back(s.wallMs);
            nets[i]->prepTimes_.push_back(s.prepMs);
            ++count[i];

            const double elapsed = secondsSince(start);
            const bool finished =
                enforce[i]
                    ? (count[i] >= nets[i]->spec_.minInferences &&
                       elapsed >= nets[i]->spec_.minDurationS)
                    : (count[i] >= onePass[i]);

            if (finished) {
                doneAt[i]   = elapsed;
                inFlight[i] = false;
                --running;
            } else {
                t0[i] = Clock::now();
                const float* input = nets[i]->inputSet_.at(count[i]);
                t1[i] = Clock::now();
                nets[i]->engine_.submit(input);
            }
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        nets[i]->timedCount_     = count[i];
        nets[i]->timedDurationS_ = doneAt[i];
        nets[i]->csv_.close();
        std::cout << "[info] '" << nets[i]->label() << "': " << count[i]
                  << " inferences in " << doneAt[i] << " s\n";
    }
}

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

    // 4. Say what is about to be measured. This log is the record of the
    //    configuration: the granted SM counts appear here, and the results
    //    directory does not carry them anywhere else.
    std::cout << "[info] allocation:  " << opts_.allocation << "\n";
    for (const auto& n : nets_)
        std::cout << "[info] network '" << n->label() << "': " << n->allocationName()
                  << ", " << n->inputCount() << " inputs, " << cfg::channels << "x"
                    << n->inputSide() << "x" << n->inputSide() << " in\n";
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
        nets_[0]->measure(solo);
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

        Network::measureInterleaved(ptrs);
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
