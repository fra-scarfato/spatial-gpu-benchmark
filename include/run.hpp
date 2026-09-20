#pragma once
//
// run.hpp — one process invocation: the networks, the protocol, the results.
//
// A Run is one process with fresh CUDA/GPU state, one dataset, one protocol,
// and ONE OR MORE networks:
//
//   one network  -> the characterisation runs
//   two networks -> the concurrent runs, one green context each
//
// A Network is the subject AND its measurement: it owns the SM allocation, the
// TensorRT engine and the inputs, and it knows how to warm itself up, measure
// itself and write its results.
//

#include <atomic>
#include <cstddef>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "gpu.hpp"
#include "inputs.hpp"
#include "stats.hpp"

// What one network needs to know about the protocol, plus where its results go.
// Deliberately not the whole Options: a network has no business knowing about
// other networks, datasets or allocation kinds.
struct RunSpec {
    std::size_t minInferences = cfg::minInferences;
    double      minDurationS  = cfg::minDurationS;
    std::string csvPrefix     = std::string(cfg::defaultCsvPrefix);
};

// Everyone spins until all have arrived, so several networks begin their timed
// phase in the same instant. The gate is what makes the two timed windows
// identical, and identical windows are what make their throughputs comparable.
// With one participant it is a no-op, which is how the solo path reuses the
// concurrent code.
class StartGate {
public:
    explicit StartGate(std::size_t participants) : total_(participants) {}

    void arriveAndWait() {
        // The last arrival releases everyone else. The memory orders are doing
        // real work: the release store publishes everything each thread wrote
        // before arriving, and the acquire load makes it visible afterwards, so
        // nobody starts the timed phase against a half-initialised world.
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == total_)
            open_.store(true, std::memory_order_release);
        // Spinning, not blocking. The wait is the start-up skew between
        // networks — microseconds — so a condition variable would cost more
        // than it saves.
        while (!open_.load(std::memory_order_acquire)) {
        }
    }

private:
    std::size_t              total_;
    std::atomic<std::size_t> arrived_{0};
    std::atomic<bool>        open_{false};
};

// One model, its resources, and its measurement.
//
// Member order is load-bearing, and it is declaration order that decides it,
// not the order in the initialiser list:
//   allocation_ -> engine_    the stream comes from the allocation and must not
//                             outlive it, so the allocation is declared first
//                             and therefore destroyed last;
//   engine_     -> inputSet_  the input set needs engine_.inputVolume();
//   both        -> the measurement state, which refers back to this object.
// Non-movable for the same reason: relocating the object would invalidate the
// engine's stream association. Hold it by unique_ptr.
class Network {
public:
    Network(const NetworkOptions& net,
            std::unique_ptr<SmAllocation> allocation,
            const Options& opts,
            const RunSpec& spec,
            const std::vector<std::string>& images);

    Network(const Network&)            = delete;
    Network& operator=(const Network&) = delete;

    const std::string& label()          const noexcept { return label_; }
    std::string        allocationName() const { return allocation_->name(); }
    std::size_t        inputCount()     const noexcept { return inputSet_.count(); }
    bool               fixedInputs()    const noexcept { return inputSet_.isFixed(); }

    void warmup();                    // cfg::numWarmup inferences, not measured
    void measure(StartGate& gate);    // the timed phase, aligned with the others
    void report() const;              // statistics table + throughput
    void writeStats() const;          // <prefix>_stats.csv

    std::size_t timedCount()     const noexcept { return timedCount_; }
    double      timedDurationS() const noexcept { return timedDurationS_; }

private:
    // One inference: fetch the input, run it, measure the three timings.
    struct Sample { float prepMs = 0, wallMs = 0, gpuMs = 0; };
    Sample step(std::size_t index);
    void   writeRow(const char* phase, std::size_t index, const Sample& s);

    std::string                   label_;
    std::unique_ptr<SmAllocation> allocation_;   // declared first -> destroyed last
    TrtEngine                     engine_;
    inputs::InputSet              inputSet_;
    RunSpec                       spec_;
    std::ofstream                 csv_;

    std::vector<float>            gpuTimes_, wallTimes_, prepTimes_;
    std::size_t                   timedCount_     = 0;
    double                        timedDurationS_ = 0.0;
};

// One process invocation.
//
// It assembles everything — the images, all the allocations in ONE joint split,
// one Network per engine — and then orchestrates them: sequentially for one
// network, in aligned threads for several. Finally it reports.
class Run {
public:
    explicit Run(const Options& opts);

    void execute();       // warm-up + timed phase
    void report() const;  // per-network results, then the aggregate

private:
    Options                               opts_;
    std::vector<std::string>              images_;
    std::vector<std::unique_ptr<Network>> nets_;
};
