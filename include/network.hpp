#pragma once
//
// network.hpp — one client of a Run: a network and its measurement.
//
// A Network is the subject AND its measurement: it owns the SM allocation, the
// TensorRT engine and the inputs, and it knows how to warm itself up, measure
// itself and write its results.
//

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "gpu.hpp"
#include "interferer.hpp"
#include "inputs.hpp"
#include "stats.hpp"

// What one network needs to know about the protocol, plus where its results go.
// Deliberately not the whole Options: a network has no business knowing about
// other networks, datasets or allocation kinds.
struct RunSpec {
    std::size_t minInferences = cfg::minInferences;
    double      minDurationS  = cfg::minDurationS;
    std::string csvPrefix     = std::string(cfg::defaultCsvPrefix);
    Periodic    periodic;                       // empty levels = back to back
};

// Set by SIGTERM/SIGINT (handler installed in main). The warm-up and timed loops
// check it after each inference: the process finishes the work in flight and
// leaves through the normal CUDA teardown. Dying mid-kernel instead made the MPS
// server take ~3 s to clean up the client, and the next clients then hung in
// context creation.
extern std::atomic<bool> g_stop;

// Everyone spins until all have arrived, so several networks begin their timed
// phase in the same instant. The gate is what makes the two timed windows
// identical, and identical windows are what make their throughputs comparable.
// With one participant it is a no-op, which is how the solo path reuses the
// concurrent code.
// Applies the CPU policy to the thread that submits: declared here because both the client and
// the orchestrator call it.
void applyCpuPolicy(const Options& o);

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
    int                inputHeight()    const noexcept { return engine_.inputHeight(); }
    int                inputWidth()     const noexcept { return engine_.inputWidth(); }
    bool               fixedInputs()    const noexcept { return inputSet_.isFixed(); }

    void warmup();                    // cfg::numWarmup inferences, not measured
    void measure(StartGate& gate);    // the timed phase, aligned with the others

    // The single-submitter variant: one thread serves every network, keeping
    // one inference in flight per network. Static so that it can reach the
    // private measurement state of each of them; the loop itself lives in
    // run.cpp next to measure(), which it deliberately does not replace.
    static void measureInterleaved(const std::vector<Network*>& nets);
    void report() const;              // statistics table + throughput
    void writeStats() const;          // <prefix>_stats.csv

    std::size_t timedCount()     const noexcept { return timedCount_; }
    double      timedDurationS() const noexcept { return timedDurationS_; }

private:
    // One inference: fetch the input, run it, measure the three timings. tStartNs and
    // tEndNs place it in time on CLOCK_MONOTONIC (host side: before the input fetch, and
    // when the host sees it complete), so its latency can be matched to what a co-runner
    // logs on the same clock, e.g. the ON and OFF phases of an itfproc stressor.
    // tReleaseNs is when the job was released: its start in the closed loop, its slot
    // in the schedule under --levels. levelHz is the base rate of its load level, 0 in
    // the closed loop, so that a file says by itself which load each job saw.
    struct Sample {
        float prepMs = 0, wallMs = 0, gpuMs = 0;
        std::int64_t tStartNs = 0, tEndNs = 0, tReleaseNs = 0;
        double levelHz = 0;
    };
    Sample step(std::size_t index);
    void   measurePeriodic(StartGate& gate);   // measure() under --levels
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
