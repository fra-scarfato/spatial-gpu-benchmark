//
// network.cpp — a Network: one client's engine, inputs, warm-up, timed phase and results.
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
#include <cmath>
#include <cstring>
#include <time.h>

std::atomic<bool> g_stop{false};
static_assert(std::atomic<bool>::is_always_lock_free);   // safe to set from a signal handler

namespace {

// steady_clock, not system_clock: it is monotonic, so an NTP step during a
// 30-second measurement window cannot make an elapsed time negative or absurd.
// On Linux this maps to CLOCK_MONOTONIC.
using Clock = std::chrono::steady_clock;

double secondsSince(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// A time point as nanoseconds of CLOCK_MONOTONIC (steady_clock's epoch on Linux), the
// clock other processes on the board can read too.
std::int64_t monoNs(Clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count();
}

// Sleeps until t, in CLOCK_MONOTONIC nanoseconds, and returns at once if t is past. An
// absolute deadline (TIMER_ABSTIME), not a relative sleep, so the lateness of one wake-up
// never carries into the next release. A signal ends the sleep only when it asked to stop.
void sleepUntil(std::int64_t t) {
    const timespec ts{static_cast<time_t>(t / 1000000000), static_cast<long>(t % 1000000000)};
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr) == EINTR && !g_stop) {
    }
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
      inputSet_(opts, images, std::size_t(engine_.inputVolume()),
                engine_.inputHeight(), engine_.inputWidth()),
      spec_(spec),
      csv_(spec.csvPrefix + "_latencies.csv")
{
    // ofstream does not throw by default, so an unopened file looks like a
    // working stream that silently discards every row.
    if (!csv_)
        throw std::runtime_error("cannot open " + spec.csvPrefix + "_latencies.csv");
    csv_ << "phase,index,image,prep_ms,wall_ms,gpu_ms,t_start_ns,t_end_ns,level_hz,t_release_ns\n";
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
    const auto t2 = Clock::now();

    // prep_ms is the host cost of fetching THIS network's input, and is a
    // disjoint slice of the step rather than extra time. wall_ms is deliberately
    // NOT a host span: it comes from events on this network's stream over
    // H2D + kernel + D2H, so it stays this network's number even when one thread
    // drives several networks at once.
    Sample s;
    s.prepMs = float(std::chrono::duration<double, std::milli>(t1 - t0).count());
    s.wallMs = float(wallMs);
    s.gpuMs  = float(gpuMs);
    s.tStartNs = monoNs(t0);
    s.tEndNs   = monoNs(t2);
    s.tReleaseNs = s.tStartNs;      // the closed loop releases a job when it starts it
    return s;
}

void Network::writeRow(const char* phase, std::size_t index, const Sample& s) {
    // "phase" separates warmup from bench so the warm-up curve stays in the
    // file without contaminating the statistics. "image" is the input's id,
    // which in fixed mode is a set index rather than a file name.
    csv_ << phase << "," << index << "," << inputSet_.idOf(index) << ","
         << s.prepMs << "," << s.wallMs << "," << s.gpuMs << ","
         << s.tStartNs << "," << s.tEndNs << "," << s.levelHz << "," << s.tReleaseNs << "\n";
}

void Network::warmup() {
    // Warm-up rows are written to the CSV so the ramp is visible, but they are
    // never pushed into the statistics vectors: warm-up is not the steady
    // state, and the steady state is what is being characterised. The first
    // inference costs ~37 ms against a ~0.54 ms steady state.
    std::cout << "[info] warm-up: " << cfg::numWarmup << " inferences over "
              << inputSet_.count() << " distinct inputs\n";

    for (std::size_t i = 0; i < cfg::numWarmup && !g_stop; ++i)
        writeRow("warmup", i, step(i));
}

void Network::measure(StartGate& gate) {
    if (!spec_.periodic.levels.empty()) {
        measurePeriodic(gate);
        return;
    }

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
        if (g_stop) break;
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

    // A co-runner stopped by the sweep writes no results, but its throughput is still wanted (the
    // inferences everyone completed, for the energy per inference), so the count goes to the log.
    if (g_stop)
        std::cerr << std::format("[info] {} timed inferences in {:.2f} s before the stop\n",
                                 timedCount_, timedDurationS_);
}

// The timed phase under --levels (Periodic in config.hpp). The schedule is pure
// arithmetic on the origin, so the processes of a run, given the same origin and
// levels, compute the same windows without talking to each other.
void Network::measurePeriodic(StartGate& gate) {
    const Periodic& p = spec_.periodic;
    gate.arriveAndWait();

    const std::int64_t now = monoNs(Clock::now());
    std::int64_t window = p.originNs ? p.originNs : now + 1000000000;
    if (window < now)
        throw std::runtime_error(std::format(
            "the origin is {:.2f} s in the past: the warm-up took longer than the sweep "
            "allowed (raise ORIGIN_DELAY_S)", (now - window) / 1e9));
    std::cout << std::format("[info] timed phase: {} levels from {:g} Hz, {} ticks each, every {}, "
                             "offset {:g} ticks; first window in {:.2f} s\n", p.levels.size(),
                             p.levels[0], p.jobsPerLevel, p.every, p.offsetFrac, (window - now) / 1e9);

    const std::int64_t first = window;
    std::size_t n = 0;
    bool overload = false;
    for (std::size_t k = 0; k < p.levels.size() && !overload && !g_stop; ++k) {
        const double tick = 1e9 / p.levels[k];
        const std::int64_t end = window + std::llround(double(p.jobsPerLevel) * tick);
        std::size_t jobs = 0;
        for (std::size_t m = 0; !g_stop; ++m) {
            const std::int64_t release = window + std::llround((p.offsetFrac + double(m * p.every)) * tick);
            if (release >= end)
                break;
            sleepUntil(release);
            Sample s = step(n);
            s.levelHz    = p.levels[k];
            s.tReleaseNs = release;
            writeRow("bench", n, s);
            gpuTimes_.push_back(s.gpuMs);
            wallTimes_.push_back(s.wallMs);
            prepTimes_.push_back(s.prepMs);
            ++n;
            ++jobs;
            const double lagMs = (s.tStartNs - release) / 1e6;
            if (lagMs > p.maxLagMs) {
                std::cout << std::format("[warn] overload at level {} ({:g} Hz): job {} started {:.0f} ms "
                                         "after its release; stopping\n", k, p.levels[k], jobs, lagMs);
                overload = true;
                break;
            }
        }
        std::cout << std::format("  ...level {} ({:g} Hz): {} jobs\n", k, p.levels[k], jobs);
        window = end + 1000000000;        // the drain gap
    }

    timedCount_     = n;
    timedDurationS_ = (monoNs(Clock::now()) - first) / 1e9;
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

            // The end stamp is when this loop sees the inference done, so it can be late by
            // up to one pass over the other networks; the start stamp is exact.
            Sample s;
            s.tEndNs = monoNs(Clock::now());
            s.tStartNs = monoNs(t0[i]);
            s.gpuMs  = float(nets[i]->engine_.finish());
            s.wallMs = float(nets[i]->engine_.wallMs());   // per network, not per thread
            s.prepMs = float(std::chrono::duration<double, std::milli>(t1[i] - t0[i]).count());

            nets[i]->writeRow("bench", count[i], s);
            nets[i]->gpuTimes_.push_back(s.gpuMs);
            nets[i]->wallTimes_.push_back(s.wallMs);
            nets[i]->prepTimes_.push_back(s.prepMs);
            ++count[i];

            const double elapsed = secondsSince(start);
            const bool finished = g_stop ||
                (enforce[i]
                    ? (count[i] >= nets[i]->spec_.minInferences &&
                       elapsed >= nets[i]->spec_.minDurationS)
                    : (count[i] >= onePass[i]));

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

// Pin this thread and, if asked, raise its scheduling policy.
//
// Called by the submitting thread itself, because affinity and scheduling policy are
// properties of a THREAD: no other thread can set them for it. Threads created
// afterwards inherit both, which is why one call covers the two-thread path too.
//
// It refuses to continue when the setting does not take. That is not defensiveness:
// `nice` prints a warning and runs the command anyway, so an arm that asked for
// priority, did not get it and still measured would be indistinguishable from a
// result showing that priority changes nothing.
void applyCpuPolicy(const Options& o) {
    if (o.cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(o.cpu, &set);
        if (sched_setaffinity(0, sizeof(set), &set) != 0)
            throw std::runtime_error("--cpu " + std::to_string(o.cpu) +
                                     " failed: " + std::strerror(errno));
    }
    if (!o.schedPolicy.empty()) {
        struct sched_param sp {};
        sp.sched_priority = o.schedPriority;
        const int policy = (o.schedPolicy == "fifo") ? SCHED_FIFO : SCHED_RR;
        if (sched_setscheduler(0, policy, &sp) != 0)
            throw std::runtime_error(
                "--priority " + o.schedPolicy + ":" + std::to_string(o.schedPriority) +
                " failed: " + std::strerror(errno) +
                " -- needs CAP_SYS_NICE, e.g. sudo setcap cap_sys_nice+ep runner");
    }
}


