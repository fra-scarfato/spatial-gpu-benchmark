#pragma once

// The synthetic participant: a third client in the same joint SM split whose work is either
// memory traffic or arithmetic and never both, so the two can be told apart. It lives in its
// own thread and stream inside its own allocation, and reports the load it applied.

#include <cstddef>
#include <memory>

#include "gpu.hpp"   // SmAllocation: this pass leaves the device layer where it is

// A synthetic participant: one kind of work, one SM partition, one number to report.
//
// It exists to separate two things a real co-runner cannot: how much of the target's
// slowdown is the co-runner's arithmetic and how much is its memory traffic. The
// memory kind's working set has to exceed L2 (4 MB here) or it measures the cache; the
// compute kind's operands have to stay inside L2 or it becomes memory-bound too. Those
// two constraints are the whole design.
class Interferer {
public:
    enum class Kind { MemSaxpy, ComputeGemm };

    Interferer(Kind kind, std::unique_ptr<SmAllocation> allocation, std::size_t megabytes);
    ~Interferer();

    Interferer(const Interferer&)            = delete;
    Interferer& operator=(const Interferer&) = delete;

    const char* kindName()       const noexcept;
    std::size_t workingSetMB()   const noexcept { return mb_; }

    void start();    // spawn the loop: submit a unit, wait for it, repeat
    void stop();     // signal, join, close the window
    void report() const;

private:
    void submit();   // enqueue one unit, counted

    Kind          kind_;
    // Owned, and declared before the stream created from it so that the stream is
    // destroyed first. Holding a reference here instead was a real bug: the
    // allocation is a unique_ptr that the Run constructor moves away, so the
    // reference dangled from the moment that constructor returned, and the first
    // virtual call on it — name(), in report() — segfaulted.
    std::unique_ptr<SmAllocation> allocation_;
    cudaStream_t  stream_   = nullptr;
    cudaEvent_t   startEvt_ = nullptr;
    cudaEvent_t   stopEvt_  = nullptr;
    void*         dA_       = nullptr;
    void*         dB_       = nullptr;
    void*         dC_       = nullptr;
    void*         handle_   = nullptr;   // cublasHandle_t, opaque here so that the
                                         // header does not drag cublas in
    std::size_t   mb_       = 0;         // working set per buffer
    std::size_t   n_        = 0;         // elements
    std::size_t   units_    = 0;
    double        bytes_    = 0.0;
    double        flops_    = 0.0;
    double        elapsedS_ = 0.0;
    std::atomic<bool> stop_{false};
    std::thread       thread_;
};
