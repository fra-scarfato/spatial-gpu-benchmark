// This file was split out of gpu.cpp; the code is unchanged.

#include "gpu.hpp"
#include <cublas_v2.h>
#include <thread>
#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "interferer.hpp"

cudaError_t launchSpin(cudaStream_t stream, int blocks, long long iters);   // spin.cu

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

Interferer::Interferer(Kind kind, std::unique_ptr<SmAllocation> allocation, std::size_t megabytes,
                       int gemmSide)
    : kind_(kind), allocation_(std::move(allocation)), mb_(megabytes), gemmSide_(gemmSide) {
    // The buffers are sized by what the kind uses. 256 MB per buffer belongs to the memory
    // kind, whose traffic has to leave the chip; the compute kind needs one GEMM side squared
    // per operand, and that side is the whole variable of a footprint sweep. Three 256 MB
    // buffers for a 64-cubed GEMM would report a working set three orders of magnitude above
    // what it touches, and the idle kind needs nothing at all.
    std::size_t bytes = 0;
    if (kind_ == Kind::MemSaxpy) {
        bytes = mb_ * 1024ull * 1024ull;
        n_ = bytes / sizeof(float);
    } else if (kind_ == Kind::ComputeGemm) {
        n_ = std::size_t(gemmSide_) * std::size_t(gemmSide_);
        bytes = n_ * sizeof(float);
    }
    if (bytes > 0) {
        if (cudaMalloc(&dA_, bytes) != cudaSuccess ||
            cudaMalloc(&dB_, bytes) != cudaSuccess ||
            cudaMalloc(&dC_, bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc (interferer) failed");
        // Touched once here, so that the first unit is not charged for faulting pages in.
        if (cudaMemset(dA_, 1, bytes) != cudaSuccess ||
            cudaMemset(dB_, 1, bytes) != cudaSuccess)
            throw std::runtime_error("cudaMemset (interferer) failed");
    }
    stream_ = allocation_->createStream();     // created inside this partition
    if (cudaEventCreate(&startEvt_) != cudaSuccess ||
        cudaEventCreate(&stopEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (interferer) failed");

}

Interferer::~Interferer() {
    stop();
    if (startEvt_) cudaEventDestroy(startEvt_);
    if (stopEvt_)  cudaEventDestroy(stopEvt_);
    if (stream_)   cudaStreamDestroy(stream_);
    if (dA_) cudaFree(dA_);
    if (dB_) cudaFree(dB_);
    if (dC_) cudaFree(dC_);
}

const char* Interferer::kindName() const noexcept {
    switch (kind_) {
        case Kind::MemSaxpy:    return "mem_saxpy";
        case Kind::ComputeGemm: return "compute_gemm";
        case Kind::Idle:        return "idle";
        case Kind::Spin:        return "spin";
    }
    return "?";
}

void Interferer::submit() {
    cublasHandle_t h = static_cast<cublasHandle_t>(handle_);

    if (kind_ == Kind::MemSaxpy) {
        // Reads x, reads and writes y, so three times the working set per unit. At
        // 256 MB that traffic cannot sit in a 4 MB cache: one FLOP per element against
        // twelve bytes moved, which is as memory-bound as a kernel gets.
        const float alpha = 1.0f;
        cublasSaxpy(h, static_cast<int>(n_), &alpha,
                    static_cast<const float*>(dA_), 1, static_cast<float*>(dB_), 1);
        bytes_ += 3.0 * double(mb_) * 1024.0 * 1024.0;
    } else if (kind_ == Kind::ComputeGemm) {
        // 512 cubed in fp32: three 1 MB operands, so the working set fits in the 4 MB
        // L2 and the unit is arithmetic rather than traffic — 268 MFLOP against 3 MB
        // of operand, if the operands stay resident. Repeated inside one unit because
        // a single GEMM lasts about a hundred microseconds, which would spend more
        // time in the launch path than in the arithmetic.
        const float one = 1.0f, zero = 0.0f;
        // The same total arithmetic as 512 cubed repeated 32 times, whatever the side, so that
        // a unit costs about the same in every arm of a footprint sweep.
        const int m = gemmSide_, k = gemmSide_, n = gemmSide_;
        const int reps =
            std::max(1, int(4300000000.0 / (double(m) * double(n) * double(k))));
        for (int rep = 0; rep < reps; ++rep)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                        static_cast<const float*>(dA_), m,
                        static_cast<const float*>(dB_), k,
                        &zero, static_cast<float*>(dC_), m);
        flops_ += double(reps) * 2.0 * m * n * k;
        } else if (kind_ == Kind::Spin) {
            // One long kernel per unit, 16 blocks of 256 threads: resident in a single wave on four
            // SMs or eight, so the unit adds one launch and sixteen block dispatches every few
            // milliseconds and nothing else. About 3e6 dependent FMAs per thread is ~10 ms here.
            if (launchSpin(stream_, 16, 3000000LL) != cudaSuccess)
                throw std::runtime_error("spin kernel launch failed");
        } else if (kind_ == Kind::Idle) {
            // Occupies its partition and asks the device for nothing: the control for whether
            // the floor of the penalty is contention at all, or merely the existence of a
            // second partition. Sleeping rather than spinning keeps the CPU out of it.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else {
        // Not SM work at all: kept as a control for the copy path, which an SM
        // partition cannot contain because it does not run on SMs.
        cudaMemcpyAsync(dB_, dA_, mb_ * 1024ull * 1024ull,
                        cudaMemcpyDeviceToDevice, stream_);
        bytes_ += 2.0 * double(mb_) * 1024.0 * 1024.0;
    }
    ++units_;
}

void Interferer::start() {
    if (thread_.joinable()) return;
    cudaStreamSynchronize(stream_);      // nothing earlier left in the accounting
    stop_.store(false, std::memory_order_relaxed);
    cudaEventRecord(startEvt_, stream_);
    thread_ = std::thread([this] {
        if (kind_ == Kind::Idle) {
            while (!stop_.load(std::memory_order_relaxed))
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return;
        }
        // The cuBLAS handle is created HERE, not in the constructor. A handle is
        // bound to the thread that created it and is not thread-safe: making it on
        // the main thread and using it here is undefined behaviour, and it
        // segfaulted. It is destroyed here for the same reason.
        cublasHandle_t h;
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
            stop_.store(true, std::memory_order_relaxed);
            return;
        }
        cublasSetStream(h, stream_);
        handle_ = h;
        while (!stop_.load(std::memory_order_relaxed)) {
            submit();
            cudaStreamSynchronize(stream_);
        }
        cublasDestroy(h);
        handle_ = nullptr;
    });
}

void Interferer::stop() {
    if (!thread_.joinable()) return;
    stop_.store(true, std::memory_order_relaxed);
    thread_.join();
    cudaEventRecord(stopEvt_, stream_);
    cudaStreamSynchronize(stream_);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, startEvt_, stopEvt_);
    elapsedS_ = ms / 1e3;
}

void Interferer::report() const {
    if (units_ == 0 && kind_ != Kind::Idle) return;
    std::cout << "\n--- interferer '" << kindName() << "' ---\n"
              << "  allocation:   " << allocation_->name() << "\n"
              << "  working set:  "
              << (kind_ == Kind::ComputeGemm
                      ? 3.0 * double(gemmSide_) * double(gemmSide_) * 4.0 / 1048576.0
                      : (kind_ == Kind::Idle ? 0.0 : double(mb_)))
              << " MB" << (kind_ == Kind::ComputeGemm ? " (three operands)" : " per buffer") << "\n"
              << (kind_ == Kind::ComputeGemm
                      ? "  GEMM side:    " + std::to_string(gemmSide_) + "\n"
                      : "")
              << "  units:        " << units_ << " in " << elapsedS_ << " s"
              << " (" << (elapsedS_ > 0 ? units_ / elapsedS_ : 0.0) << " units/s)\n";
    if (bytes_ > 0)
        std::cout << "  bytes moved:  " << bytes_ / 1e9 << " GB ("
                  << (elapsedS_ > 0 ? bytes_ / elapsedS_ / 1e9 : 0.0) << " GB/s)\n";
    if (flops_ > 0)
        std::cout << "  arithmetic:   " << flops_ / 1e9 << " GFLOP ("
                  << (elapsedS_ > 0 ? flops_ / elapsedS_ / 1e12 : 0.0) << " TFLOP/s)\n";
    if (bytes_ > 0 && flops_ > 0)
        std::cout << "  FLOP per byte:" << flops_ / bytes_ << "\n";
}

