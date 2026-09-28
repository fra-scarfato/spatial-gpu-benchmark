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

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

Interferer::Interferer(Kind kind, std::unique_ptr<SmAllocation> allocation, std::size_t megabytes)
    : kind_(kind), allocation_(std::move(allocation)), mb_(megabytes) {
    const std::size_t bytes = mb_ * 1024ull * 1024ull;
    n_ = bytes / sizeof(float);
    if (cudaMalloc(&dA_, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (interferer A) failed");
    if (cudaMalloc(&dB_, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (interferer B) failed");
    if (cudaMalloc(&dC_, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (interferer C) failed");
    // Touched once here, so that the first unit is not charged for faulting pages in.
    if (cudaMemset(dA_, 1, bytes) != cudaSuccess || cudaMemset(dB_, 1, bytes) != cudaSuccess)
        throw std::runtime_error("cudaMemset (interferer) failed");

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
        const int m = 512, k = 512, n = 512;
        for (int rep = 0; rep < 32; ++rep)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                        static_cast<const float*>(dA_), m,
                        static_cast<const float*>(dB_), k,
                        &zero, static_cast<float*>(dC_), m);
        flops_ += 32.0 * 2.0 * m * n * k;
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
    if (units_ == 0) return;
    std::cout << "\n--- interferer '" << kindName() << "' ---\n"
              << "  allocation:   " << allocation_->name() << "\n"
              << "  working set:  " << mb_ << " MB per buffer\n"
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

