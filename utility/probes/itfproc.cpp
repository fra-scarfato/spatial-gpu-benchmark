// utility/probes/itfproc.cpp — the interferer as its own client process.
//
// The in-process interferer shares the target's CUDA context, and therefore its launch
// path. That is exactly the variable the MPS static arms need to move: here the same work
// stands alone, in its own process, with its own context, confined by an MPS static
// partition. Same kernels, same sizes, same reporting, so an MPS row and a green-context
// row differ in the process model and in nothing else.
//
//   itfproc mem_saxpy    [--mb 256]     SAXPY over two buffers of 256 MiB, three times
//                                       their size moved per pass (read x, read y, write y)
//   itfproc compute_gemm [--reps 32]    32 fp32 512-cubed SGEMMs per unit, operands
//                                       resident in L2
//
// Prints the same lines the runner's interferer prints, once a second, so the log carries
// the load over time and not only its total, and prints the total again on SIGTERM, which
// is how the sweep stops it.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

volatile std::sig_atomic_t g_stop = 0;
void onSignal(int) { g_stop = 1; }

// Every CUDA and cuBLAS call is checked: a process that silently did nothing would be
// indistinguishable from a co-runner that costs nothing, which is the one mistake that
// would invalidate the whole comparison.
void must(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "[itfproc] %s failed: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void must(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "[itfproc] %s failed (cublas status %d)\n", what, static_cast<int>(s));
        std::exit(1);
    }
}

void report(const std::string& kind, unsigned mb, long units, double seconds,
            double bytes, double flops) {
    const double perSecond = seconds > 0 ? units / seconds : 0.0;
    std::printf("  units:        %ld in %.5f s (%.4f units/s)\n", units, seconds, perSecond);
    if (kind == "mem_saxpy") {
        std::printf("  bytes moved:  %.2f GB (%.3f GB/s)\n", bytes / 1e9, bytes / 1e9 / seconds);
        (void)mb;
    } else {
        std::printf("  arithmetic:   %.2f GFLOP (%.6f TFLOP/s)\n",
                    flops / 1e9, flops / 1e12 / seconds);
    }
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::string kind = argc > 1 ? argv[1] : "mem_saxpy";
    unsigned mb = 256;
    int reps = 32;
    const int gemmN = 512;

    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mb") && i + 1 < argc) {
            mb = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "[itfproc] unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (kind != "mem_saxpy" && kind != "compute_gemm") {
        std::fprintf(stderr, "[itfproc] kind must be mem_saxpy or compute_gemm, not '%s'\n",
                     kind.c_str());
        return 2;
    }
    const bool mem = (kind == "mem_saxpy");

    std::signal(SIGTERM, onSignal);
    std::signal(SIGINT, onSignal);

    cudaStream_t stream = nullptr;
    must(cudaStreamCreate(&stream), "cudaStreamCreate");

    cublasHandle_t handle;
    must(cublasCreate(&handle), "cublasCreate");
    must(cublasSetStream(handle, stream), "cublasSetStream");

    const std::size_t bytesPerBuffer = static_cast<std::size_t>(mb) << 20;
    const int elements = static_cast<int>(bytesPerBuffer / sizeof(float));

    float* dx = nullptr;
    float* dy = nullptr;
    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    if (mem) {
        must(cudaMalloc(&dx, bytesPerBuffer), "cudaMalloc(x)");
        must(cudaMalloc(&dy, bytesPerBuffer), "cudaMalloc(y)");
        must(cudaMemsetAsync(dx, 0, bytesPerBuffer, stream), "cudaMemsetAsync(x)");
        must(cudaMemsetAsync(dy, 0, bytesPerBuffer, stream), "cudaMemsetAsync(y)");
    } else {
        const std::size_t operand = static_cast<std::size_t>(gemmN) * gemmN * sizeof(float);
        must(cudaMalloc(&dA, operand), "cudaMalloc(A)");
        must(cudaMalloc(&dB, operand), "cudaMalloc(B)");
        must(cudaMalloc(&dC, operand), "cudaMalloc(C)");
        // Non-zero operands: zeros could let a compiler or a library short-circuit.
        must(cudaMemsetAsync(dA, 1, operand, stream), "cudaMemsetAsync(A)");
        must(cudaMemsetAsync(dB, 1, operand, stream), "cudaMemsetAsync(B)");
        must(cudaMemsetAsync(dC, 0, operand, stream), "cudaMemsetAsync(C)");
    }
    must(cudaStreamSynchronize(stream), "warm-up synchronize");

    std::printf("\n--- interferer '%s' (separate process) ---\n", kind.c_str());
    std::printf("  process:      its own CUDA context\n");
    std::printf("  working set:  %u MB per buffer\n", mb);
    report(kind, mb, 0, 1.0, 0.0, 0.0);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    double bytes = 0.0;
    double flops = 0.0;
    long units = 0;
    double lastReport = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    while (!g_stop) {
        if (mem) {
            must(cublasSaxpy(handle, elements, &alpha, dx, 1, dy, 1), "cublasSaxpy");
        } else {
            for (int r = 0; r < reps; ++r) {
                must(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, gemmN, gemmN, gemmN,
                                 &alpha, dA, gemmN, dB, gemmN, &beta, dC, gemmN),
                     "cublasSgemm");
            }
        }
        must(cudaStreamSynchronize(stream), "synchronize");
        ++units;
        if (mem) {
            bytes += 3.0 * static_cast<double>(bytesPerBuffer);
        } else {
            flops += static_cast<double>(reps) * 2.0 * std::pow(static_cast<double>(gemmN), 3);
        }

        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed - lastReport >= 1.0) {
            lastReport = elapsed;
            report(kind, mb, units, elapsed, bytes, flops);
        }
    }

    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n--- interferer '%s' final ---\n", kind.c_str());
    report(kind, mb, units, elapsed, bytes, flops);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return units > 0 ? 0 : 1;
}
