// utility/probes/itfproc.cu — the interferer as its own client process.
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
//   itfproc saxpy --level L [--mb 256]  the same SAXPY as mem_saxpy, written as our own kernel
//                                       so it can be throttled from inside: each thread moves a
//                                       burst of elements, then sleeps. L in [0, 1] is the
//                                       ACHIEVED BANDWIDTH as a fraction of the same kernel at
//                                       full speed on the same partition. L = 0 keeps a kernel
//                                       resident that moves no data.
//   itfproc saxpy --period-us P --duty D  TIME-BASED: all threads move data at full speed during
//                                       the first D x P of every period P on the GPU's global
//                                       clock, together, and wait for the rest. D is the fraction
//                                       of TIME on; the achieved bandwidth fraction is printed.
//   itfproc saxpy ... --scramble        the same bytes, still whole 128-byte lines read by a whole
//                                       warp, but the lines are visited in a scrambled order
//                                       across the buffers: only the DRAM locality changes.
//
// Prints the same lines the runner's interferer prints, once a second, so the log carries
// the load over time and not only its total, and prints the total again on SIGTERM, which
// is how the sweep stops it.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

__device__ __forceinline__ unsigned long long globalTimerNs() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// A wait of exactly `ns`, timed on the GPU's own clock. __nanosleep alone is only a hint (anything
// from zero to about twice the request, capped near a millisecond), which made the achieved level
// stop following the setpoint below a third of full speed.
__device__ __forceinline__ void waitNs(unsigned long long ns) {
    const unsigned long long t0 = globalTimerNs();
    while (globalTimerNs() - t0 < ns) __nanosleep(200);
}

// y = a*x + y, grid-stride. After every `burst` elements a thread waits `sleepNs`: thousands of
// threads, each out of phase with the others, so the pressure on memory is lowered uniformly at
// microsecond scale rather than switched on and off. burst == 0 is level 0: the kernel stays
// resident for `idleNs` and touches no memory.
// periodNs != 0 is the time-based mode: before each burst a thread checks the global clock and,
// outside the first onNs of the period, waits for the next period, so every thread is on and off
// at the same moments.
// lineMask != 0 scrambles: element i goes to line (i / 32) * odd constant, modulo the number of
// lines (a power of two, so the map is a bijection), at the same lane. A warp still touches one
// whole line; consecutive lines land far apart.
__global__ void saxpyThrottled(const float* __restrict__ x, float* __restrict__ y, size_t n,
                               float a, unsigned burst, unsigned sleepNs,
                               unsigned long long idleNs, unsigned long long periodNs,
                               unsigned long long onNs, size_t lineMask) {
    if (burst == 0) {
        waitNs(idleNs);
        return;
    }
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned k = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
        if (periodNs && k == 0) {
            const unsigned long long phase = globalTimerNs() % periodNs;
            if (phase >= onNs) waitNs(periodNs - phase);
        }
        const size_t j = lineMask ? (i & ~((lineMask << 5) | 31)) |
                                        ((((i >> 5) * 0x9E3779B1ull) & lineMask) << 5) | (i & 31) : i;
        y[j] = a * x[j] + y[j];
        if (++k == burst) {
            k = 0;
            if (sleepNs) waitNs(sleepNs);
        }
    }
}

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
    if (kind == "mem_saxpy" || kind == "saxpy") {
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
    double level = 1.0;
    double periodUs = 0.0, duty = 1.0;
    bool scramble = false;
    unsigned scrambleKb = 0;
    const int gemmN = 512;

    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mb") && i + 1 < argc) {
            mb = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--level") && i + 1 < argc) {
            level = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--period-us") && i + 1 < argc) {
            periodUs = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--duty") && i + 1 < argc) {
            duty = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--scramble")) {
            scramble = true;
        } else if (!std::strcmp(argv[i], "--scramble-kb") && i + 1 < argc) {
            scramble = true;
            scrambleKb = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            std::fprintf(stderr, "[itfproc] unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    if (kind != "mem_saxpy" && kind != "compute_gemm" && kind != "saxpy") {
        std::fprintf(stderr, "[itfproc] kind must be mem_saxpy, compute_gemm or saxpy, not '%s'\n",
                     kind.c_str());
        return 2;
    }
    if (level < 0.0 || level > 1.0) {
        std::fprintf(stderr, "[itfproc] --level must be in [0, 1], got %g\n", level);
        return 2;
    }
    if (periodUs > 0.0 && (kind != "saxpy" || duty <= 0.0 || duty > 1.0 || level != 1.0)) {
        std::fprintf(stderr, "[itfproc] --period-us is for saxpy, with --duty in (0, 1] and no --level\n");
        return 2;
    }
    if (scramble && (kind != "saxpy" || (mb & (mb - 1)) || (scrambleKb & (scrambleKb - 1)) ||
                     scrambleKb > mb * 1024u || (scrambleKb && scrambleKb < 4))) {
        std::fprintf(stderr, "[itfproc] --scramble is for saxpy, with --mb and --scramble-kb powers of two, 4 KB to the buffer\n");
        return 2;
    }
    const bool throttled = (kind == "saxpy");
    const bool mem = (kind == "mem_saxpy") || throttled;

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

    if (throttled) {
        const size_t n = static_cast<size_t>(elements);
        int perSm = 0, sms = 0;
        must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, saxpyThrottled, 256, 0),
             "occupancy");
        must(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
        const unsigned blocks = static_cast<unsigned>(perSm * sms);
        const unsigned burst = 16;
        const double perThread = std::ceil(static_cast<double>(n) / (blocks * 256.0));
        const double burstsPerThread = std::ceil(perThread / burst);

        const unsigned long long periodNs = static_cast<unsigned long long>(periodUs * 1e3);
        const unsigned long long onNs = static_cast<unsigned long long>(periodUs * 1e3 * duty);
        const size_t lineMask = !scramble ? 0 : scrambleKb ? scrambleKb * 1024ull / 128 - 1 : n / 32 - 1;
        // One unit is one kernel over the whole vectors; its duration is the measurement.
        auto unitSeconds = [&](unsigned b, unsigned sleepNs, unsigned long long idleNs,
                               unsigned long long pNs = 0) {
            const auto a = std::chrono::steady_clock::now();
            saxpyThrottled<<<blocks, 256, 0, stream>>>(dx, dy, n, alpha, b, sleepNs, idleNs, pNs, onNs, lineMask);
            must(cudaGetLastError(), "saxpyThrottled launch");
            must(cudaStreamSynchronize(stream), "synchronize");
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
        };
        auto median = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };

        // 1. Full speed on this partition: the reference every level is a fraction of.
        std::vector<double> warm;
        for (int i = 0; i < 9; ++i) warm.push_back(unitSeconds(burst, 0, 0));
        const double tFull = median(warm);

        // 2. The wait that gives the setpoint, found once and then frozen, so the level does not
        //    hunt during the measurement. With exact waits a unit lasts tFull + bursts * wait (the
        //    first measurements agreed with this to 2 %), so the model gives the wait directly and
        //    three short corrections absorb the rest: under a second even at L = 0.1, well inside
        //    the 3 s the sweep gives a co-runner before the target starts.
        unsigned b = burst, sleepNs = 0;
        unsigned long long idleNs = 0;
        if (level == 0.0) {
            b = 0;
            idleNs = static_cast<unsigned long long>(tFull * 1e9);
        } else if (level < 1.0) {
            const double target = tFull / level;
            double s = (target - tFull) / burstsPerThread * 1e9;
            for (int round = 0; round < 3; ++round) {
                std::vector<double> t;
                for (int i = 0; i < 3; ++i) t.push_back(unitSeconds(b, static_cast<unsigned>(s), 0));
                s = std::max(0.0, s + (target - median(t)) / burstsPerThread * 1e9);
            }
            sleepNs = static_cast<unsigned>(s);
        }
        std::printf("  throttle:     %u blocks x 256, burst %u, sleep %u ns, full-speed unit %.3f ms\n",
                    blocks, b, sleepNs, tFull * 1e3);
        if (periodNs)
            std::printf("  time-based:   period %.0f us, on %.0f us (duty %.2f), all threads together\n",
                        periodUs, periodUs * duty, duty);
        if (scramble) std::printf("  scramble:     %zu lines of 128 B in scrambled order within each window\n", lineMask + 1);

        // 3. The measurement. The achieved level of a unit is tFull over its duration (the same
        //    bytes in more time); the line printed every second carries the median of that window.
        std::vector<double> window;
        const auto t0 = std::chrono::steady_clock::now();
        while (!g_stop) {
            const double t = unitSeconds(b, sleepNs, idleNs, periodNs);
            window.push_back(level == 0.0 ? 0.0 : tFull / t);
            ++units;
            if (level > 0.0) bytes += 3.0 * static_cast<double>(bytesPerBuffer);
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (elapsed - lastReport >= 1.0) {
                lastReport = elapsed;
                report(kind, mb, units, elapsed, bytes, flops);
                std::printf("  level:        setpoint %.2f achieved %.4f\n", level, median(window));
                std::fflush(stdout);
                window.clear();
            }
        }
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("\n--- interferer '%s' final ---\n", kind.c_str());
        report(kind, mb, units, elapsed, bytes, flops);
        cudaFree(dx);
        cudaFree(dy);
        cublasDestroy(handle);
        cudaStreamDestroy(stream);
        return units > 0 ? 0 : 1;
    }

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
