// probes/itfproc.cu — the interferer as its own client process.
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
//   itfproc saxpy --period-us P --duty D --burst B  the same, each thread checking the clock every B
//                                       elements instead of 16. At 16 a thread's burst lasts ~23 us
//                                       on 8 SMs, so an ON part shorter than that overshoots; at B = 1
//                                       the ON part is followed to ~1.5 us, and a P of ~100 us spreads
//                                       even a low mean bandwidth evenly over the reporter's 0.3 ms
//                                       windows (block 2's smooth uniform, 2026-10-11).
//   itfproc saxpy --onoff-ms P --duty D  ON/OFF: whole full-speed units (the Steady kernel at L = 1)
//                                       back to back during the first D x P ms of every period, then
//                                       nothing resident until the next one. Each ON span is logged
//                                       on CLOCK_MONOTONIC, the clock of the runner's per-inference
//                                       stamps. With a small --mb and a short P the same mode spreads
//                                       the same mean bandwidth uniformly in time.
//   itfproc saxpy ... --scramble        the same bytes, still whole 128-byte lines read by a whole
//                                       warp, but the lines are visited in a scrambled order
//                                       across the buffers: only the DRAM locality changes.
//
// Three kinds separate the presence of a co-runner from its use of the L2 (the professor's
// follow-up of 2026-10-10). Each launch is one kernel; with --period-us P a launch starts every
// P us (the host waits in between, so nothing is resident in the gap), without it the kernels
// run back to back. The log carries launches per second and the wall time per launch.
//   itfproc spin --kernel-us K           a register-only FMA chain that stays resident for exactly
//                                       K us (GPU clock) and touches no memory: presence alone, at a
//                                       launch rate of 1/K back to back
//   itfproc gemm --side S --sets N       one fp32 S-cubed SGEMM per launch, round-robin over N
//                                       distinct (A, B, C) sets: the working set grows with N while
//                                       the work per kernel stays the same
//   itfproc l2read --kernel-us K --level L --kb 1024   a K-us kernel whose threads re-read a small
//                                       array through the L2 (bypassing L1), throttled to L x its
//                                       full read rate; same length and launch rate as spin, so
//                                       the difference between the two is the L2 traffic
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
#include <thread>
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
// The Steady path (--level, no period, no scramble): exactly the kernel of the first stress campaign
// (commit e0e5630), kept apart from saxpyThrottled below. Folding the time-based and scrambled modes
// into one kernel added a counter, a clock check and an index remap to every element, which made the
// full-speed unit 3.6 % slower (5.87 -> 6.09 ms on 8 SMs, 2026-10-09); a Steady level is a fraction
// of that full speed, so every level then pressed less on memory than the calibrated one.
__global__ void saxpyLevel(const float* __restrict__ x, float* __restrict__ y, size_t n, float a,
                           unsigned burst, unsigned sleepNs, unsigned long long idleNs) {
    if (burst == 0) {
        waitNs(idleNs);
        return;
    }
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned k = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
        if (sleepNs && ++k == burst) {
            k = 0;
            waitNs(sleepNs);
        }
    }
}

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

// spin: an FMA chain in registers, resident for exactly `ns` on the GPU clock. The one store is
// behind a condition that is never true, so the compiler keeps the loop and memory is never touched
// (src/spin.cu is the same chain, bounded by an iteration count instead of a time).
__global__ void spinFor(float* sink, unsigned long long ns) {
    const unsigned long long t0 = globalTimerNs();
    float a = threadIdx.x * 1e-3f;
    while (globalTimerNs() - t0 < ns)
        for (int i = 0; i < 64; ++i) a = fmaf(a, 0.999f, 1e-3f);
    if (a == -1.0f) *sink = a;
}

// l2read: for exactly `ns`, each thread reads `burst` floats of a small array with __ldcg (cache
// global: served by the L2, not the SM's L1), then waits `sleepNs`. The array is a few MB at most,
// so after the first pass the reads stay in the L2 and DRAM sees almost nothing. The bursts done
// are counted once per warp, so the counter itself adds no measurable traffic. burst == 0 reads
// nothing: the kernel is then resident and idle for `ns`, like spin.
__global__ void l2ReadFor(const float* __restrict__ a, size_t n, unsigned long long ns,
                          unsigned burst, unsigned sleepNs, float* sink,
                          unsigned long long* bursts) {
    const unsigned long long t0 = globalTimerNs();
    if (burst == 0) {
        waitNs(ns);
        return;
    }
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t i = (blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x) % n;
    float acc = 0.0f;
    unsigned done = 0;
    while (globalTimerNs() - t0 < ns) {
        for (unsigned k = 0; k < burst; ++k) {
            acc += __ldcg(a + i);
            i += stride;
            if (i >= n) i -= n;
        }
        ++done;
        if (sleepNs) waitNs(sleepNs);
    }
    if (acc == -1.0f) *sink = acc;
    const unsigned warpDone = __reduce_add_sync(0xffffffffu, done);
    if ((threadIdx.x & 31) == 0) atomicAdd(bursts, static_cast<unsigned long long>(warpDone));
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

// The presence and L2 kinds: spin, gemm and l2read. One launch is one kernel (one SGEMM for gemm).
//
//  - Pacing. With periodUs > 0 every launch is synchronized and the next one starts periodUs after
//    the previous start; the host busy-waits for that moment, because a sleep would add tens of us
//    of jitter at the 100-us periods used. Without a period the kernels are queued back to back and
//    synchronized every couple of milliseconds, so the GPU never waits for the host.
//  - l2read's level. As for saxpy, L is the ACHIEVED read rate as a fraction of the same kernel at
//    full speed on the same partition: a full-speed measurement gives the time of one burst, the
//    model wait = burst * (1/L - 1) gives the first guess, and three short corrections freeze it.
//  - Logging. Once a second: launches per second, the wall time per launch, and the work done
//    (GFLOP/s for gemm, L2 GB/s and the achieved level for l2read), in the same "units" format as
//    the other kinds so the sweep's log readers keep working.
int runPaced(const std::string& kind, double kernelUs, double periodUs, int side, int sets,
             unsigned kb, double level) {
    std::printf("\n--- interferer '%s' (separate process) ---\n", kind.c_str());
    std::printf("  process:      its own CUDA context\n");
    cudaStream_t stream = nullptr;
    must(cudaStreamCreate(&stream), "cudaStreamCreate");
    cublasHandle_t handle = nullptr;
    const unsigned long long kernelNs = static_cast<unsigned long long>(kernelUs * 1e3);

    // 1. Resources. spin and l2read fill the partition with resident blocks, as saxpy does.
    int perSm = 0, sms = 0;
    must(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
    must(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
             &perSm, kind == "spin" ? (const void*)spinFor : (const void*)l2ReadFor, 256, 0),
         "occupancy");
    const unsigned blocks = static_cast<unsigned>(perSm * sms);
    std::vector<float*> A, B, C;
    float* arr = nullptr;
    float* sink = nullptr;
    unsigned long long* dBursts = nullptr;
    const size_t n = static_cast<size_t>(kb) * 1024 / sizeof(float);
    must(cudaMalloc(&sink, sizeof(float)), "cudaMalloc(sink)");
    if (kind == "gemm") {
        must(cublasCreate(&handle), "cublasCreate");
        must(cublasSetStream(handle, stream), "cublasSetStream");
        const size_t operand = static_cast<size_t>(side) * side * sizeof(float);
        for (int k = 0; k < sets; ++k) {
            float *a, *b, *c;
            must(cudaMalloc(&a, operand), "cudaMalloc(A)");
            must(cudaMalloc(&b, operand), "cudaMalloc(B)");
            must(cudaMalloc(&c, operand), "cudaMalloc(C)");
            // Non-zero operands: zeros could let a library short-circuit.
            must(cudaMemsetAsync(a, 1, operand, stream), "cudaMemsetAsync(A)");
            must(cudaMemsetAsync(b, 1, operand, stream), "cudaMemsetAsync(B)");
            must(cudaMemsetAsync(c, 0, operand, stream), "cudaMemsetAsync(C)");
            A.push_back(a), B.push_back(b), C.push_back(c);
        }
    } else if (kind == "l2read") {
        must(cudaMalloc(&arr, n * sizeof(float)), "cudaMalloc(array)");
        must(cudaMemsetAsync(arr, 1, n * sizeof(float), stream), "cudaMemsetAsync(array)");
        must(cudaMalloc(&dBursts, sizeof(unsigned long long)), "cudaMalloc(counter)");
    }
    must(cudaStreamSynchronize(stream), "warm-up synchronize");

    const unsigned burst = 16;
    unsigned b = burst, sleepNs = 0;
    const float one = 1.0f, zero = 0.0f;
    int next = 0;
    auto launch = [&]() {
        if (kind == "spin") {
            spinFor<<<blocks, 256, 0, stream>>>(sink, kernelNs);
        } else if (kind == "l2read") {
            l2ReadFor<<<blocks, 256, 0, stream>>>(arr, n, kernelNs, b, sleepNs, sink, dBursts);
        } else {
            must(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, side, side, side, &one, A[next],
                             side, B[next], side, &zero, C[next], side), "cublasSgemm");
            next = (next + 1) % sets;
        }
        must(cudaGetLastError(), "launch");
    };
    // L2 bytes read by `k` kernels at the current setting, from the device counter.
    auto bytesRead = [&](int k) {
        unsigned long long z = 0, done = 0;
        must(cudaMemcpy(dBursts, &z, sizeof z, cudaMemcpyHostToDevice), "reset counter");
        for (int i = 0; i < k; ++i) launch();
        must(cudaMemcpy(&done, dBursts, sizeof done, cudaMemcpyDeviceToHost), "read counter");
        return static_cast<double>(done) * burst * sizeof(float);   // done: bursts of all threads
    };

    // 2. l2read's level (see above). The rate is bytes per kernel, the kernel length being fixed.
    double fullRate = 0.0;
    if (kind == "l2read") {
        fullRate = bytesRead(5) / 5.0;
        const double threads = static_cast<double>(blocks) * 256.0;
        const double burstNs = kernelNs / (fullRate / (threads * burst * sizeof(float)));
        if (level == 0.0) {
            b = 0;
        } else if (level < 1.0) {
            double s = burstNs * (1.0 / level - 1.0);
            for (int round = 0; round < 3; ++round) {
                sleepNs = static_cast<unsigned>(s);
                const double got = bytesRead(5) / 5.0 / fullRate;
                s = std::max(0.0, (burstNs + s) * got / level - burstNs);
            }
            sleepNs = static_cast<unsigned>(s);
        }
        std::printf("  l2read:       %zu KB array, %u blocks x 256, burst %u, sleep %u ns, "
                    "full speed %.1f GB/s\n", n * sizeof(float) / 1024, blocks, b, sleepNs,
                    fullRate / kernelUs * 1e-3);
    }
    if (kind == "gemm")
        std::printf("  gemm:         side %d, %d sets of A, B, C: working set %.2f MB\n", side, sets,
                    3.0 * side * side * sizeof(float) * sets / 1048576.0);
    if (kind == "spin") std::printf("  spin:         %u blocks x 256, resident %.0f us per launch\n",
                                    blocks, kernelUs);
    std::printf("  launches:     %s\n", periodUs > 0 ? ("one every " + std::to_string(periodUs) +
                                                       " us").c_str() : "back to back");
    std::fflush(stdout);

    // 3. The measurement.
    const double flopsPerLaunch = 2.0 * std::pow(static_cast<double>(side), 3);
    const int batch = periodUs > 0 ? 1 : std::max(1, static_cast<int>(2000.0 / std::max(kernelUs, 50.0)));
    long launches = 0, windowLaunches = 0;
    double bytes = 0.0, flops = 0.0, lastReport = 0.0;
    unsigned long long windowBursts = 0;
    if (kind == "l2read") must(cudaMemset(dBursts, 0, sizeof(unsigned long long)), "reset counter");
    const auto t0 = std::chrono::steady_clock::now();
    auto due = t0;
    while (!g_stop) {
        for (int i = 0; i < batch; ++i) launch();
        must(cudaStreamSynchronize(stream), "synchronize");
        launches += batch, windowLaunches += batch;
        if (kind == "gemm") flops += batch * flopsPerLaunch;
        if (periodUs > 0) {
            due += std::chrono::nanoseconds(static_cast<long long>(periodUs * 1e3));
            while (std::chrono::steady_clock::now() < due) {}
        }
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed - lastReport >= 1.0) {
            const double window = elapsed - lastReport;
            if (kind == "l2read") {
                must(cudaMemcpy(&windowBursts, dBursts, sizeof windowBursts, cudaMemcpyDeviceToHost), "read counter");
                must(cudaMemset(dBursts, 0, sizeof(unsigned long long)), "reset counter");
                const double wb = static_cast<double>(windowBursts) * burst * sizeof(float);
                bytes += wb;
                std::printf("  level:        setpoint %.2f achieved %.4f (L2 reads %.1f GB/s)\n", level,
                            fullRate > 0 ? wb / windowLaunches / fullRate : 0.0, wb / window / 1e9);
            }
            report(kind == "l2read" ? "saxpy" : kind, 0, launches, elapsed, bytes, flops);
            std::printf("  launches:     %.0f /s, %.1f us each\n", windowLaunches / window,
                        window / windowLaunches * 1e6);
            std::fflush(stdout);
            lastReport = elapsed;
            windowLaunches = 0;
        }
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n--- interferer '%s' final ---\n", kind.c_str());
    report(kind == "l2read" ? "saxpy" : kind, 0, launches, elapsed, bytes, flops);
    for (size_t k = 0; k < A.size(); ++k) cudaFree(A[k]), cudaFree(B[k]), cudaFree(C[k]);
    cudaFree(arr), cudaFree(sink), cudaFree(dBursts);
    if (handle) cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return launches > 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string kind = argc > 1 ? argv[1] : "mem_saxpy";
    unsigned mb = 256;
    int reps = 32;
    double level = 1.0;
    double periodUs = 0.0, duty = 1.0;
    double onoffMs = 0.0;       // saxpy ON/OFF: the period, in ms
    bool scramble = false;
    unsigned scrambleKb = 0;
    const int gemmN = 512;
    double kernelUs = 0.0;      // spin, l2read: the length of one kernel
    int side = 288, sets = 1;   // gemm
    unsigned kb = 1024;         // l2read: the array re-read through the L2
    unsigned burstArg = 16;     // saxpy --period-us: elements per thread between clock checks

    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--mb") && i + 1 < argc) {
            mb = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--level") && i + 1 < argc) {
            level = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--period-us") && i + 1 < argc) {
            periodUs = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--onoff-ms") && i + 1 < argc) {
            onoffMs = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--duty") && i + 1 < argc) {
            duty = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--kernel-us") && i + 1 < argc) {
            kernelUs = std::atof(argv[++i]);
        } else if (!std::strcmp(argv[i], "--side") && i + 1 < argc) {
            side = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--sets") && i + 1 < argc) {
            sets = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--kb") && i + 1 < argc) {
            kb = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        } else if (!std::strcmp(argv[i], "--burst") && i + 1 < argc) {
            burstArg = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
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
    if (kind == "spin" || kind == "gemm" || kind == "l2read") {
        // Here --period-us paces the launches; --duty, --scramble and --mb do not apply.
        const bool timed = kind != "gemm";
        if ((timed && kernelUs < 1.0) || (!timed && kernelUs != 0.0) || level < 0.0 || level > 1.0 ||
            (kind != "l2read" && level != 1.0) || side < 16 || sets < 1 || kb < 4 ||
            (periodUs > 0.0 && periodUs < kernelUs) || duty != 1.0 || scramble) {
            std::fprintf(stderr, "[itfproc] usage: spin --kernel-us K | gemm --side S --sets N | "
                                 "l2read --kernel-us K --level L [--kb KB], each with optional "
                                 "--period-us P >= K\n");
            return 2;
        }
        std::signal(SIGTERM, onSignal);
        std::signal(SIGINT, onSignal);
        return runPaced(kind, kernelUs, periodUs, side, sets, kb, level);
    }
    if (kind != "mem_saxpy" && kind != "compute_gemm" && kind != "saxpy") {
        std::fprintf(stderr, "[itfproc] kind must be mem_saxpy, compute_gemm, saxpy, spin, gemm or "
                             "l2read, not '%s'\n", kind.c_str());
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
    if (burstArg != 16 && (periodUs == 0.0 || burstArg < 1)) {
        std::fprintf(stderr, "[itfproc] --burst is for saxpy --period-us, at least 1\n");
        return 2;
    }
    if (onoffMs > 0.0 && (kind != "saxpy" || duty <= 0.0 || duty > 1.0 || level != 1.0 ||
                          periodUs > 0.0 || scramble)) {
        std::fprintf(stderr, "[itfproc] --onoff-ms is for saxpy, with --duty in (0, 1] and no --level, --period-us or --scramble\n");
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
        const bool steady = periodUs == 0.0 && !scramble;   // saxpyLevel; otherwise saxpyThrottled
        must(steady ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, saxpyLevel, 256, 0)
                    : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, saxpyThrottled, 256, 0),
             "occupancy");
        must(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0), "SM count");
        const unsigned blocks = static_cast<unsigned>(perSm * sms);
        const unsigned burst = burstArg;
        const double perThread = std::ceil(static_cast<double>(n) / (blocks * 256.0));
        const double burstsPerThread = std::ceil(perThread / burst);

        const unsigned long long periodNs = static_cast<unsigned long long>(periodUs * 1e3);
        const unsigned long long onNs = static_cast<unsigned long long>(periodUs * 1e3 * duty);
        const size_t lineMask = !scramble ? 0 : scrambleKb ? scrambleKb * 1024ull / 128 - 1 : n / 32 - 1;
        // One unit is one kernel over the whole vectors; its duration is the measurement.
        auto unitSeconds = [&](unsigned b, unsigned sleepNs, unsigned long long idleNs,
                               unsigned long long pNs = 0) {
            const auto a = std::chrono::steady_clock::now();
            if (steady)
                saxpyLevel<<<blocks, 256, 0, stream>>>(dx, dy, n, alpha, b, sleepNs, idleNs);
            else
                saxpyThrottled<<<blocks, 256, 0, stream>>>(dx, dy, n, alpha, b, sleepNs, idleNs, pNs, onNs, lineMask);
            must(cudaGetLastError(), "saxpy launch");
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
        std::printf("  throttle:     %u blocks x 256, burst %u, sleep %u ns, full-speed unit %.3f ms (%s kernel)\n",
                    blocks, b, sleepNs, tFull * 1e3, steady ? "steady" : "time-based/scrambled");
        if (periodNs)
            std::printf("  time-based:   period %.0f us, on %.0f us (duty %.2f), all threads together\n",
                        periodUs, periodUs * duty, duty);
        if (scramble) std::printf("  scramble:     %zu lines of 128 B in scrambled order within each window\n", lineMask + 1);

        // 3a. ON/OFF. During ON, whole full-speed units back to back; a unit is started only if it
        //     ends inside ON, so ON is a whole number of units, at most one unit shorter than
        //     D x P. During OFF the host sleeps after a synchronize, so no kernel is resident: the
        //     difference from the in-kernel throttles, which keep a kernel resident all the time.
        //     If the host wakes late, the schedule restarts from that moment instead of running ON
        //     phases back to back to catch up.
        if (onoffMs > 0.0) {
            using namespace std::chrono;
            using Tick = steady_clock::duration;
            const auto period = duration_cast<Tick>(duration<double, std::milli>(onoffMs));
            const auto on = duration_cast<Tick>(duration<double, std::milli>(onoffMs * duty));
            const auto unit = duration_cast<Tick>(duration<double>(tFull));
            if (on < unit) {
                std::fprintf(stderr, "[itfproc] the ON part (%.2f ms) is shorter than one unit (%.3f ms)\n",
                             onoffMs * duty, tFull * 1e3);
                return 2;
            }
            auto ns = [](steady_clock::time_point t) {
                return static_cast<long long>(duration_cast<nanoseconds>(t.time_since_epoch()).count());
            };
            std::printf("  on/off:       period %.1f ms, on %.2f ms (duty %.2f), whole units of %.3f ms\n",
                        onoffMs, onoffMs * duty, duty, tFull * 1e3);
            const auto t0 = steady_clock::now();
            auto start = t0;              // the start of the current period
            double onSeconds = 0.0;
            long periods = 0;
            while (!g_stop) {
                const auto a = steady_clock::now();
                auto e = a;
                while (!g_stop && steady_clock::now() + unit <= start + on) {
                    unitSeconds(burst, 0, 0);
                    ++units;
                    bytes += 3.0 * static_cast<double>(bytesPerBuffer);
                    e = steady_clock::now();
                }
                onSeconds += duration<double>(e - a).count();
                ++periods;
                std::printf("  phase:        on %lld %lld\n", ns(a), ns(e));
                start += period;
                if (start < steady_clock::now()) start = steady_clock::now();
                std::this_thread::sleep_until(start);
                const double elapsed = duration<double>(steady_clock::now() - t0).count();
                if (elapsed - lastReport >= 1.0) {
                    lastReport = elapsed;
                    report(kind, mb, units, elapsed, bytes, flops);
                    std::printf("  on/off:       achieved duty %.3f over %ld periods\n", onSeconds / elapsed, periods);
                }
                std::fflush(stdout);
            }
            const double elapsed = duration<double>(steady_clock::now() - t0).count();
            std::printf("\n--- interferer '%s' final ---\n", kind.c_str());
            report(kind, mb, units, elapsed, bytes, flops);
            std::printf("  on/off:       achieved duty %.3f over %ld periods\n", onSeconds / elapsed, periods);
            cudaFree(dx);
            cudaFree(dy);
            cublasDestroy(handle);
            cudaStreamDestroy(stream);
            return units > 0 ? 0 : 1;
        }

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
