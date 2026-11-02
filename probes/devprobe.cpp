// devprobe.cpp — the device facts the memory-interferer experiments are built on.
//
// Two numbers: how big L2 is, and how much bandwidth a pure copy actually reaches.
// The size sweep is the point of it. A copy that fits in L2 runs far faster than one
// that does not, so the sweep locates the boundary empirically instead of trusting a
// device property, and it says how large a memory-bound interferer's working set has
// to be before its traffic leaves the chip at all.
//
// CUDA runtime only, so it builds with g++ exactly like the runner does. No nvcc.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CK(call)                                                                \
    do {                                                                        \
        cudaError_t e_ = (call);                                                \
        if (e_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s -> %s\n", #call, cudaGetErrorString(e_));  \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Copy `mb` megabytes device-to-device and report GB/s moved. A copy reads the
// source and writes the destination, so both count.
static double copyGBs(std::size_t mb) {
    const std::size_t bytes = mb * 1024ull * 1024ull;
    float* a = nullptr;
    float* b = nullptr;
    CK(cudaMalloc(&a, bytes));
    CK(cudaMalloc(&b, bytes));
    CK(cudaMemset(a, 1, bytes));

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    for (int i = 0; i < 3; ++i) CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());

    // Small working sets need many repetitions to time meaningfully, large ones
    // need few, or the probe takes longer than the experiment it sizes.
    const int iters = (mb >= 256) ? 10 : 200;

    CK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) CK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));

    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, t0, t1));
    const double gbs = (2.0 * bytes * iters) / (ms / 1e3) / 1e9;

    CK(cudaEventDestroy(t0));
    CK(cudaEventDestroy(t1));
    CK(cudaFree(a));
    CK(cudaFree(b));
    return gbs;
}

int main() {
    cudaDeviceProp p {};
    CK(cudaGetDeviceProperties(&p, 0));

    // Read through the attribute API rather than the struct fields: CUDA 13 removed
    // memoryClockRate and memoryBusWidth from cudaDeviceProp, and an attribute that
    // no longer exists comes back as an error instead of failing to compile.
    int sms = 0, l2 = 0, memClock = 0, busWidth = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const bool haveClock = cudaDeviceGetAttribute(&memClock, cudaDevAttrMemoryClockRate, 0) == cudaSuccess;
    const bool haveBus   = cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, 0) == cudaSuccess;

    std::printf("device             %s\n", p.name);
    std::printf("SMs                %d  (SM partitions are a multiple of 2)\n", sms);
    std::printf("L2                 %.1f MB\n", l2 / 1048576.0);
    // These two attributes are reported on Tegra but they do NOT describe the memory
    // clock: cudaDevAttrMemoryClockRate returns the GPU clock (1300 MHz here), so a
    // bandwidth derived from them comes out at 83 GB/s while a copy measures 190.
    // Printed for the record, explicitly not as a baseline.
    if (haveClock && haveBus && memClock > 0 && busWidth > 0)
        std::printf("attributes        %d kHz x %d bits  (the GPU clock, not the memory\n"
                    "                                     clock: do not derive bandwidth)\n",
                    memClock, busWidth);
    std::printf("\n");

    std::printf("D2D copy bandwidth by working set:\n");
    for (std::size_t mb : {2ull, 4ull, 32ull, 256ull, 1024ull}) {
        const double gbs = copyGBs(mb);
        // Two buffers are live at once, so the working set is twice the argument.
        const bool fits = (2 * mb * 1024ull * 1024ull <= static_cast<std::size_t>(l2));
        std::printf("  %5zu MB  %7.1f GB/s%s\n", mb, gbs, fits ? "   (both buffers fit L2)" : "");
    }
    return 0;
}
