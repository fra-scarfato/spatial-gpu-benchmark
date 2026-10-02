// The spin participant's kernel: a dependent chain of FMAs in registers.
//
// It touches no memory at all - the one store is behind a condition that is never true and is
// there only so the compiler cannot delete the loop - so a co-runner made of it is resident on
// its SMs and asks nothing of the L1, the L2, the DRAM or the launch path. It is the control for
// the question the traces leave: whether a co-runner costs the target anything at its kernel
// boundaries merely by having a kernel on the device.
#include <cuda_runtime.h>

__global__ static void spinKernel(float* sink, long long iters) {
    float a = threadIdx.x * 1e-3f;
    for (long long i = 0; i < iters; ++i) a = fmaf(a, 0.999f, 1e-3f);
    if (a == -1.0f) *sink = a;
}

cudaError_t launchSpin(cudaStream_t stream, int blocks, long long iters) {
    spinKernel<<<blocks, 256, 0, stream>>>(nullptr, iters);
    return cudaGetLastError();
}
