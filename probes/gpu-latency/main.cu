// RRZE-HPC gpu-benches, gpu-latency/main.cu, patched for use as a stress reporter.
// Upstream and the exact changes: see UPSTREAM in this directory. The pchase kernel and the
// construction of the chain are unchanged; main() measures fixed sizes for a fixed duration.
#include "dtime.hpp"
#include "gpu-error.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>

using namespace std;

typedef int64_t dtype;

__device__ unsigned int smid() {
  unsigned int r;

  asm("mov.u32 %0, %%smid;" : "=r"(r));

  return r;
}

template <typename T>
__global__ void pchase(T *buf, T *__restrict__ dummy_buf, int64_t N) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t *idx = buf;

  const int unroll_factor = 32;
#pragma unroll 1
  for (int64_t n = 0; n < N; n += unroll_factor) {
#pragma unroll
    for (int u = 0; u < unroll_factor; u++) {
      idx = (int64_t *)*idx;
    }
  }

  if (tidx > 12313) {
    dummy_buf[0] = (int64_t)idx;
  }
}

// One chain over LEN cache lines, built exactly as upstream builds it for each sample.
struct Chain {
  int64_t LEN;
  int64_t iters;
  int64_t *buf = NULL;
  int64_t *dbuf = NULL;
  dtype *dummy_buf = NULL;
};

static Chain buildChain(int64_t LEN, int64_t iters, std::mt19937 &g) {
  const int cl_size = 128 / sizeof(int64_t);
  const int skip_factor = 1;
  Chain c;
  c.LEN = LEN;
  c.iters = iters > 0 ? iters : max(LEN, (int64_t)100000);

  vector<int64_t> order(LEN);
  GPU_ERROR(cudaMallocManaged(&c.buf, skip_factor * cl_size * LEN * sizeof(dtype)));
  GPU_ERROR(cudaMalloc(&c.dbuf, skip_factor * cl_size * LEN * sizeof(dtype)));
  GPU_ERROR(cudaMallocManaged(&c.dummy_buf, sizeof(dtype)));
  for (int64_t i = 0; i < LEN; i++) {
    order[i] = i + 1;
  }
  order[LEN - 1] = 0;

  shuffle(begin(order), end(order) - 1, g);

  for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
    dtype idx = 0;
    for (int64_t i = 0; i < LEN; i++) {

      c.buf[(idx * cl_size + cl_lane) * skip_factor] =
          skip_factor * (order[i] * cl_size + cl_lane + (order[i] == 0 ? 1 : 0));
      idx = order[i];
    }
  }
  c.buf[skip_factor * (order[LEN - 2] * cl_size + cl_size - 1)] = 0;

  for (int64_t n = 0; n < LEN * cl_size * skip_factor; n++) {
    c.buf[n] = (int64_t)c.dbuf + c.buf[n] * sizeof(int64_t *);
  }

  GPU_ERROR(cudaMemcpy(c.dbuf, c.buf, skip_factor * cl_size * LEN * sizeof(dtype),
                       cudaMemcpyHostToDevice));

  pchase<dtype><<<1, 1>>>(c.buf, c.dummy_buf, c.iters);
  pchase<dtype><<<1, 1>>>(c.buf, c.dummy_buf, c.iters);
  GPU_ERROR(cudaDeviceSynchronize());
  return c;
}

//   gpu-latency --sizes-kb 2048,65536 --duration 30 [--iters N]
// One line per sample: buffer size, dereferences, elapsed time, nanoseconds per dereference.
// The sizes are sampled in turn, so every size covers the same stretch of time. --iters sets the
// dereferences per sample (a multiple of 32, the kernel's unroll); without it, upstream's
// max(LEN, 100000), which averages each sample over about 0.3 s at DRAM latency.
// Added for the reporter: the same chase as pchase, timed from inside in windows of W dereferences,
// so that one long launch yields K window latencies with no per-window launch or event overhead.
// ticks[0] marks the start and ticks[w] the end of window w, read with clock64 (SM cycles; the
// clock is pinned). Before each mark the thread branches on the last pointer it loaded, which it
// can only do once that load has completed, so a mark is never read ahead of the access it closes.
template <typename T>
__global__ void pchase_windows(T *buf, T *__restrict__ dummy_buf, int64_t W, int64_t K,
                               long long *ticks) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t *idx = buf;
  const int unroll_factor = 32;
  for (int64_t w = 0;; w++) {
    if ((uintptr_t)idx == 1)
      asm volatile("trap;");
    ticks[w] = clock64();
    if (w == K)
      break;
#pragma unroll 1
    for (int64_t n = 0; n < W; n += unroll_factor) {
#pragma unroll
      for (int u = 0; u < unroll_factor; u++) {
        idx = (int64_t *)*idx;
      }
    }
  }
  if (tidx > 12313) {
    dummy_buf[0] = (int64_t)idx;
  }
}

int main(int argc, char **argv) {
  vector<int64_t> sizesKB = {2048, 65536};
  double duration = 30;
  int64_t iters = 0;
  int64_t window = 0;            // > 0: in-kernel windows of this many dereferences
  int64_t windows = 2000;        // windows per launch
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--sizes-kb") && i + 1 < argc) {
      sizesKB.clear();
      for (char *t = strtok(argv[++i], ","); t; t = strtok(NULL, ","))
        sizesKB.push_back(atoll(t));
    } else if (!strcmp(argv[i], "--duration") && i + 1 < argc) {
      duration = atof(argv[++i]);
    } else if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
      iters = atoll(argv[++i]);
    } else if (!strcmp(argv[i], "--window") && i + 1 < argc) {
      window = atoll(argv[++i]);
    } else if (!strcmp(argv[i], "--windows") && i + 1 < argc) {
      windows = atoll(argv[++i]);
    } else {
      cerr << "usage: " << argv[0]
           << " [--sizes-kb 2048,65536] [--duration 30] [--iters N] [--window W [--windows K]]\n";
      return 2;
    }
  }

#ifdef __NVCC__
  GPU_ERROR(cudaFuncSetAttribute(
      pchase<dtype>, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
#endif

  std::random_device rd;
  std::mt19937 g(rd());

  vector<Chain> chains;
  for (int64_t kb : sizesKB)
    chains.push_back(buildChain(kb * 1024 / 128, iters, g));   // 128-byte cache lines

  cudaEvent_t start, stop;
  GPU_ERROR(cudaEventCreate(&start));
  GPU_ERROR(cudaEventCreate(&stop));

  if (window > 0) {
    // One line per window: size, dereferences, SM cycles per dereference, ns per dereference. The
    // first window of a launch includes the kernel's start and is dropped.
    if (window % 32) {
      cerr << "--window must be a multiple of 32, the kernel's unroll\n";
      return 2;
    }
    long long *ticks = NULL;
    GPU_ERROR(cudaMallocManaged(&ticks, (windows + 1) * sizeof(long long)));
    const double ghz = 1.3005;   // the pinned SM clock; the guard refuses to measure otherwise
    cout << "# size_kb window cycles_per_access ns_per_access\n" << flush;
    const double t0 = dtime();
    while (dtime() - t0 < duration) {
      for (Chain &c : chains) {
        pchase_windows<dtype><<<1, 1>>>(c.buf, c.dummy_buf, window, windows, ticks);
        GPU_ERROR(cudaDeviceSynchronize());
        GPU_ERROR(cudaGetLastError());
        for (int64_t w = 2; w <= windows; w++) {
          const double cyc = double(ticks[w] - ticks[w - 1]) / window;
          cout << c.LEN * 128 / 1024 << " " << window << " " << fixed << setprecision(2) << cyc
               << " " << cyc / ghz << "\n";
        }
        cout << flush;
      }
    }
    GPU_ERROR(cudaFree(ticks));
    return 0;
  }

  cout << "# size_kb iters ms ns_per_access\n" << flush;
  const double t0 = dtime();
  while (dtime() - t0 < duration) {
    for (Chain &c : chains) {
      GPU_ERROR(cudaEventRecord(start));
      pchase<dtype><<<1, 1>>>(c.buf, c.dummy_buf, c.iters);
      GPU_ERROR(cudaEventRecord(stop));
      GPU_ERROR(cudaEventSynchronize(stop));
      float milliseconds = 0;
      GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
      GPU_ERROR(cudaGetLastError());
      cout << c.LEN * 128 / 1024 << " " << c.iters << " " << fixed << setprecision(4)
           << milliseconds << " " << setprecision(2)
           << milliseconds * 1e6 / c.iters << "\n";
    }
    cout << flush;
  }
  for (Chain &c : chains) {
    GPU_ERROR(cudaFree(c.buf));
    GPU_ERROR(cudaFree(c.dbuf));
    GPU_ERROR(cudaFree(c.dummy_buf));
  }
  return 0;
}
