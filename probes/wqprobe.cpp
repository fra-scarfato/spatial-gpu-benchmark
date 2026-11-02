// wqprobe — Phase A reconnaissance for the green-context work queue experiment.
//
// Answers three questions the experiment design depends on:
//   1. what the device reports by default for its workqueue configuration
//      (wqConcurrencyLimit + sharingScope), and how that tracks
//      CUDA_DEVICE_MAX_CONNECTIONS;
//   2. whether a green context created with an explicit workqueue configuration
//      can be READ BACK afterwards, and what it reports as granted;
//   3. whether asking for DeviceCtx versus GreenCtxBalanced is observable at all
//      — if the two read back identically, the whole experiment is void.
//
// Build: make -C probes wqprobe

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

static const char* scopeName(cudaDevWorkqueueConfigScope s) {
    switch (s) {
        case cudaDevWorkqueueConfigScopeDeviceCtx:         return "DeviceCtx";
        case cudaDevWorkqueueConfigScopeGreenCtxBalanced:  return "GreenCtxBalanced";
        default:                                           return "(unrecognised)";
    }
}

static void says(const char* what, cudaError_t e) {
    printf("  %-46s -> %s\n", what, cudaGetErrorString(e));
}

int main() {
    printf("CUDART_VERSION=%d\n", CUDART_VERSION);
    const char* env = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
    printf("CUDA_DEVICE_MAX_CONNECTIONS=%s\n\n", env ? env : "(unset)");

    int dev = 0;
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) { says("cudaSetDevice", e); return 1; }

    // ---- 1. device defaults ------------------------------------------------
    printf("--- device defaults (before any green context) ---\n");
    cudaDevResource wq{};
    e = cudaDeviceGetDevResource(dev, &wq, cudaDevResourceTypeWorkqueueConfig);
    if (e != cudaSuccess) { says("cudaDeviceGetDevResource(WQConfig)", e); return 1; }
    printf("  type=%d  device=%d  wqConcurrencyLimit=%u  sharingScope=%s (%d)\n",
           (int)wq.type, wq.wqConfig.device, wq.wqConfig.wqConcurrencyLimit,
           scopeName(wq.wqConfig.sharingScope), (int)wq.wqConfig.sharingScope);

    cudaDevResource sm{};
    e = cudaDeviceGetDevResource(dev, &sm, cudaDevResourceTypeSm);
    if (e == cudaSuccess)
        printf("  SM resource: type=%d  minSmPartitionSize=%d  smCoscheduledAlignment=%d\n",
               (int)sm.type, sm.sm.minSmPartitionSize, sm.sm.smCoscheduledAlignment);
    else
        says("cudaDeviceGetDevResource(Sm)", e);

    cudaDevResource pre{};
    e = cudaDeviceGetDevResource(dev, &pre, cudaDevResourceTypeWorkqueue);
    printf("  pre-existing WQ resource                 -> ");
    if (e == cudaSuccess) printf("type=%d (queryable)\n", (int)pre.type);
    else                  printf("%s\n", cudaGetErrorString(e));

    // ---- 2. create a green context per scope, then read back what was granted
    printf("\n--- green contexts: request vs granted ---\n");
    const unsigned requested = wq.wqConfig.wqConcurrencyLimit;

    struct Arm { const char* label; cudaDevWorkqueueConfigScope scope; unsigned limit; };
    const Arm arms[] = {
        {"devctx (today)",  cudaDevWorkqueueConfigScopeDeviceCtx,        requested},
        {"balanced",        cudaDevWorkqueueConfigScopeGreenCtxBalanced, requested},
        {"balanced limit=1",cudaDevWorkqueueConfigScopeGreenCtxBalanced, 1},
    };

    for (const Arm& a : arms) {
        // The descriptor must carry an SM resource and the workqueue config.
        // One workqueue config resource at most, per the API docs.
        cudaDevResource res[2] = {{}, {}};
        res[0] = sm;
        res[1].type                        = cudaDevResourceTypeWorkqueueConfig;
        res[1].wqConfig.device             = dev;
        res[1].wqConfig.sharingScope       = a.scope;
        res[1].wqConfig.wqConcurrencyLimit = a.limit;

        cudaDevResourceDesc_t desc{};
        e = cudaDevResourceGenerateDesc(&desc, res, 2);
        if (e != cudaSuccess) { says("cudaDevResourceGenerateDesc", e); continue; }

        cudaExecutionContext_t ctx = nullptr;
        e = cudaGreenCtxCreate(&ctx, desc, dev, 0);
        if (e != cudaSuccess) { says("cudaGreenCtxCreate", e); continue; }

        cudaDevResource gotWq{}, gotSm{};
        e = cudaExecutionCtxGetDevResource(ctx, &gotWq, cudaDevResourceTypeWorkqueueConfig);
        const bool gotWqOk = (e == cudaSuccess);
        const cudaError_t e2 = cudaExecutionCtxGetDevResource(ctx, &gotSm, cudaDevResourceTypeSm);
        const bool gotSmOk = (e2 == cudaSuccess);

        printf("  %-16s requested limit=%-4u scope=%-15s | granted ", a.label,
               a.limit, scopeName(a.scope));
        if (gotWqOk)
            printf("limit=%-4u scope=%-15s", gotWq.wqConfig.wqConcurrencyLimit,
                   scopeName(gotWq.wqConfig.sharingScope));
        else
            printf("WQ query failed (%s)        ", cudaGetErrorString(e));
        if (gotSmOk) printf(" | SMs=%d", gotSm.sm.smCount);
        printf("\n");

        cudaExecutionCtxDestroy(ctx);
    }

    printf("\n");
    return 0;
}
