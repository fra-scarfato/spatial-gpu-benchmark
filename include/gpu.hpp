#pragma once
//
// gpu.hpp — the GPU layer.
//
// Every CUDA and TensorRT call in this program lives in gpu.cpp. This header
// exposes the three types the other layers need:
//
//   PinnedHostBuffer   page-locked host memory, so that the async copies are
//                      actually async. An interface, so code above can own
//                      pinned memory without seeing a CUDA type.
//   SmAllocation       the seam for spatial partitioning: which SMs may this
//                      process use?
//   TrtEngine          the engine, its context, its buffers.
//
// Nothing here knows about images, datasets, statistics or files. See
// README.md for the methodology and the reasoning behind it.
//

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "config.hpp"

// TensorRT insists on a logger when the runtime is created. Warnings and
// errors only: the informational spam is interesting exactly once and noise
// forever after.
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// Page-locked (pinned) host memory.
//
// Why pinned at all: cudaMemcpyAsync is only genuinely asynchronous when the
// host side is page-locked. From ordinary pageable memory it silently degrades
// into a synchronous copy, which would put the host round trip inside the
// measurement. Measured cost of getting this wrong: 0.0419 ms per iteration
// pinned against 0.0963 ms pageable.
//
// No CUDA type appears in this interface, so the layers above do not have to
// include cuda_runtime.h.
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    explicit PinnedHostBuffer(std::size_t floats);
    ~PinnedHostBuffer();

    // Hand-written move operations rather than the defaults, because moving a
    // raw pointer with the default operations would leave two owners for one
    // cudaMallocHost block and free it twice.
    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept;
    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept;

    float*       data()       noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    std::size_t  floats()     const noexcept { return floats_; }
    bool         empty()      const noexcept { return data_ == nullptr; }

private:
    float*      data_   = nullptr;
    std::size_t floats_ = 0;
};

// WHICH SMs may this process run on?
//
// This is the seam for the spatial-partitioning work. It is a class rather
// than a flag because each mechanism hooks into the CUDA lifecycle at a
// different point and has to hold its own state:
//
//   whole      nothing to do.
//   greenctx   implemented: provisions its SMs on CONSTRUCTION and returns a
//              stream bound to them from createStream().
//   mps        planned: sets CUDA_MPS_SM_PARTITION on CONSTRUCTION, because
//              the variable has to be in place before CUDA initialises.
//   nvtaskset  planned: patches the task descriptor's SM mask per launch,
//              i.e. in enqueue().
//
// A mechanism provisions whatever it needs when it is CONSTRUCTED, so an
// allocation is usable as soon as it exists. There is deliberately no separate
// init() step: a caller cannot forget it, and cannot run it twice. That is not
// hypothetical — an early version had an init() and called it twice in the
// same process, which quietly created a second green context and leaked the
// first.
//
// Two consequences follow, and both matter:
//   - construction must be the first CUDA-touching call in the process, which
//     is why main() builds the allocations before anything else;
//   - a constructor that throws does NOT run its destructor, so an
//     implementation has to acquire its resources last and not throw after.
//
// Ordering rule for callers: the allocation has to outlive the stream it
// creates, so where both live in one object the allocation must be declared
// first (destruction then runs in the reverse order).
class SmAllocation {
public:
    virtual ~SmAllocation() = default;

    virtual std::string name() const = 0;   // for logs and CSV metadata

    // Called once, when the execution stream is created. The default is a
    // plain stream on the primary context.
    virtual cudaStream_t createStream();

    // Called around every kernel launch. The default just launches it.
    virtual bool enqueue(nvinfer1::IExecutionContext& ctx, cudaStream_t stream);
};

// The baseline: the whole GPU, nothing partitioned. Both hooks keep their
// default behaviour, so this class is only a name.
class WholeGpu : public SmAllocation {
public:
    std::string name() const override { return "whole"; }
};

// One pre-split group of SMs, ready to become a green context.
//
// A plain value: the resource specification plus the numbers needed to report
// what was asked for and what was actually assigned. Once cudaGreenCtxCreate
// has consumed it the specification is spent, because from then on the green
// context owns those SMs.
struct SmPartition {
    cudaDevResource resource{};   // the SMs assigned to this group
    int requestedSms = 0;         // what the plan asked for
    int grantedSms   = 0;         // what the split actually assigned
    int totalSms     = 0;         // device total, for reporting
    int percent      = 100;       // requested share, for reporting
};

// Decide who gets which SMs, ONCE, for the whole run.
//
// This class exists because of a measured trap. cudaDeviceGetDevResource
// reports the FULL device every time, even while green contexts are alive
// (verified on this board), and SM oversubscription is explicitly permitted.
// So N independent "split off my share" calls are NOT guaranteed to be
// disjoint, and the API exposes no SM identities to check afterwards.
//
// The pattern that is actually disjoint is to split the device resource into N
// groups in a SINGLE cudaDevSmResourceSplit call and give one group to each
// green context. Doing it the other way — split off one group, then re-split
// the remainder — is discouraged by the guide, because the remainder is
// unstructured and comes with fewer guarantees.
//
// Percentages are of the device total and must not sum to more than 100.
// Counts are rounded up to the platform's alignment (2 SMs on this SoC), and
// both the requested and the granted count are kept, because the driver may
// hand back something different from what was asked.
class PartitionPlan {
public:
    explicit PartitionPlan(const std::vector<int>& percents);

    std::size_t        size()     const noexcept { return parts_.size(); }
    const SmPartition& operator[](std::size_t i) const { return parts_[i]; }

    int totalSms()    const noexcept { return totalSms_; }
    int leftoverSms() const noexcept { return leftover_; }   // assigned to nobody
    int alignment()   const noexcept { return alignment_; }

    std::string summary() const;   // e.g. "16 SMs (alignment 2): 12 (75%), 4 (25%); unused 0"

private:
    std::vector<SmPartition> parts_;
    int totalSms_  = 0;
    int leftover_  = 0;
    int alignment_ = 2;
    int minPart_   = 2;
};

// The work queue half of a green context's resource descriptor.
//
// A green context can be provisioned with work queues as well as SMs, and the
// guide's warning is precise: if independent work — kernels submitted on
// different CUDA streams — maps to the SAME work queue, a false dependence
// appears and the work is serialised even when SMs are free. The scope chooses
// how the driver shares queues between contexts; the limit declares how many
// concurrent stream-ordered workloads we expect, which the driver consults as a
// hint when assigning them.
//
// `set == false` is the important case: the descriptor then carries the SM group
// only, exactly as every run up to and including Experiment 2 did. That keeps
// those results comparable and makes the new arms differ from them in one
// documented way rather than in whatever the driver felt like doing.
struct WqConfig {
    bool                        set   = false;
    cudaDevWorkqueueConfigScope scope = cudaDevWorkqueueConfigScopeDeviceCtx;
    unsigned                    limit = 0;
};

// A green context provisioned with one group of SMs.
//
// The workflow, from the CUDA Programming Guide §4.6.4:
//   1. get the device's SM resource          cudaDeviceGetDevResource
//   2. split it into one group per network   cudaDevSmResourceSplit
//   3. finalise a resource descriptor     
//   4. create the green context              cudaGreenCtxCreate
//   5. create streams on it                  cudaExecutionCtxStreamCreate
//
// Steps 1 and 2 are drawn from PartitionPlan because they have to be done
// jointly to be disjoint; this class owns steps 3-5 for a single group.
//
// Everything launched on a stream created here can only use that group's SMs.
// Libraries respect this as long as they are handed the stream — which is why
// TrtEngine needed no changes at all to work under green contexts.
//
// The driver's actual grant is queried back with
// cudaExecutionCtxGetDevResource and reported, so the results record what was
// really provisioned rather than what was requested.
class GreenContextAllocation : public SmAllocation {
public:
    explicit GreenContextAllocation(const SmPartition& part, const WqConfig& wq);
    ~GreenContextAllocation() override;

    // Not copyable: it owns a CUDA resource.
    GreenContextAllocation(const GreenContextAllocation&)            = delete;
    GreenContextAllocation& operator=(const GreenContextAllocation&) = delete;

    std::string  name() const override;        // e.g. "greenctx 12/16 SMs (75%)"
    cudaStream_t createStream() override;      // a stream bound to the green context

    int grantedSms() const noexcept { return grantedSms_; }

private:
    int                    percent_      = 100;
    int                    requestedSms_ = 0;
    int                    totalSms_     = 0;
    int                    grantedSms_   = 0;
    cudaExecutionContext_t execCtx_      = nullptr;
    WqConfig               wq_;          // as requested; the grant is logged only
};


// Build one allocation per network.
//
// A single call for the whole run, because the decision is global: for
// "greenctx" the device's SM resource is split ONCE into one group per
// percentage, which is what makes the partitions disjoint (see PartitionPlan
// for why a per-network split cannot promise that). For "whole" every network
// is unpartitioned.
//
// This must be the first CUDA-touching call in the process.
std::vector<std::unique_ptr<SmAllocation>> makeSmAllocations(
    const std::string& kind, const std::vector<int>& percents, const WqConfig& wq);

// The engine, its context, and its device buffers.
//
// The engine is the compiled blueprint; the context is the instantiation that
// owns the per-run state.
//
// Note the deliberate asymmetry between the two ends of the pipeline:
//   - the INPUT is the caller's data (an InputSet owns several tensors and
//     cycles through them), so it is passed to infer() on every call;
//   - the OUTPUT is this object's product, a single tensor overwritten by every
//     inference, so the engine owns its host buffer.
// One end is an argument, the other is a result.
class TrtEngine {
public:
    TrtEngine(const std::string& enginePath, SmAllocation& allocation);
    ~TrtEngine();

    // Owns CUDA resources: not copyable.
    TrtEngine(const TrtEngine&)            = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    std::int64_t inputVolume()  const noexcept { return inputVolume_; }

    // The spatial geometry of the input, read from the engine at construction.
    // The preprocessing needs it and must not assume 224, or a square: a network
    // with any input size is a valid subject, and the contract that matters is
    // the shape baked into the engine. Validated there, so by the time anyone
    // asks, the input is known to be three-channel with two spatial dimensions.
    int inputHeight() const noexcept { return inputHeight_; }
    int inputWidth()  const noexcept { return inputWidth_; }

    const std::string& inputName()  const noexcept { return inputName_; }

    // The raw output (logits) of the last inference.
    // The engine's outputs. Most networks here have one, RetinaNet has ten (the
    // FPN's per-level class and box tensors), so the interface is an index rather
    // than a single tensor.
    std::size_t outputCount() const noexcept { return outputNames_.size(); }
    const float* outputHost(std::size_t i) const noexcept { return hostOutputs_[i].data(); }

    // One inference.
    //
    // `hostInput` points at pinned host memory holding exactly inputVolume()
    // floats and is owned by the caller — the size contract is checked once,
    // at construction, against the shape baked into the engine, so it is not
    // re-checked here. The result lands in outputHost().
    //
    // Uploads, launches, downloads, synchronises. Returns the pure GPU compute
    // time in milliseconds: CUDA events around the kernel launch only, so no
    // copies and no CPU work are included.
    void   submit(const float* hostInput);   // start it, do not wait
    double finish();                         // wait for it, return its GPU time
    double wallMs();                         // H2D + kernel + D2H, from events on
                                             // this network's stream: per network
    bool   ready();                          // has the stream drained? (no blocking)
    double infer(const float* hostInput);    // submit() + finish(): the usual path

private:
    static std::vector<char> readFile(const std::string& path);

    Logger        logger_;       // must outlive runtime_
    SmAllocation& allocation_;   // must outlive the stream it created
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string  inputName_;
    std::vector<std::string>  outputNames_;    // every output, in engine order
    std::int64_t inputVolume_  = 0;
    int          inputHeight_  = 0;   // spatial geometry, from the engine
    int          inputWidth_   = 0;
    std::int64_t outputVolume_ = 0;

    // Device staging: both ends are the engine's own working memory.
    void* dInput_  = nullptr;
    std::vector<std::int64_t> outputVolumes_;  // elements per output
    std::vector<void*>        dOutputs_;       // device buffer per output

    // Host side: only the OUTPUT lives here. The input has no host buffer in
    // this class because it belongs to the caller (see infer()).
    std::vector<PinnedHostBuffer> hostOutputs_;

    cudaStream_t stream_   = nullptr;
    cudaEvent_t  startEvt_ = nullptr;
    cudaEvent_t  stopEvt_  = nullptr;
    cudaEvent_t  spanStartEvt_ = nullptr;   // the whole step, copies included
    cudaEvent_t  spanStopEvt_  = nullptr;
};
