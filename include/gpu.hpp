#pragma once
// =====================================================================
// gpu.hpp — the GPU layer.
//
// Every CUDA and TensorRT call in this program lives in gpu.cpp. This
// header exposes the types the other layers need:
//   - PinnedHostBuffer : page-locked host memory (for genuinely async
//                        copies) -- an interface, so other layers can
//                        own pinned buffers without calling CUDA
//   - SmAllocation     : the seam for spatial partitioning (which SMs?)
//   - TrtEngine        : the engine, its context, its device buffers
//
// Nothing here knows about images, datasets, statistics or files.
// See README.md for the methodology and the lessons learned.
// =====================================================================

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "config.hpp"

// ---------------------------------------------------------------------
// TensorRT logger: warnings and errors only.
//
// The API forces us to hand it a logger when creating the runtime. The
// informational spam is interesting exactly once and noise forever
// after.
// ---------------------------------------------------------------------
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// ---------------------------------------------------------------------
// PinnedHostBuffer — page-locked host memory.
//
// Why pinned: `cudaMemcpyAsync` is only genuinely asynchronous if the
// host side is page-locked. From ordinary (pageable) memory it
// silently degrades to a synchronous copy. Since the whole pipeline is
// built around async copies, the buffers the GPU reads from must be
// pinned.
//
// The header deliberately does not expose any CUDA type: the CUDA calls
// live in the .cpp, so the layers above do not include cuda_runtime.h.
// ---------------------------------------------------------------------
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;
    explicit PinnedHostBuffer(std::size_t floats);
    ~PinnedHostBuffer();

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

// ---------------------------------------------------------------------
// SmAllocation — WHICH SMs may this process run on?
//
// This is the seam for the spatial-partitioning work. It is a class
// rather than a flag because the mechanisms that answer the question
// hook into the CUDA lifecycle at different points, and each needs to
// hold its own state:
//
//   whole        nothing to do
//   greenctx     provisions its SMs on CONSTRUCTION, and returns a stream
//                bound to them from createStream()
//   mps          sets CUDA_MPS_SM_PARTITION on CONSTRUCTION, i.e. before
//                CUDA initialises
//   nvtaskset    patches the task descriptor's SM mask per launch, in
//                enqueue()
//
// A mechanism provisions whatever it needs when it is CONSTRUCTED, so an
// allocation is usable as soon as it exists: there is no separate init()
// step that a caller could forget, or run twice. Consequently construction
// must happen before any other CUDA call in the process -- main does it
// first, via makeSmAllocation().
//
// (A constructor that throws does not run the destructor, so an
// implementation must acquire its resources last and not throw afterwards.)
//
// Ordering rule for the caller: the allocation must outlive the stream it
// creates, so in main it is declared BEFORE the engine (destruction runs in
// reverse).
// ---------------------------------------------------------------------
class SmAllocation {
public:
    virtual ~SmAllocation() = default;

    virtual std::string name() const = 0;   // for logs and CSV metadata

    // When the execution stream is created. Default: a plain stream on the
    // primary context.
    virtual cudaStream_t createStream();

    // Around every kernel launch. Default: just launch it.
    virtual bool enqueue(nvinfer1::IExecutionContext& ctx, cudaStream_t stream);
};

// The baseline: the whole GPU, no partitioning.
class WholeGpu : public SmAllocation {
public:
    std::string name() const override { return "whole"; }
    // createStream() and enqueue() keep their defaults.
};

// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
// SmPartition — one pre-split group of SMs, ready to become a green context.
//
// It is a plain value: the resource specification plus the numbers needed to
// report what was asked for and what was actually assigned. Once
// cudaGreenCtxCreate has consumed it, the specification is spent — the green
// context owns the provisioned SMs.
// ---------------------------------------------------------------------
struct SmPartition {
    cudaDevResource resource{};   // the SMs assigned to this group
    int requestedSms = 0;         // what the plan asked for
    int grantedSms   = 0;         // what the split actually assigned
    int totalSms     = 0;         // device total, for reporting
    int percent      = 100;       // requested share, for reporting
};

// ---------------------------------------------------------------------
// PartitionPlan — decide who gets which SMs, ONCE, for the whole run.
//
// This class exists because of a measured trap: cudaDeviceGetDevResource
// always reports the FULL device, even when green contexts are already alive
// (verified on this board), and SM oversubscription is permitted. So N
// independent "split off my share" calls are NOT guaranteed to be disjoint —
// and the API exposes no SM identities to check afterwards.
//
// The documented pattern (and the one that is actually disjoint) is to split
// the device resource into N groups in a SINGLE cudaDevSmResourceSplit call
// and give one group to each green context. Doing it the other way — split off
// one group, then re-split the remainder — is explicitly discouraged: the
// remainder is unstructured and offers fewer guarantees.
//
// Percentages are of the device total; they must sum to at most 100%. Counts
// are rounded to the platform's alignment (2 SMs on this SoC) and the split
// reports what each group actually got, so the requested and granted values
// are both recorded.
// ---------------------------------------------------------------------
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

// ---------------------------------------------------------------------
// GreenContextAllocation — a green context provisioned with a group of SMs.
//
// The workflow, from the CUDA Programming Guide §4.6.4:
//   1. get the device's SM resource          PartitionPlan (jointly, once)
//   2. split into one group per network      PartitionPlan
//   3. finalize a resource descriptor        cudaDevResourceGenerateDesc
//   4. create the green context              cudaGreenCtxCreate
//   5. create streams on it                  cudaExecutionCtxStreamCreate
//
// Steps 1-2 belong to the plan, because they must be done jointly to be
// disjoint. This class owns steps 3-5 for ONE group. Everything launched on a
// stream created here can only use the group's SMs, and libraries (TensorRT
// included) respect that as long as they are handed the stream — so
// TrtEngine needs no changes at all.
//
// The driver's actual grant is queried back with cudaExecutionCtxGetDevResource
// and reported, so the results record what was really provisioned rather than
// what was asked for.
// ---------------------------------------------------------------------
class GreenContextAllocation : public SmAllocation {
public:
    explicit GreenContextAllocation(const SmPartition& part);
    ~GreenContextAllocation() override;

    // Not copyable: owns a CUDA resource.
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
};

// Build one allocation per network. This is a single call for the whole run
// because the SM sharing decision is global: for "greenctx" the device's SM
// resource is split ONCE into one group per percentage, so the partitions are
// disjoint by construction (see PartitionPlan for why a per-network split
// cannot guarantee that). For "whole" every network is unpartitioned.
//
// This must be the first CUDA-touching call in the process.
std::vector<std::unique_ptr<SmAllocation>> makeSmAllocations(
    const std::string& kind, const std::vector<int>& percents);

// ---------------------------------------------------------------------
// TrtEngine — the engine, its context, its device buffers.
//
// The engine is the compiled blueprint; the context is the instantiation
// that owns per-run state.
//
// Note the deliberate asymmetry between the two ends of the pipeline:
//   - the INPUT is the caller's data (an InputSet owns several tensors and
//     cycles through them), so it is passed to infer() per call;
//   - the OUTPUT is this object's product, a single tensor overwritten by
//     every inference, so the engine owns its host buffer.
// One end is an argument, the other is a result.
// ---------------------------------------------------------------------
class TrtEngine {
public:
    TrtEngine(const std::string& enginePath, SmAllocation& allocation);
    ~TrtEngine();

    // Owns CUDA resources: not copyable.
    TrtEngine(const TrtEngine&)            = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    std::int64_t inputVolume()  const noexcept { return inputVolume_; }
    std::int64_t outputVolume() const noexcept { return outputVolume_; }

    const std::string& inputName()  const noexcept { return inputName_; }
    const std::string& outputName() const noexcept { return outputName_; }

    // The raw output (logits) of the last inference.
    const float* outputHost() const noexcept { return hostOutput_.data(); }

    // One inference. `hostInput` is pinned host memory holding
    // inputVolume() floats, owned by the caller; the result lands in
    // outputHost(). Uploads, launches, downloads, synchronises.
    // Returns the pure GPU compute time in milliseconds — CUDA events
    // around the kernel launch only, so no copies and no CPU work.
    double infer(const float* hostInput);

private:
    static std::vector<char> readFile(const std::string& path);

    Logger        logger_;       // must outlive runtime_
    SmAllocation& allocation_;   // must outlive the stream it created
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string  inputName_, outputName_;
    std::int64_t inputVolume_  = 0;
    std::int64_t outputVolume_ = 0;

    // Device staging: both ends are the engine's own working memory.
    void* dInput_  = nullptr;
    void* dOutput_ = nullptr;

    // Host side: only the OUTPUT lives here. The input has no host buffer
    // in this class because it belongs to the caller (see infer()).
    PinnedHostBuffer hostOutput_;

    cudaStream_t stream_   = nullptr;
    cudaEvent_t  startEvt_ = nullptr;
    cudaEvent_t  stopEvt_  = nullptr;
};
