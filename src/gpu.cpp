//
// gpu.cpp — the GPU layer implementation. Every CUDA and TensorRT call in the
// program is in this file and nowhere else; the rest of the code talks to the
// interfaces in gpu.hpp and never includes cuda_runtime.h.
//

#include "gpu.hpp"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Turn a CUDA error into an exception, naming the call that failed. Most of
// the CUDA API returns a status rather than throwing, and an unchecked status
// here means a silent wrong answer rather than a crash.
void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
}

} // namespace

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

PinnedHostBuffer::PinnedHostBuffer(std::size_t floats)
    : data_(nullptr), floats_(floats)
{
    if (floats == 0) return;
    // cudaMallocHost is the page-locked allocator (cudaHostAlloc with default
    // flags is equivalent). It counts against a system-wide pinned limit, which
    // on this board is about 8 GB against the 18 MB actually needed.
    if (cudaMallocHost(&data_, floats * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMallocHost failed (" +
                                 std::to_string(floats * sizeof(float)) + " bytes)");
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (data_) cudaFreeHost(data_);
}

// Move semantics: steal the pointer, then null the source so its destructor
// has nothing to free. This is why the members cannot just be defaulted.
PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), floats_(other.floats_)
{
    other.data_   = nullptr;
    other.floats_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        if (data_) cudaFreeHost(data_);   // release what we were holding first
        data_         = other.data_;
        floats_       = other.floats_;
        other.data_   = nullptr;
        other.floats_ = 0;
    }
    return *this;
}

// The default hooks, used by the unpartitioned baseline and as the fallback for
// any mechanism that does not need to do anything special.
cudaStream_t SmAllocation::createStream() {
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess)
        throw std::runtime_error("cudaStreamCreate failed");
    return s;
}

bool SmAllocation::enqueue(nvinfer1::IExecutionContext& ctx, cudaStream_t stream) {
    return ctx.enqueueV3(stream);
}

std::vector<std::unique_ptr<SmAllocation>> makeSmAllocations(
    const std::string& kind, const std::vector<int>& percents)
{
    if (percents.empty())
        throw std::runtime_error("makeSmAllocations needs at least one network");

    if (kind == "whole") {
        // No partitioning: every network is simply on the GPU, and they
        // contend for it on their own. The percentages are ignored here.
        std::vector<std::unique_ptr<SmAllocation>> out;
        out.reserve(percents.size());
        for (std::size_t i = 0; i < percents.size(); ++i)
            out.push_back(std::make_unique<WholeGpu>());
        return out;
    }

    if (kind == "greenctx") {
        // ONE joint split for every network, so the groups are disjoint by
        // construction. The plan only has to outlive the loop below, since
        // each GreenContextAllocation copies what it needs as it is built.
        const PartitionPlan plan(percents);
        if (plan.size() > 1) {
            // With a single network the split is trivially "the whole device"
            // and needs no explaining; with several, the granted counts and any
            // capacity left unassigned are worth printing. Percentages summing
            // to less than 100 means SMs nobody may use, which is a mistake
            // rather than a choice, so it gets a warning.
            std::cout << "[info] partition plan: " << plan.summary() << "\n";
            if (plan.leftoverSms() > 0)
                std::cout << "[info] warning: " << plan.leftoverSms()
                          << " SMs are unused (the percentages sum to less than 100%)\n";
        }
        std::vector<std::unique_ptr<SmAllocation>> out;
        out.reserve(plan.size());
        for (std::size_t i = 0; i < plan.size(); ++i)
            out.push_back(std::make_unique<GreenContextAllocation>(plan[i]));
        return out;
    }

    throw std::runtime_error("unknown allocation: " + kind);
}

PartitionPlan::PartitionPlan(const std::vector<int>& percents) {
    if (percents.empty())
        throw std::runtime_error("PartitionPlan needs at least one partition");
    for (int p : percents)
        if (p < 1 || p > 100)
            throw std::runtime_error("each SM percentage must be between 1 and 100");

    const int device = 0;   // single-GPU board; the current device is 0 anyway

    // The guide recommends initialising the primary context before creating a
    // green context, so that the creation itself does not pay for it.
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    // The DEVICE resource. This always reports the WHOLE device, even when
    // green contexts are already alive (measured on this board) — which is
    // exactly why every group has to come out of one split, below.
    cudaDevResource full{};
    cudaCheck(cudaDeviceGetDevResource(device, &full, cudaDevResourceTypeSm),
              "cudaDeviceGetDevResource");
    totalSms_  = int(full.sm.smCount);
    // The alignment and the minimum partition size are properties of the SoC
    // (2 SMs each on this one). The fallbacks only guard against a driver that
    // leaves them unset.
    alignment_ = full.sm.smCoscheduledAlignment ? int(full.sm.smCoscheduledAlignment) : 2;
    minPart_   = full.sm.minSmPartitionSize      ? int(full.sm.minSmPartitionSize)      : 2;

    // Percentages -> SM counts. The params array must be zero initialised,
    // reserved fields included, which is what the vectors below do.
    std::vector<cudaDevSmResourceGroupParams> params(percents.size());
    long long requestedTotal = 0;
    for (std::size_t i = 0; i < percents.size(); ++i) {
        const unsigned align = static_cast<unsigned>(alignment_);
        // Integer arithmetic throughout: add half the divisor so the division
        // rounds to nearest rather than truncating, then round that up to the
        // alignment and clamp to what the device allows.
        const long long target =
            (static_cast<long long>(totalSms_) * percents[i] + 50) / 100;
        unsigned want = static_cast<unsigned>(target);
        want = ((want + align - 1) / align) * align;
        want = std::max(want, static_cast<unsigned>(minPart_));
        want = std::min(want, static_cast<unsigned>(totalSms_));

        params[i].smCount            = want;
        // The co-scheduled count is the grain the hardware schedules at; it has
        // to be given, and the alignment is the right value for it.
        params[i].coscheduledSmCount = align;
        requestedTotal += want;
    }
    // Checked against the clamped, aligned counts rather than the raw
    // percentages, so the error message describes what was actually asked for.
    if (requestedTotal > totalSms_)
        throw std::runtime_error("the requested SM percentages exceed the device's " +
                                 std::to_string(totalSms_) + " SMs");

    // ONE split for ALL groups. This is the line that makes the partitions
    // disjoint; splitting them one at a time would not.
    std::vector<cudaDevResource> groups(percents.size());
    cudaDevResource remainder{};
    cudaCheck(cudaDevSmResourceSplit(groups.data(), unsigned(percents.size()), &full,
                                     &remainder, 0, params.data()),
              "cudaDevSmResourceSplit");

    // Record both the request and the grant. The grant is what the driver
    // decided, and the two differ whenever rounding or alignment kicks in.
    parts_.resize(percents.size());
    int assigned = 0;
    for (std::size_t i = 0; i < percents.size(); ++i) {
        parts_[i].resource     = groups[i];
        parts_[i].requestedSms = int(params[i].smCount);
        parts_[i].grantedSms   = int(groups[i].sm.smCount);
        parts_[i].totalSms     = totalSms_;
        parts_[i].percent      = percents[i];
        assigned += parts_[i].grantedSms;
    }
    leftover_ = std::max(0, totalSms_ - assigned);
}

std::string PartitionPlan::summary() const {
    std::string s = std::to_string(totalSms_) + " SMs (alignment " +
                    std::to_string(alignment_) + "): ";
    for (std::size_t i = 0; i < parts_.size(); ++i) {
        if (i) s += ", ";
        s += std::to_string(parts_[i].grantedSms) + " (" +
             std::to_string(parts_[i].percent) + "%)";
        // Only mention the request when it differs, so the common case stays
        // readable and the discrepancy is what jumps out.
        if (parts_[i].grantedSms != parts_[i].requestedSms)
            s += " [asked " + std::to_string(parts_[i].requestedSms) + "]";
    }
    s += "; unused " + std::to_string(leftover_);
    return s;
}

// Construction provisions the green context from an already-split group, and
// has to be the first CUDA-touching call for that context in the process.
//
// Exception safety: a constructor that throws does NOT run the destructor, so
// the green context is created LAST and nothing after it may throw. That is
// why the readback below ignores its own error instead of checking it.
GreenContextAllocation::GreenContextAllocation(const SmPartition& part)
    : percent_(part.percent),
      requestedSms_(part.requestedSms),
      totalSms_(part.totalSms),
      grantedSms_(part.grantedSms)
{
    // Finalise the specification, then provision the SMs. The resource itself
    // came from PartitionPlan, which did the joint split.
    //
    // The copy is not redundant: the API takes a mutable pointer and may write
    // through it, and `part` is const.
    cudaDevResource group = part.resource;
    cudaDevResourceDesc_t desc{};
    cudaCheck(cudaDevResourceGenerateDesc(&desc, &group, 1),
              "cudaDevResourceGenerateDesc");
    cudaCheck(cudaGreenCtxCreate(&execCtx_, desc, /*device=*/0, 0), "cudaGreenCtxCreate");

    // Ask what we actually got. The driver may round or align, so the grant is
    // recorded rather than the request; if the query fails we keep the number
    // the split reported, which is the best information available.
    cudaDevResource got{};
    if (cudaExecutionCtxGetDevResource(execCtx_, &got, cudaDevResourceTypeSm) == cudaSuccess &&
        got.type == cudaDevResourceTypeSm)
        grantedSms_ = int(got.sm.smCount);

    std::cout << "[info] green context: " << percent_ << "% requested, "
              << grantedSms_ << " of " << totalSms_ << " SMs provisioned\n";
}

GreenContextAllocation::~GreenContextAllocation() {
    if (execCtx_) cudaExecutionCtxDestroy(execCtx_);
}

std::string GreenContextAllocation::name() const {
    return "greenctx " + std::to_string(grantedSms_) + "/" +
           std::to_string(totalSms_) + " SMs (" + std::to_string(percent_) + "%)";
}

cudaStream_t GreenContextAllocation::createStream() {
    // Only reachable if construction was skipped or failed, which would mean
    // the SMs were never provisioned. Better to say so than to launch on
    // whatever stream happens to exist.
    if (!execCtx_)
        throw std::runtime_error("green context is missing (was the allocation constructed?)");
    cudaStream_t s = nullptr;
    // cudaStreamNonBlocking is already the default for a green-context stream;
    // asking for it explicitly documents that the stream never synchronises
    // with the legacy default stream.
    cudaCheck(cudaExecutionCtxStreamCreate(&s, execCtx_, cudaStreamNonBlocking, 0),
              "cudaExecutionCtxStreamCreate");
    return s;
}

// Reading the engine file is how the engine loads itself, so it stays a private
// detail here rather than becoming a public file utility.
std::vector<char> TrtEngine::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open file: " + path);

    // The size is only discoverable by seeking to the end and back.
    file.seekg(0, std::ios::end);
    const std::size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), size);
    return data;
}

TrtEngine::TrtEngine(const std::string& enginePath, SmAllocation& allocation)
    : allocation_(allocation)
{
    // 1. Deserialize the engine. Engine files are hardware-specific: the
    //    kernels inside were compiled for THIS GPU and THIS TensorRT version.
    //    A mismatch returns nullptr rather than throwing, so check it —
    //    otherwise the next call segfaults with no message at all.
    const std::vector<char> engineData = readFile(enginePath);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine_) throw std::runtime_error("failed to deserialize engine: " + enginePath);

    // 2. Discover the I/O tensors. Ask the engine rather than hardcoding
    //    "input"/"output": engines can name their tensors anything, and some
    //    have more than one input. From TRT 8 on, everything is addressed by
    //    name.
    nvinfer1::Dims inputDims{}, outputDims{};
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::Dims dims = engine_->getTensorShape(name);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName_  = name;
            inputDims   = dims;
        } else {
            outputName_ = name;
            outputDims  = dims;
        }
    }

    // 3. Buffer sizes: the product of the dimensions, in elements.
    inputVolume_ = outputVolume_ = 1;
    for (int d = 0; d < inputDims.nbDims; ++d)  inputVolume_  *= inputDims.d[d];
    for (int d = 0; d < outputDims.nbDims; ++d) outputVolume_ *= outputDims.d[d];

    // Cheap insurance. The image constants in config.hpp must agree with the
    // shape baked into the engine, otherwise preprocess() writes the wrong
    // number of floats into the buffer and every result is silently garbage.
    // The input shape cannot be changed at run time, so this is the one place
    // the contract can be checked.
    const std::int64_t expected =
        std::int64_t(cfg::channels) * cfg::imageSize * cfg::imageSize;
    if (inputVolume_ != expected)
        throw std::runtime_error(
            "config.hpp image constants do not match the engine's input shape");

    // 4. Device buffers, plus a pinned host buffer for the output (pinned
    //    because of the async download in infer()).
    const std::size_t inputBytes  = std::size_t(inputVolume_)  * sizeof(float);
    const std::size_t outputBytes = std::size_t(outputVolume_) * sizeof(float);
    if (cudaMalloc(&dInput_,  inputBytes)  != cudaSuccess)
        throw std::runtime_error("cudaMalloc (input) failed");
    if (cudaMalloc(&dOutput_, outputBytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (output) failed");
    hostOutput_ = PinnedHostBuffer(std::size_t(outputVolume_));

    // 5. Context and tensor addresses. The addresses are set once here because
    //    the buffers never move, so enqueueV3 needs no per-call binding array.
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("failed to create execution context");
    context_->setTensorAddress(inputName_.c_str(),  dInput_);
    context_->setTensorAddress(outputName_.c_str(), dOutput_);

    // 6. The stream comes from the allocation — under a green context that is a
    //    stream bound to its SMs — plus the two timing events.
    stream_ = allocation_.createStream();
    if (cudaEventCreate(&startEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (start) failed");
    if (cudaEventCreate(&stopEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (stop) failed");
}

TrtEngine::~TrtEngine() {
    // The context goes first, because it refers to the buffers freed below.
    context_.reset();

    if (startEvt_) cudaEventDestroy(startEvt_);
    if (stopEvt_)  cudaEventDestroy(stopEvt_);
    if (stream_)   cudaStreamDestroy(stream_);
    if (dInput_)   cudaFree(dInput_);
    if (dOutput_)  cudaFree(dOutput_);
    // hostOutput_ (pinned), engine_ and runtime_ free themselves.
}

double TrtEngine::infer(const float* hostInput) {
    const std::size_t inputBytes  = std::size_t(inputVolume_)  * sizeof(float);
    const std::size_t outputBytes = std::size_t(outputVolume_) * sizeof(float);

    // The whole choreography on one stream, so the order is guaranteed by the
    // stream and no host-side waiting is needed in between.
    //
    // The two events bracket the KERNEL ONLY. That is the whole point of doing
    // it this way: the number returned is free of copy time and CPU work, so it
    // is the pure GPU compute time that partitioning actually moves. Putting
    // the events around the copies instead would fold the host round trip into
    // the measurement.
    //
    // Every call is checked. The async ones can only fail for an immediate
    // reason (a bad argument, an invalid pointer), which is why they still
    // return a status right here; an error inside the kernel or the copied data
    // surfaces later, at the synchronisation, and that is the status that
    // matters most.
    //
    // Worth doing even though this is the measured path: cudaCheck builds its
    // message only on failure, so the happy path costs one predictable branch
    // per call, and an unchecked CUDA failure here would look like a wrong
    // latency rather than like an error.
    cudaCheck(cudaMemcpyAsync(dInput_, hostInput, inputBytes,
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync (host to device)");
    cudaCheck(cudaEventRecord(startEvt_, stream_), "cudaEventRecord (start)");
    if (!allocation_.enqueue(*context_, stream_))     // the partitioning hook
        throw std::runtime_error("inference failed");
    cudaCheck(cudaEventRecord(stopEvt_, stream_), "cudaEventRecord (stop)");
    cudaCheck(cudaMemcpyAsync(hostOutput_.data(), dOutput_, outputBytes,
                              cudaMemcpyDeviceToHost, stream_),
              "cudaMemcpyAsync (device to host)");
    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvt_, stopEvt_);
    return double(ms);
}
