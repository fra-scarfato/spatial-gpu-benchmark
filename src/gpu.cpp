// =====================================================================
// gpu.cpp — the GPU layer implementation.
// =====================================================================

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

// Turn a CUDA error into an exception, with the failing call named.
void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
}

} // namespace

// ---------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cerr << "[TRT] " << msg << std::endl;
}

// ---------------------------------------------------------------------
// PinnedHostBuffer
// ---------------------------------------------------------------------
PinnedHostBuffer::PinnedHostBuffer(std::size_t floats)
    : data_(nullptr), floats_(floats)
{
    if (floats == 0) return;
    if (cudaMallocHost(&data_, floats * sizeof(float)) != cudaSuccess)
        throw std::runtime_error("cudaMallocHost failed (" +
                                 std::to_string(floats * sizeof(float)) + " bytes)");
}

PinnedHostBuffer::~PinnedHostBuffer() {
    if (data_) cudaFreeHost(data_);
}

PinnedHostBuffer::PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
    : data_(other.data_), floats_(other.floats_)
{
    other.data_   = nullptr;
    other.floats_ = 0;
}

PinnedHostBuffer& PinnedHostBuffer::operator=(PinnedHostBuffer&& other) noexcept {
    if (this != &other) {
        if (data_) cudaFreeHost(data_);
        data_         = other.data_;
        floats_       = other.floats_;
        other.data_   = nullptr;
        other.floats_ = 0;
    }
    return *this;
}

// ---------------------------------------------------------------------
// SmAllocation — the default (unpartitioned) behaviour of the hooks
// ---------------------------------------------------------------------
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
        // contend for it on their own.
        std::vector<std::unique_ptr<SmAllocation>> out;
        out.reserve(percents.size());
        for (std::size_t i = 0; i < percents.size(); ++i)
            out.push_back(std::make_unique<WholeGpu>());
        return out;
    }

    if (kind == "greenctx") {
        // ONE joint split for every network, so the groups are disjoint by
        // construction. The plan outlives the construction of all the
        // allocations below, which is all it needs to do.
        const PartitionPlan plan(percents);
        if (plan.size() > 1) {
            // With several networks the split itself needs explaining, and so
            // does any capacity left unassigned.
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

// ---------------------------------------------------------------------
// PartitionPlan
// ---------------------------------------------------------------------
PartitionPlan::PartitionPlan(const std::vector<int>& percents) {
    if (percents.empty())
        throw std::runtime_error("PartitionPlan needs at least one partition");
    for (int p : percents)
        if (p < 1 || p > 100)
            throw std::runtime_error("each SM percentage must be between 1 and 100");

    const int device = 0;   // single-GPU board; the current device is 0 anyway

    // The guide recommends initialising the primary context before creating a
    // green context, so the creation does not pay for it.
    cudaCheck(cudaSetDevice(device), "cudaSetDevice");

    // The DEVICE resource. Note: this always reports the WHOLE device, even
    // when green contexts are already alive (measured on this board). That is
    // precisely why every group has to come out of one split, below.
    cudaDevResource full{};
    cudaCheck(cudaDeviceGetDevResource(device, &full, cudaDevResourceTypeSm),
              "cudaDeviceGetDevResource");
    totalSms_  = int(full.sm.smCount);
    alignment_ = full.sm.smCoscheduledAlignment ? int(full.sm.smCoscheduledAlignment) : 2;
    minPart_   = full.sm.minSmPartitionSize      ? int(full.sm.minSmPartitionSize)      : 2;

    // Requested percentages -> SM counts, rounded up to the alignment and
    // clamped to what the device allows. The params array must be zero
    // initialised, including its reserved fields.
    std::vector<cudaDevSmResourceGroupParams> params(percents.size());
    long long requestedTotal = 0;
    for (std::size_t i = 0; i < percents.size(); ++i) {
        const unsigned align = static_cast<unsigned>(alignment_);
        const long long target =
            (static_cast<long long>(totalSms_) * percents[i] + 50) / 100;
        unsigned want = static_cast<unsigned>(target);
        want = ((want + align - 1) / align) * align;
        want = std::max(want, static_cast<unsigned>(minPart_));
        want = std::min(want, static_cast<unsigned>(totalSms_));

        params[i].smCount            = want;
        params[i].coscheduledSmCount = align;
        requestedTotal += want;
    }
    if (requestedTotal > totalSms_)
        throw std::runtime_error("the requested SM percentages exceed the device's " +
                                 std::to_string(totalSms_) + " SMs");

    // ONE split for ALL groups: that is what makes the partitions disjoint.
    // Splitting them one at a time (re-splitting the remainder) is what the
    // guide discourages, because the remainder is unstructured.
    std::vector<cudaDevResource> groups(percents.size());
    cudaDevResource remainder{};
    cudaCheck(cudaDevSmResourceSplit(groups.data(), unsigned(percents.size()), &full,
                                     &remainder, 0, params.data()),
              "cudaDevSmResourceSplit");

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
        if (parts_[i].grantedSms != parts_[i].requestedSms)
            s += " [asked " + std::to_string(parts_[i].requestedSms) + "]";
    }
    s += "; unused " + std::to_string(leftover_);
    return s;
}

// ---------------------------------------------------------------------
// GreenContextAllocation
// ---------------------------------------------------------------------
// Construction provisions the green context from an already-split group. It
// must be the first CUDA-touching call for that context in the process.
//
// Exception safety: a constructor that throws does NOT run the destructor, so
// the green context is created LAST and nothing after it may throw. That is
// why the verification below ignores its error and only the log line follows.
GreenContextAllocation::GreenContextAllocation(const SmPartition& part)
    : percent_(part.percent),
      requestedSms_(part.requestedSms),
      totalSms_(part.totalSms),
      grantedSms_(part.grantedSms)
{
    // Finalise the specification and provision the SMs. The
    // resource itself came from PartitionPlan, which did the joint split.
    cudaDevResource group = part.resource;   // the API wants a mutable pointer
    cudaDevResourceDesc_t desc{};
    cudaCheck(cudaDevResourceGenerateDesc(&desc, &group, 1),
              "cudaDevResourceGenerateDesc");
    cudaCheck(cudaGreenCtxCreate(&execCtx_, desc, /*device=*/0, 0), "cudaGreenCtxCreate");

    // Ask what we actually got. The driver may round or align, so record the
    // grant rather than the request; fall back to the split's number if the
    // query fails.
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
    if (!execCtx_)
        throw std::runtime_error("green context is missing (was the allocation constructed?)");
    cudaStream_t s = nullptr;
    // cudaStreamNonBlocking is the default for a green context stream, but we
    // ask for it explicitly: it never synchronises with the legacy default
    // stream.
    cudaCheck(cudaExecutionCtxStreamCreate(&s, execCtx_, cudaStreamNonBlocking, 0),
              "cudaExecutionCtxStreamCreate");
    return s;
}

// ---------------------------------------------------------------------
// TrtEngine
// ---------------------------------------------------------------------

// Read a whole binary file into memory. Reading the engine is how the
// engine loads itself, so it stays a private detail here rather than a
// public helper.
std::vector<char> TrtEngine::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open file: " + path);

    // seek to the end to learn the size, then back to the start
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
    //    kernels inside were compiled for THIS GPU and THIS TRT version.
    //    A mismatch returns nullptr, so check it.
    const std::vector<char> engineData = readFile(enginePath);
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine_) throw std::runtime_error("failed to deserialize engine: " + enginePath);

    // 2. Discover the I/O tensors. Ask the engine rather than hardcode
    //    "input"/"output": engines can name things anything, and some
    //    have several inputs. TRT 8+ addresses everything by name.
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

    // 3. Buffer sizes = product of the dimensions (elements).
    inputVolume_ = outputVolume_ = 1;
    for (int d = 0; d < inputDims.nbDims; ++d)  inputVolume_  *= inputDims.d[d];
    for (int d = 0; d < outputDims.nbDims; ++d) outputVolume_ *= outputDims.d[d];

    // Cheap insurance: the input contract in config.hpp must match the
    // shape baked into the engine, otherwise preprocess() writes the
    // wrong number of floats into the buffer.
    const std::int64_t expected =
        std::int64_t(cfg::channels) * cfg::imageSize * cfg::imageSize;
    if (inputVolume_ != expected)
        throw std::runtime_error(
            "config.hpp image constants do not match the engine's input shape");

    // 4. Device buffers, and a pinned host buffer for the output.
    const std::size_t inputBytes  = std::size_t(inputVolume_)  * sizeof(float);
    const std::size_t outputBytes = std::size_t(outputVolume_) * sizeof(float);
    if (cudaMalloc(&dInput_,  inputBytes)  != cudaSuccess)
        throw std::runtime_error("cudaMalloc (input) failed");
    if (cudaMalloc(&dOutput_, outputBytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (output) failed");
    hostOutput_ = PinnedHostBuffer(std::size_t(outputVolume_));

    // 5. Context + tensor addresses.
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("failed to create execution context");
    context_->setTensorAddress(inputName_.c_str(),  dInput_);
    context_->setTensorAddress(outputName_.c_str(), dOutput_);

    // 6. The stream comes from the allocation (a green context would
    //    hand back one bound to its SMs), plus the timing events.
    stream_ = allocation_.createStream();
    if (cudaEventCreate(&startEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (start) failed");
    if (cudaEventCreate(&stopEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (stop) failed");
}

TrtEngine::~TrtEngine() {
    // Release per-run state before freeing the buffers it referenced.
    context_.reset();

    if (startEvt_) cudaEventDestroy(startEvt_);
    if (stopEvt_)  cudaEventDestroy(stopEvt_);
    if (stream_)   cudaStreamDestroy(stream_);
    if (dInput_)   cudaFree(dInput_);
    if (dOutput_)  cudaFree(dOutput_);
    // hostOutput_ (pinned) frees itself; engine_ and runtime_ too.
}

double TrtEngine::infer(const float* hostInput) {
    const std::size_t inputBytes  = std::size_t(inputVolume_)  * sizeof(float);
    const std::size_t outputBytes = std::size_t(outputVolume_) * sizeof(float);

    // The async choreography on our stream. The CUDA events bracket the
    // kernel launch only, so the number we return is free of copy noise
    // and CPU work — it is the number partitioning will move.
    cudaMemcpyAsync(dInput_, hostInput, inputBytes, cudaMemcpyHostToDevice, stream_);
    cudaEventRecord(startEvt_, stream_);
    if (!allocation_.enqueue(*context_, stream_))     // the partitioning hook
        throw std::runtime_error("inference failed");
    cudaEventRecord(stopEvt_, stream_);
    cudaMemcpyAsync(hostOutput_.data(), dOutput_, outputBytes,
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvt_, stopEvt_);
    return double(ms);
}
