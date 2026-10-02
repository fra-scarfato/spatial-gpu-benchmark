// engine.cpp - the TensorRT engine and the pinned host buffers.
#include "gpu.hpp"

#include <fstream>
#include <iostream>

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
// ---------------------------------------------------------------- interferer
//
// A synthetic participant that occupies one SM partition and does one kind of work
// only. A real co-runner cannot be used for this: its SM budget moves its arithmetic
// and its memory traffic together, so "the co-runner got bigger" says nothing about
// which of the two cost the target.
//
// Two constraints define the design. The memory kind's working set has to be well above
// L2, which is 4 MB on this SoC, or it measures the cache rather than DRAM. The compute
// kind's operands have to stay inside L2, or it becomes memory-bound as well and the
// contrast disappears.
//
// It is in-process and inside the same joint split as the networks, because a second
// process serialises, as the two-process arms of the first campaign showed.

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
    nvinfer1::Dims inputDims{};
    std::vector<nvinfer1::Dims> outputDims;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::Dims dims = engine_->getTensorShape(name);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName_  = name;
            inputDims   = dims;
        } else {
            // Every output, in engine order. A network with ten of them is still
            // one network with one input.
            outputNames_.emplace_back(name);
            outputDims.push_back(dims);
        }
    }
    if (outputNames_.empty())
        throw std::runtime_error("the engine has no output tensors");

    // 3. Buffer sizes: the product of the dimensions, in elements.
    inputVolume_ = 1;
    for (int d = 0; d < inputDims.nbDims; ++d) inputVolume_ *= inputDims.d[d];
    outputVolumes_.assign(outputNames_.size(), 1);
    for (std::size_t o = 0; o < outputDims.size(); ++o)
        for (int d = 0; d < outputDims[o].nbDims; ++d)
            outputVolumes_[o] *= outputDims[o].d[d];

    // The input geometry comes from the engine, not from config.hpp, so a network
    // with any input size works without a source change. What the preprocessing
    // does assume is three channels and two spatial dimensions; anything else is
    // refused here. Refused loudly, because getting it wrong is invisible: the
    // timing is unaffected and only the accuracy quietly dies.
    if (inputDims.nbDims < 3)
        throw std::runtime_error("the engine's input has fewer than three dimensions");
    inputHeight_ = inputDims.d[inputDims.nbDims - 2];
    inputWidth_  = inputDims.d[inputDims.nbDims - 1];
    const std::int64_t inChannels = inputDims.d[inputDims.nbDims - 3];
    if (inChannels != cfg::channels || inputHeight_ <= 0 || inputWidth_ <= 0)
        throw std::runtime_error(
            "the preprocessing assumes three channels and two spatial dimensions");

    // 4. Device buffers, plus a pinned host buffer for the output (pinned
    //    because of the async download in infer()).
    const std::size_t inputBytes = std::size_t(inputVolume_) * sizeof(float);
    if (cudaMalloc(&dInput_, inputBytes) != cudaSuccess)
        throw std::runtime_error("cudaMalloc (input) failed");
    dOutputs_.assign(outputNames_.size(), nullptr);
    hostOutputs_.reserve(outputNames_.size());
    for (std::size_t o = 0; o < outputNames_.size(); ++o) {
        const std::size_t bytes = std::size_t(outputVolumes_[o]) * sizeof(float);
        if (cudaMalloc(&dOutputs_[o], bytes) != cudaSuccess)
            throw std::runtime_error("cudaMalloc (output) failed");
        hostOutputs_.emplace_back(std::size_t(outputVolumes_[o]));
    }

    // 5. Context and tensor addresses. The addresses are set once here because
    //    the buffers never move, so enqueueV3 needs no per-call binding array.
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("failed to create execution context");
    context_->setTensorAddress(inputName_.c_str(), dInput_);
    for (std::size_t o = 0; o < outputNames_.size(); ++o)
        context_->setTensorAddress(outputNames_[o].c_str(), dOutputs_[o]);

    // 6. The stream comes from the allocation — under a green context that is a
    //    stream bound to its SMs — plus the two timing events.
    stream_ = allocation_.createStream();
    if (cudaEventCreate(&startEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (start) failed");
    if (cudaEventCreate(&stopEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (stop) failed");
    if (cudaEventCreate(&spanStartEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (span start) failed");
    if (cudaEventCreate(&spanStopEvt_) != cudaSuccess)
        throw std::runtime_error("cudaEventCreate (span stop) failed");
}

TrtEngine::~TrtEngine() {
    // The context goes first, because it refers to the buffers freed below.
    context_.reset();

    if (startEvt_) cudaEventDestroy(startEvt_);
    if (stopEvt_)  cudaEventDestroy(stopEvt_);
    if (spanStartEvt_) cudaEventDestroy(spanStartEvt_);
    if (spanStopEvt_)  cudaEventDestroy(spanStopEvt_);
    if (stream_)   cudaStreamDestroy(stream_);
    if (dInput_)   cudaFree(dInput_);
    for (void* p : dOutputs_)
        if (p) cudaFree(p);
    // hostOutputs_ (pinned), engine_ and runtime_ free themselves.
}

void TrtEngine::submit(const float* hostInput) {
    const std::size_t inputBytes  = std::size_t(inputVolume_)  * sizeof(float);

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
    // The span events bracket the whole step, copies included; the pair below
    // still brackets the kernel alone, so gpu_ms keeps its meaning.
    cudaCheck(cudaEventRecord(spanStartEvt_, stream_),
              "cudaEventRecord (span start)");
    cudaCheck(cudaMemcpyAsync(dInput_, hostInput, inputBytes,
                              cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync (host to device)");
    cudaCheck(cudaEventRecord(startEvt_, stream_), "cudaEventRecord (start)");
    if (!allocation_.enqueue(*context_, stream_))     // the partitioning hook
        throw std::runtime_error("inference failed");
    cudaCheck(cudaEventRecord(stopEvt_, stream_), "cudaEventRecord (stop)");
    for (std::size_t o = 0; o < dOutputs_.size(); ++o)
        cudaCheck(cudaMemcpyAsync(hostOutputs_[o].data(), dOutputs_[o],
                                  std::size_t(outputVolumes_[o]) * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream_),
                  "cudaMemcpyAsync (device to host)");

    cudaCheck(cudaEventRecord(spanStopEvt_, stream_),
              "cudaEventRecord (span stop)");
}

// Non-blocking. cudaSuccess means the stream has drained; cudaErrorNotReady
// means there is still work on it, which is the answer the single-thread
// driver needs. Anything else is a real error and is raised as one.
bool TrtEngine::ready() {
    const cudaError_t st = cudaStreamQuery(stream_);
    if (st == cudaSuccess)
        return true;
    if (st == cudaErrorNotReady)
        return false;
    cudaCheck(st, "cudaStreamQuery");
    return false;   // unreachable: cudaCheck threw
}

  // The whole stream-side step: H2D + kernel + D2H, from events recorded on
  // this network's own stream. Valid once the stream has drained, which
  // finish() has already ensured.
  //
  // Events rather than the host clock, deliberately. A host span for the same
  // step would include whatever the submitting thread was doing between the
  // copy and the synchronisation: nothing when each network has its own
  // thread, but the OTHER network's service time when one thread drives both.
  // Taken on the stream, the number stays a property of the network it is
  // filed under, whichever way the run was submitted.
  double TrtEngine::wallMs() {
      float ms = 0.0f;
      cudaCheck(cudaEventElapsedTime(&ms, spanStartEvt_, spanStopEvt_),
                "cudaEventElapsedTime (step)");
      return double(ms);
  }

  double TrtEngine::finish() {
    cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, startEvt_, stopEvt_);
    return double(ms);
}

double TrtEngine::infer(const float* hostInput) {
    submit(hostInput);
    return finish();
}
