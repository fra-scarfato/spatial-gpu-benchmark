#pragma once
//
// engine.hpp - the TensorRT engine, its context and its buffers.
//
// TrtEngine owns everything a single network needs to run: the runtime, the engine, the execution
// context, the device buffers and the host side of the outputs. It takes an SmAllocation by
// reference rather than making one, so the partition stays the caller's decision. PinnedHostBuffer
// is here because the async copies need page-locked host memory; an interface, so that code above
// can own pinned memory without seeing a CUDA type.
//

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "allocation.hpp"

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
