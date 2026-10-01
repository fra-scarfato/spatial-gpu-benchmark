//
// gpu.cpp - the device layer: the one CUDA call the other two layers make.
//
// The partitioning lives in allocation.cpp and the engine in engine.cpp. What is left here is the
// error check both of them call, which has to be at file scope rather than inside the anonymous
// namespace it started in: internal linkage meant nothing outside this file could link to it, and
// the split is what exposed that.
//

#include "gpu.hpp"

#include <stdexcept>
#include <string>

// Turn a CUDA error into an exception, naming the call that failed. Most of the CUDA API returns a
// status rather than throwing, and an unchecked status here means a silent wrong answer rather than
// a crash.
void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(err));
}
