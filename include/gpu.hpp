#pragma once
//
// gpu.hpp - the device layer, and the seam the others include.
//
// The CUDA and TensorRT work was one translation unit of 570 lines; it is now three, by concern:
//
//   allocation.hpp/.cpp   the seam for spatial partitioning: which SMs may this client use, and
//                         the joint split that makes the partitions disjoint. PartitionPlan is
//                         internal to allocation.cpp: the factory is the only thing that builds one
//   engine.hpp/.cpp       the TensorRT engine, its context, its buffers, and the pinned host
//                         memory the async copies need
//   gpu.hpp/.cpp          this file: the two things both layers call, and the allocation factory
//
// Nothing here knows about images, datasets, statistics or files. See README.md for the
// methodology and DOC.md for the design notes.
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

#include "allocation.hpp"
#include "engine.hpp"

// Defined in gpu.cpp and called from all three files: a CUDA call that must not be ignored.
void cudaCheck(cudaError_t err, const char* what);



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
