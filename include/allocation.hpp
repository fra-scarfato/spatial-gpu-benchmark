#pragma once
//
// allocation.hpp - spatial partitioning: which SMs may this client use?
//
// SmAllocation is the seam the rest of the program sees; WholeGpu and GreenContextAllocation are
// the two implementations, and the factory that builds them from a list of percentages lives in
// gpu.hpp. PartitionPlan, which does the joint split the green contexts need, is internal to
// allocation.cpp: one factory is the only caller, so it has no business in a header.
//
// The resource comes from the *device* rather than from a context, so that the partitions are
// disjoint: see DOC.md section 4.4 for why a single cudaDevSmResourceSplit call is not an
// optimisation but a requirement.
//

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace nvinfer1 { class IExecutionContext; }

#include "config.hpp"

// Defined in gpu.cpp, called from here.
void cudaCheck(cudaError_t err, const char* what);
const char* wqScopeName(cudaDevWorkqueueConfigScope scope);


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
