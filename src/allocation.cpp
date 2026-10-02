// allocation.cpp - the SM partitioning layer.
#include "gpu.hpp"

#include <iostream>

// Internal to this file, with the plan:
const char* wqScopeName(cudaDevWorkqueueConfigScope scope) {
    switch (scope) {
        case cudaDevWorkqueueConfigScopeDeviceCtx:        return "device_ctx";
        case cudaDevWorkqueueConfigScopeGreenCtxBalanced: return "green_ctx_balanced";
        default:                                          return "(unknown)";
    }
}

// Internal to this file: the joint split needs a plan, and nothing outside builds one.

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
    const std::string& kind, const std::vector<int>& percents, const WqConfig& wq)
{
    if (percents.empty())
        throw std::runtime_error("makeSmAllocations needs at least one network");

    // What the platform offers BEFORE we ask for anything, logged for EVERY
    // allocation kind — including "whole", which gets no green context and
    // therefore no descriptor to put a work queue configuration in, but whose
    // streams still run within the device's queue budget. The default is the
    // value of CUDA_DEVICE_MAX_CONNECTIONS (8 on this board, 2 under MPS), so
    // this line is what makes each run's log say which budget it ran with. That
    // was previously unknowable after the fact, and the two experiments we have
    // were compared without it.
    cudaCheck(cudaSetDevice(0), "cudaSetDevice");
    cudaDevResource deviceWq{};
    if (cudaDeviceGetDevResource(0, &deviceWq, cudaDevResourceTypeWorkqueueConfig) == cudaSuccess &&
        deviceWq.type == cudaDevResourceTypeWorkqueueConfig)
        std::cout << "[info] work queue default: limit "
                  << deviceWq.wqConfig.wqConcurrencyLimit << ", scope "
                  << wqScopeName(deviceWq.wqConfig.sharingScope) << "\n";
    else
        std::cout << "[info] work queue default: not queryable on this device\n";

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
        if (wq.set)
            std::cout << "[info] work queue requested: limit " << wq.limit
                      << ", scope " << wqScopeName(wq.scope) << "\n";

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
            out.push_back(std::make_unique<GreenContextAllocation>(plan[i], wq));
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
// the green context is created LAST and nothing after it may throw. That is why
// the READBACKS below ignore their own errors instead of checking them.

GreenContextAllocation::GreenContextAllocation(const SmPartition& part, const WqConfig& wq)
    : percent_(part.percent),
      requestedSms_(part.requestedSms),
      totalSms_(part.totalSms),
      grantedSms_(part.grantedSms),
      wq_(wq)
{
    // Finalise the specification, then provision the SMs and, if asked, the work
    // queues. The SM resource came from PartitionPlan, which did the joint split.
    //
    // The copy is not redundant: the API takes a mutable pointer and may write
    // through it, and `part` is const.
    //
    // With wq_.set the descriptor carries a SECOND resource, the work queue
    // configuration. There is no split API for that one: its fields are set in
    // place on a resource fetched from the device, and at most one may be present
    // in a descriptor.
    cudaDevResource resources[2] = {part.resource, {}};
    unsigned numResources = 1;
    if (wq_.set) {
        resources[1].type                        = cudaDevResourceTypeWorkqueueConfig;
        resources[1].wqConfig.device             = 0;
        resources[1].wqConfig.sharingScope       = wq_.scope;
        resources[1].wqConfig.wqConcurrencyLimit = wq_.limit;
        numResources = 2;
    }
    cudaDevResourceDesc_t desc{};
    // Both of these throw BEFORE the green context exists, so a failure leaves
    // nothing behind. Every CUDA call after the create must not throw.
    cudaCheck(cudaDevResourceGenerateDesc(&desc, resources, numResources),
              "cudaDevResourceGenerateDesc");
    cudaCheck(cudaGreenCtxCreate(&execCtx_, desc, /*device=*/0, 0), "cudaGreenCtxCreate");

    // Ask what we actually got, for the SMs and for the work queues. The driver
    // may round or align the SMs; for the queues, what comes back is what the
    // context was PROVISIONED with, which is not proof that a particular queue
    // assignment was achieved — the whole request is a hint. It is proof that the
    // knob was applied, which is what this line is for. If either query fails we
    // keep what we already knew, which is the best information available.
    cudaDevResource got{};
    if (cudaExecutionCtxGetDevResource(execCtx_, &got, cudaDevResourceTypeSm) == cudaSuccess &&
        got.type == cudaDevResourceTypeSm)
        grantedSms_ = int(got.sm.smCount);

    std::cout << "[info] green context: " << percent_ << "% requested, "
              << grantedSms_ << " of " << totalSms_ << " SMs provisioned\n";

    cudaDevResource gotWq{};
    if (wq_.set &&
        cudaExecutionCtxGetDevResource(execCtx_, &gotWq, cudaDevResourceTypeWorkqueueConfig) == cudaSuccess &&
        gotWq.type == cudaDevResourceTypeWorkqueueConfig)
        std::cout << "[info] green context: work queues granted limit "
                  << gotWq.wqConfig.wqConcurrencyLimit << ", scope "
                  << wqScopeName(gotWq.wqConfig.sharingScope)
                  << " (requested " << wq_.limit << ", " << wqScopeName(wq_.scope) << ")\n";
    else if (wq_.set)
        std::cout << "[info] green context: work queue grant not queryable\n";
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
