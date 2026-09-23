#pragma once
//
// config.hpp — every parameter of the experiment, in one place.
//
//   cfg::    compile-time constants: the ImageNet input contract and the
//            protocol defaults. `inline constexpr` so there is no runtime
//            initialisation and no static-init-order surprise; `std::array`
//            rather than a C array so the size travels with it; and
//            `std::string_view` for the string so it does not allocate.
//
//   Options  run-time parameters, filled from the command line.
//
// The cfg:: image constants must agree with how the .engine was built. The
// input shape is baked into the engine, and if the two disagree the
// preprocessed buffer is the wrong size, which gives garbage rather than an
// error. TrtEngine compares them at construction, so a mismatch fails loudly.
//

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cfg {

// The constants the models were trained with. Do not "improve" them: changing
// any one of these silently degrades accuracy.
inline constexpr std::size_t                 channels        = 3;     // RGB
inline constexpr int                         imageSize       = 224;   // network input (crop)
inline constexpr int                         resizeShortSide = 256;   // resize before crop
inline constexpr std::array<float, channels> mean            = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, channels> stddev          = {0.229f, 0.224f, 0.225f};

inline constexpr std::size_t numWarmup = 200;   // warm-up inferences, discarded

// Size of the fixed preprocessed input set (fixed-input mode only).
//
// This does NOT set the number of samples: the timed loop cycles the set as
// many times as it needs to. Its only job is input DIVERSITY — enough distinct,
// evenly spaced images that no single image's quirks can bias the result. 32
// images cost 18 MB of pinned memory and cover the dataset's size
// distribution.
inline constexpr std::size_t numFixedInputs = 32;

// Run-length bounds. 0 disables a bound; with both at 0 the loop makes a single
// pass over the inputs, which is the fast development mode.
inline constexpr std::size_t minInferences = 5000;  // at least this many inferences
inline constexpr double      minDurationS  = 30.0;  // ...and at least this many seconds

inline constexpr std::string_view defaultCsvPrefix = "bench";

} // namespace cfg

// The run-time parameters, filled from the command line.
//
// A run is one or more networks sharing the same dataset, protocol and
// allocation kind. One entry in `networks` is the solo case (Experiment 1);
// two entries is the concurrent case, where each network gets its own green
// context and both run in the same time window (Experiment 2).
//
// (Experiment 3 does not appear here because it is two processes, so each has
// its own Options with a single network in it. The matrix that ties them
// together lives in sweep.sh.)

// One network in the run.
struct NetworkOptions {
    std::string enginePath;             // .engine file
    int         smPercent = 100;        // share of the device's SMs
    std::string label;                  // used in logs and output file names
};

// The run-length rule, shared by every network so that their windows are
// comparable. 0 disables a bound; both at 0 means a single pass.
struct Protocol {
    std::size_t minInferences = cfg::minInferences;
    double      minDurationS  = cfg::minDurationS;
};

struct Options {
    // Inputs, shared by every network.
    std::string imageDir;                         // dataset directory
    std::size_t maxImages = 0;                    // dataset scan limit, 0 = all

    // Fixed preprocessed input set: preprocessing happens once, outside the
    // timed loop, and the same inputs are reused in every configuration. When
    // false, each iteration decodes and preprocesses an image instead, which is
    // the end-to-end pipeline.
    bool        fixedInputs    = true;
    std::size_t numFixedInputs = cfg::numFixedInputs;

    Protocol protocol;

    // Output base name. With one network the files are
    // <csvPrefix>_latencies.csv; with several, each network gets
    // <csvPrefix>_<label>_latencies.csv so runs never overwrite one another.
    std::string csvPrefix = std::string(cfg::defaultCsvPrefix);

    // SM allocation: one shared kind, one share per network.
    //   "whole"    : no partitioning, the whole GPU (the baseline)
    //   "greenctx" : one Green Context per network, provisioned jointly so the
    //                partitions are disjoint (CUDA 13.1+, granularity 2 SMs)
    std::string allocation = "whole";

    // Green-context work queue configuration, for the experiment that studies
    // the false-dependency problem.
    //
    // Two fields because they are two different knobs, and conflating them is
    // the easy mistake:
    //   wqScope  the cudaDevWorkqueueConfigScope of the work queue resource
    //            placed in each green context's descriptor. "" means put no
    //            such resource in the descriptor at all, which is what every run
    //            up to and including Experiment 2 did, so those results stay
    //            comparable and the new arms differ from them in one documented
    //            way.
    //   wqLimit  the wqConcurrencyLimit to declare: how many concurrent
    //            stream-ordered workloads we tell the driver to expect. This is
    //            a HINT the driver consults when assigning queues; it is not a
    //            cap on the number of streams, and setting it to 1 does not
    //            mean "one stream".
    //
    // Deliberately NOT CUDA_DEVICE_MAX_CONNECTIONS, which bounds how many work
    // queue connections exist at all. That variable is read from the environment
    // before CUDA initialises, so it stays a property of whoever launches the
    // process (sweep.sh) rather than an option here.
    std::string wqScope;      // "device_ctx" | "green_ctx_balanced" | ""
    unsigned    wqLimit = 0;  // 0 = do not set one

    // How many threads submit work to the GPU. 1, the default, drives every
    // network from a single thread while keeping one inference in flight per
    // network, so the launch path is never entered from two places at once.
    // 2 is one thread per network, the naive application: it is what the
    // two-thread measurements of Experiments 2, 4 and 5 used, and sweep.sh
    // asks for it explicitly wherever that is what it means.
    unsigned submitThreads = 1;

    // The networks: one entry is solo, two is concurrent.
    std::vector<NetworkOptions> networks;
};

// The parser, the usage text and the defaults all describe the same thing
// (Options), so they live together in config.cpp. Adding an option means
// editing this header and that file, with the usage text right there to remind
// you what the option is supposed to mean.

// Parse the command line. On a usage error it prints the usage and exits; on an
// invalid value it throws.
Options parseArgs(int argc, char** argv);

// Print the usage text to stderr.
void printUsage(const char* prog);
