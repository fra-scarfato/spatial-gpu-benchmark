#pragma once
// =====================================================================
// config.hpp — every parameter of the experiment, in one place.
//
//   cfg::     compile-time constants: the ImageNet input contract and
//             the protocol defaults. `inline constexpr` so they are
//             computed at compile time with no runtime initialisation
//             and no static-init-order surprises; `std::array` instead
//             of C arrays so they carry their size; `std::string_view`
//             for the string constant so it does not allocate.
//
//   Options   run-time parameters, filled from the command line.
//
// WARNING: the cfg:: image constants must agree with how the .engine
// was built. The input shape is baked into the engine; if they
// disagree, the preprocessed buffer is the wrong size and you get
// garbage (or a crash) instead of a clear error. TrtEngine checks the
// sizes match at construction, so a mismatch fails loudly.
// =====================================================================

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace cfg {

// --- ImageNet input contract -----------------------------------------
// The constants the models were trained with. Do not "improve" them:
// changing any of these silently degrades accuracy.
inline constexpr std::size_t                 channels        = 3;     // RGB
inline constexpr int                         imageSize       = 224;   // network input (crop)
inline constexpr int                         resizeShortSide = 256;   // resize before crop
inline constexpr std::array<float, channels> mean            = {0.485f, 0.456f, 0.406f};
inline constexpr std::array<float, channels> stddev          = {0.229f, 0.224f, 0.225f};

// --- protocol defaults -------------------------------------------------
inline constexpr std::size_t numWarmup = 200;   // warm-up inferences, discarded

// Size of the fixed preprocessed input set (fixed-input mode only).
// This does NOT set the number of samples: the timed loop cycles the set
// as many times as needed. Its only job is input DIVERSITY — enough
// distinct, evenly spaced images that no single image's quirks can bias
// the result. 32 images cost 18 MB of pinned memory and cover the
// dataset's size distribution.
inline constexpr std::size_t numFixedInputs = 32;

// Run-length bounds. 0 disables a bound; if both are 0, the loop makes a
// single pass over the inputs (fast, for development).
inline constexpr std::size_t minInferences = 5000;  // at least this many inferences
inline constexpr double      minDurationS  = 30.0;  // ...and at least this many seconds

inline constexpr std::string_view defaultCsvPrefix = "bench";

} // namespace cfg

// ---------------------------------------------------------------------
// The run-time parameters, filled from the command line.
//
// A run is one OR MORE networks sharing the same dataset, protocol and
// allocation kind. One entry in `networks` is the solo case (Experiment 1);
// two entries is the concurrent case, where each network gets its own green
// context and both run in the same time window (Experiment 2).
// ---------------------------------------------------------------------

// One network in the run.
struct NetworkOptions {
    std::string enginePath;             // .engine file
    int         smPercent = 100;        // share of the device's SMs
    std::string label;                  // used in logs and output file names
};

// The run-length rule, shared by every network so their windows are
// comparable. 0 disables a bound; both 0 = a single pass.
struct Protocol {
    std::size_t minInferences = cfg::minInferences;
    double      minDurationS  = cfg::minDurationS;
};

struct Options {
    // --- inputs, shared by every network ---
    std::string imageDir;                         // dataset directory
    std::size_t maxImages = 0;                    // dataset scan limit, 0 = all

    // Fixed preprocessed input set (phase 1: preprocessing is done once,
    // outside the timed loop, and the same inputs are reused in every
    // configuration). When false, each iteration decodes and preprocesses
    // an image, i.e. the end-to-end pipeline.
    bool        fixedInputs    = true;
    std::size_t numFixedInputs = cfg::numFixedInputs;

    // --- protocol, shared ---
    Protocol protocol;

    // --- output, shared base name ---
    // With one network the files are <csvPrefix>_latencies.csv. With several,
    // each network gets <csvPrefix>_<label>_latencies.csv so runs never
    // overwrite one another.
    std::string csvPrefix = std::string(cfg::defaultCsvPrefix);

    // --- SM allocation: shared kind, per-network share ---
    //   "whole"    : no partitioning, the whole GPU (baseline)
    //   "greenctx" : one Green Context per network, provisioned jointly so the
    //                partitions are disjoint (CUDA 13.1+, granularity 2 SMs)
    std::string allocation = "whole";

    // --- the networks: one entry = solo, two = concurrent ---
    std::vector<NetworkOptions> networks;
};

// ---------------------------------------------------------------------
// Command line.
//
// The parser, the usage text and the defaults all describe the same thing
// (Options), so they live together in config.cpp: adding an option means
// editing this header and that file, with the usage text right there to
// remind you.
// ---------------------------------------------------------------------

// Parse the positional arguments and options. On a usage error, prints the
// usage and exits; on an invalid value, throws.
Options parseArgs(int argc, char** argv);

// Print the usage text to stderr.
void printUsage(const char* prog);
