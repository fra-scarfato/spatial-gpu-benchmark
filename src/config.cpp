// =====================================================================
// config.cpp — the command line.
//
// The parser, the usage text and the defaults in config.hpp all describe
// the same thing (Options), so they live together: adding an option means
// editing config.hpp (the field) and this file (the parser + the usage
// text), with the two right next to each other.
//
// Note that this file makes no CUDA call and touches no GPU state: it only
// turns argv into an Options struct.
// =====================================================================

#include "config.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------
// printUsage
// ---------------------------------------------------------------------
void printUsage(const char* prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog << " <engine.engine> <image_dir> [options]        # one network\n"
        << "  " << prog << " --dataset <dir> --net <engine>:<pct>[:<label>] # one or more\n"
        << "          [--net <engine>:<pct>[:<label>] ...]\n"
        << "\n"
        << "Inputs:\n"
        << "  --fixed <N>           fixed set of N preprocessed inputs, cycled\n"
        << "                        (default " << cfg::numFixedInputs << "). Decoded and preprocessed once,\n"
        << "                        outside the timed loop.\n"
        << "  --end-to-end          decode and preprocess each image inside the loop\n"
        << "                        instead (the end-to-end pipeline, no fixed set).\n"
        << "  --images <N>          limit the image pool to the first N (0 = all).\n"
        << "                        In dataset mode this IS the input set, so it sets\n"
        << "                        the length of a pass. In fixed mode it only selects\n"
        << "                        the pool the set is sampled from.\n"
        << "  --dataset <dir>       the dataset directory (with the --net form)\n"
        << "\n"
        << "Networks:\n"
        << "  --net <spec>          engine:pct[:label]. Repeat it to run several\n"
        << "                        networks concurrently, each on its own green\n"
        << "                        context; the percentages must sum to <= 100.\n"
        << "\n"
        << "Run length (a bound of 0 disables it):\n"
        << "  --min-inferences <N>  at least N inferences (default " << cfg::minInferences << ")\n"
        << "  --min-duration <S>    at least S seconds (default " << cfg::minDurationS << ")\n"
        << "  --one-pass            ignore both bounds: one pass over the input set\n"
        << "                        (" << cfg::numFixedInputs << " inferences in fixed mode, the whole\n"
        << "                        pool in dataset mode). This is NOT necessarily fast.\n"
        << "\n"
        << "Output and allocation:\n"
        << "  --csv <prefix>        output prefix (default \"" << cfg::defaultCsvPrefix << "\")\n"
        << "  --allocation <name>   SM allocation: 'whole' (default, no partitioning)\n"
        << "                        or 'greenctx' (one Green Context per network)\n"
        << "  --sm-percent <P>      for greenctx with the positional <engine> form:\n"
        << "                        SMs to provision as a percentage of the device's\n"
        << "                        (default 100). With --net, put the percentage in\n"
        << "                        the --net spec instead.\n"
        << "\n"
        << "Notes:\n"
        << "  On this SoC SM counts are rounded to a multiple of 2, and the actual\n"
        << "  grant is reported. With the default bounds the loop runs until BOTH are\n"
        << "  satisfied, cycling the input set if it runs out; in dataset mode those\n"
        << "  bounds can stop the run before the pool is exhausted, so a full pass\n"
        << "  needs --one-pass.\n";
}

// ---------------------------------------------------------------------
// parseArgs
// ---------------------------------------------------------------------
namespace {

std::string nextValue(int& i, int argc, char** argv, const std::string& arg) {
    if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + arg);
    return argv[++i];
}

// <engine>:<percent>[:<label>]
NetworkOptions parseNetSpec(const std::string& spec) {
    const std::size_t c1 = spec.find(':');
    if (c1 == std::string::npos)
        throw std::runtime_error(
            "--net expects <engine>:<percent>[:<label>], got: " + spec);

    const std::size_t c2 = spec.find(':', c1 + 1);

    NetworkOptions n;
    n.enginePath = spec.substr(0, c1);
    if (n.enginePath.empty())
        throw std::runtime_error("--net: empty engine path in: " + spec);

    const std::string pct =
        spec.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
    n.smPercent = std::stoi(pct);
    if (n.smPercent < 1 || n.smPercent > 100)
        throw std::runtime_error("--net: percentage must be between 1 and 100 in: " + spec);

    if (c2 != std::string::npos)
        n.label = spec.substr(c2 + 1);

    return n;
}

} // namespace

Options parseArgs(int argc, char** argv) {
    // --help wins over everything, including missing arguments.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
    }

    Options o;
    bool positionalNetwork = false;
    int  i = 1;

    // Backward-compatible positional form: <engine> <image_dir> ...
    if (i < argc && argv[i][0] != '-') {
        NetworkOptions n;
        n.enginePath = argv[i++];
        if (i >= argc) {
            std::cerr << "[error] missing <image_dir> after " << n.enginePath << "\n";
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
        o.imageDir = argv[i++];
        o.networks.push_back(n);
        positionalNetwork = true;
    }

    for (; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--net")
            o.networks.push_back(parseNetSpec(nextValue(i, argc, argv, arg)));
        else if (arg == "--dataset")
            o.imageDir = nextValue(i, argc, argv, arg);
        else if (arg == "--fixed")
            o.numFixedInputs = std::stoull(nextValue(i, argc, argv, arg));
        else if (arg == "--end-to-end")
            o.fixedInputs = false;
        else if (arg == "--images")
            o.maxImages = std::stoull(nextValue(i, argc, argv, arg));
        else if (arg == "--min-inferences")
            o.protocol.minInferences = std::stoull(nextValue(i, argc, argv, arg));
        else if (arg == "--min-duration")
            o.protocol.minDurationS = std::stod(nextValue(i, argc, argv, arg));
        else if (arg == "--one-pass") {
            o.protocol.minInferences = 0;
            o.protocol.minDurationS  = 0.0;
        } else if (arg == "--csv")
            o.csvPrefix = nextValue(i, argc, argv, arg);
        else if (arg == "--allocation")
            o.allocation = nextValue(i, argc, argv, arg);
        else if (arg == "--sm-percent") {
            if (!positionalNetwork)
                throw std::runtime_error(
                    "--sm-percent applies to the positional <engine> form; with "
                    "--net, put the percentage in the --net spec");
            o.networks[0].smPercent = std::stoi(nextValue(i, argc, argv, arg));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (o.networks.empty())
        throw std::runtime_error("no network given: pass <engine> or --net <engine>:<pct>");
    if (positionalNetwork && o.networks.size() > 1)
        throw std::runtime_error(
            "do not mix the positional <engine> form with --net: express every "
            "network with --net, and put each percentage in its own --net spec");
    if (o.imageDir.empty())
        throw std::runtime_error("no dataset given: pass <image_dir> or --dataset <dir>");
    if (o.networks.size() > 1 && o.csvPrefix == "") 
        throw std::runtime_error("--csv needs a non-empty prefix");
    if (o.fixedInputs && o.numFixedInputs == 0)
        throw std::runtime_error("--fixed needs a set size of at least 1");

    for (std::size_t k = 0; k < o.networks.size(); ++k)
        if (o.networks[k].label.empty())
            o.networks[k].label = "net" + std::to_string(k + 1);

    return o;
}
