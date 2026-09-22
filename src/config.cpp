//
// config.cpp — the command line.
//
// The parser, the usage text and the defaults in config.hpp all describe the
// same thing, so they live together: adding an option means editing config.hpp
// for the field and this file for the parser and the usage text, with the two
// right next to each other.
//
// Nothing here calls CUDA or touches GPU state. It only turns argv into an
// Options struct.
//

#include "config.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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
        << "Work queues (greenctx only; omitted means the descriptor carries SMs only):\n"
        << "  --wq-scope <s>        sharing scope for each green context's work queues.\n"
        << "                        'device_ctx' is the driver default: every context\n"
        << "                        shares one set of queues, so independent work on\n"
        << "                        different streams can land on the SAME queue and\n"
        << "                        acquire a false dependency. 'green_ctx_balanced'\n"
        << "                        asks the driver to give balanced green contexts\n"
        << "                        non-overlapping queues where it can.\n"
        << "  --wq-limit <N>        expected number of concurrent stream-ordered\n"
        << "                        workloads, declared to the driver as a hint when it\n"
        << "                        assigns queues. NOT a stream limit, and NOT\n"
        << "                        CUDA_DEVICE_MAX_CONNECTIONS (that one bounds how many\n"
        << "                        queues exist and is read from the environment before\n"
        << "                        CUDA starts, so set it in the shell).\n"
        << "\n"
        << "Notes:\n"
        << "  On this SoC SM counts are rounded to a multiple of 2, and the actual\n"
        << "  grant is reported. With the default bounds the loop runs until BOTH are\n"
        << "  satisfied, cycling the input set if it runs out; in dataset mode those\n"
        << "  bounds can stop the run before the pool is exhausted, so a full pass\n"
        << "  needs --one-pass.\n";
}

namespace {

// Fetch the value that follows an option. It takes the loop index by reference
// and advances it, so the caller's loop consumes both the option and its value;
// forgetting that would silently treat the value as another option.
std::string nextValue(int& i, int argc, char** argv, const std::string& arg) {
    if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + arg);
    return argv[++i];
}

// Parse <engine>:<percent>[:<label>].
//
// The engine path may itself contain colons in principle, but the engine
// extension never does, so the first colon always separates the path from the
// percentage and the second, if there is one, introduces the label.
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

    // The percentage runs from after the first colon to the second colon, or to
    // the end of the string when there is no label.
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
    // --help is checked first, over the raw argv, so that it works even when
    // the rest of the command line is wrong or incomplete. Asking for help must
    // never depend on having asked for everything else correctly.
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

    // The older positional form: <engine> <image_dir> ... — frontend passed, so
    // the very first argument is the engine path. Anything starting with '-'
    // means the caller used the --net form instead.
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
            // Both bounds to zero is the "run one pass" signal; see Network's
            // stop rule.
            o.protocol.minInferences = 0;
            o.protocol.minDurationS  = 0.0;
        } else if (arg == "--csv")
            o.csvPrefix = nextValue(i, argc, argv, arg);
        else if (arg == "--allocation")
            o.allocation = nextValue(i, argc, argv, arg);
        else if (arg == "--sm-percent") {
            // Only meaningful for the positional form, where there is exactly
            // one network to apply it to. In the --net form each network
            // carries its own percentage, so accepting it here would be
            // ambiguous — refuse rather than guess which network it meant.
            if (!positionalNetwork)
                throw std::runtime_error(
                    "--sm-percent applies to the positional <engine> form; with "
                    "--net, put the percentage in the --net spec");
            o.networks[0].smPercent = std::stoi(nextValue(i, argc, argv, arg));
        } else if (arg == "--wq-scope") {
            // Validated here rather than at use, so a typo fails before a single
            // inference is measured instead of silently falling back to the
            // driver default. A silent fallback would produce an arm labelled
            // 'balanced' that is really the control, which is the one failure
            // mode that would invalidate the whole matrix while looking fine.
            o.wqScope = nextValue(i, argc, argv, arg);
            if (o.wqScope != "device_ctx" && o.wqScope != "green_ctx_balanced")
                throw std::runtime_error(
                    "--wq-scope expects device_ctx or green_ctx_balanced, got: " + o.wqScope);
        } else if (arg == "--wq-limit") {
            o.wqLimit = static_cast<unsigned>(std::stoul(nextValue(i, argc, argv, arg)));
            if (o.wqLimit == 0)
                throw std::runtime_error(
                    "--wq-limit must be at least 1 (omit it to leave the driver default)");
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    // Consistency checks. These are the combinations that would otherwise
    // produce a run that looks fine and measures the wrong thing.
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
    // Work queue options configure a green context's descriptor, so with the
    // unpartitioned allocation there is no descriptor to put them in. Refusing
    // is the honest outcome: ignoring them would produce a run that looks like a
    // work queue arm and measures plain whole-GPU contention.
    if ((!o.wqScope.empty() || o.wqLimit != 0) && o.allocation != "greenctx")
        throw std::runtime_error(
            "--wq-scope/--wq-limit configure a green context, so they need "
            "--allocation greenctx");

    // Unlabelled networks still need a name, because the label is what keeps
    // their output files apart: with several networks each writes
    // <prefix>_<label>_latencies.csv, and two networks sharing a label would
    // overwrite each other.
    for (std::size_t k = 0; k < o.networks.size(); ++k)
        if (o.networks[k].label.empty())
            o.networks[k].label = "net" + std::to_string(k + 1);

    return o;
}
