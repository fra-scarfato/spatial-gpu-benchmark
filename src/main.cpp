// =====================================================================
// main.cpp — wiring only: parse the arguments, then run.
//
// Everything else lives elsewhere:
//   config.hpp / .cpp   every parameter, and the command line
//   gpu.hpp / .cpp      the GPU layer: engine, SM partitioning, allocations
//   inputs.hpp / .cpp   the tensors (fixed set, or the end-to-end pipeline)
//   stats.hpp / .cpp    the statistics
//   run.hpp / .cpp      a Network (one model + its measurement) and a Run
//                       (one process invocation: 1..N networks)
// See README.md for the methodology, the measurements and the design notes.
// =====================================================================

#include <cstdlib>
#include <exception>
#include <iostream>

#include "config.hpp"
#include "run.hpp"

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);

        Run run(opts);     // assemble: images, allocations, networks
        run.execute();     // warm-up + timed phase (sequential or aligned threads)
        run.report();      // per-network tables + CSVs, then the aggregate

    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
