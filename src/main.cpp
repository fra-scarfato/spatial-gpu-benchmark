//
// main.cpp — wiring only: parse the arguments, then run.
//
// Everything else lives elsewhere:
//   config.hpp / .cpp   every parameter, and the command line
//   gpu.hpp / .cpp      the GPU layer: engine, SM partitioning, allocations
//   inputs.hpp / .cpp   the tensors (fixed set, or the end-to-end pipeline)
//   stats.hpp / .cpp    the statistics
//   run.hpp / .cpp      a Network (one model plus its measurement) and a Run
//                       (one process invocation: 1..N networks)
//
// See README.md for the methodology, the measurements and the design notes.
//

#include <cstdlib>
#include <exception>
#include <iostream>

#include "config.hpp"
#include "run.hpp"

int main(int argc, char** argv) {
    try {
        const Options opts = parseArgs(argc, argv);

        // Three steps, in this order, because each depends on the last: the
        // Run constructor is where CUDA is first touched (it provisions the
        // allocations before building any engine), execute() does the
        // measurement, and report() writes it out.
        Run run(opts);     // assemble: images, allocations, networks
        run.execute();     // warm-up + timed phase (sequential or aligned threads)
        run.report();      // per-network tables + CSVs, then the aggregate

    } catch (const std::exception& e) {
        // One catch-all at the top. The lower layers throw rather than exiting,
        // so this is the only place that decides the process should die — and
        // it is what keeps a CUDA failure from looking like a segfault with no
        // message.
        std::cerr << "[error] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
