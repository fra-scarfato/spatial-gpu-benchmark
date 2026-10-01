#pragma once
//
// run.hpp - one experiment: the clients, the joint split, the protocol, the report.
//
// The per-client run is in network.hpp. This is what drives them together - including the
// synthetic participant, which is a client like any other - and what prints the record of a
// configuration, since the granted SM counts appear nowhere else.
//

#include <memory>
#include <string>
#include <vector>

#include "config.hpp"
#include "gpu.hpp"
#include "inputs.hpp"
#include "interferer.hpp"
#include "network.hpp"
#include "stats.hpp"

class Run {
public:
    explicit Run(const Options& opts);

    void execute();       // warm-up + timed phase
    void report() const;  // per-network results, then the aggregate

private:
    Options                               opts_;
    std::vector<std::string>              images_;
    std::vector<std::unique_ptr<Network>> nets_;

    std::unique_ptr<Interferer>           interferer_;   // optional participant
};
