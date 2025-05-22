// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <exception>

#include "test/util/net.hpp"

#include "test/tanto/run.hpp"

using namespace ronin::nn::common::test;

namespace {

bool run(const util::NetCmdArgs &args) {
    if (args.mode.empty()) {
        fprintf(stderr, "Missing mode\n");
        return false;
    }
    if (args.mode == "global") {
        run_global(args);
        return true;
    }
    if (args.mode == "global_dsc") {
        run_global_dsc(args);
        return true;
    }
    fprintf(stderr, "Invalid mode: %s\n", args.mode.c_str());
    return false;
}

} // namespace

//
//    Main program
//

int main(int argc, char **argv) {
    util::NetCmdArgs args;
    if (!util::parse_net_cmd_args(argc, argv, args)) {
        return 1;
    }
    try {
        if (!run(args)) {
            return 1;
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

