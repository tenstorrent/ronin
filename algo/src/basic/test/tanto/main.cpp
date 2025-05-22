// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cassert>
#include <string>
#include <unordered_map>
#include <exception>

#include "test/tanto/common.hpp"

namespace {

std::unordered_map<std::string, Algo> str_algo_map = {
    {"eltwise_binary", Algo::ELTWISE_BINARY},
    {"eltwise_sfpu", Algo::ELTWISE_SFPU},
    {"bcast", Algo::BCAST},
    {"matmul_single", Algo::MATMUL_SINGLE},
    {"matmul_multi", Algo::MATMUL_MULTI},
    {"reduce", Algo::REDUCE},
    {"transpose_wh", Algo::TRANSPOSE_WH},
    {"unpack_tilize", Algo::UNPACK_TILIZE},
    {"unpack_untilize", Algo::UNPACK_UNTILIZE}
};

void usage() {
    fprintf(stderr, "Usage: test_tanto <op>\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "<op> is one of\n");
    fprintf(stderr, "    eltwise_binary\n");
    fprintf(stderr, "    eltwise_sfpu\n");
    fprintf(stderr, "    bcast\n");
    fprintf(stderr, "    matmul_single\n");
    fprintf(stderr, "    matmul_multi\n");
    fprintf(stderr, "    reduce\n");
    fprintf(stderr, "    transpose_wh\n");
    fprintf(stderr, "    unpack_tilize\n");
    fprintf(stderr, "    unpack_untilize\n");
    fprintf(stderr, "\n");
}

bool parse_args(int argc, char **argv, Algo &algo) {
    if (argc != 2) {
        return false;
    }
    auto it = str_algo_map.find(argv[1]);
    if (it == str_algo_map.end()) {
        return false;
    }
    algo = it->second;
    return true;
}

} // namespace

int main(int argc, char **argv) {
    Algo algo = Algo(0);
    if (!parse_args(argc, argv, algo)) {
        usage();
        return 1;
    }
    try {
        switch (algo) {
        case Algo::ELTWISE_BINARY:
            main_eltwise_binary();
            break;
        case Algo::ELTWISE_SFPU:
            main_eltwise_sfpu();
            break;
        case Algo::BCAST:
            main_bcast();
            break;
        case Algo::MATMUL_SINGLE:
            main_matmul_single();
            break;
        case Algo::MATMUL_MULTI:
            main_matmul_multi();
            break;
        case Algo::REDUCE:
            main_reduce();
            break;
        case Algo::TRANSPOSE_WH:
            main_transpose_wh();
            break;
        case Algo::UNPACK_TILIZE:
            main_unpack_tilize();
            break;
        case Algo::UNPACK_UNTILIZE:
            main_unpack_untilize();
            break;
        default:
            assert(false);
            break;
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

