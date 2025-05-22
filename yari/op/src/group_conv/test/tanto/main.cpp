// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <string>
#include <unordered_map>
#include <exception>

#include "test/tanto/run.hpp"

bool str_to_int(const char *s, int &v) {
    char *p;
    long t = strtol(s, &p, 10);
    if (*p != '\0') {
        return false;
    }
    int r = int(t);
    if (long(r) != t) {
        return false;
    }
    if (r <= 0) {
        return false;
    }
    v = r;
    return true;
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"basic_batch", Algo::BASIC_BATCH},
    {"dw_batch", Algo::DW_BATCH},
    {"dw_spatial", Algo::DW_SPATIAL},
    {"dsc_batch", Algo::DSC_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::BASIC_BATCH:
    case Algo::DW_BATCH:
    case Algo::DSC_BATCH:
        if (batch_size > 8 && 
                batch_size != 16 && 
                batch_size != 32 && 
                batch_size != 64) {
            return false;
        }
        break;
    case Algo::DW_SPATIAL:
        if (batch_size > 8 && batch_size != 16) {
            return false;
        }
        break;
    default:
        assert(false);
        return false;
    }
    return true;
}

bool parse_args(
        int argc, 
        char **argv, 
        Algo &algo, 
        int &N, 
        int &batch_size,
        int &repeat) {
    int argp = 1;
    // repeat
    repeat = 0;
    if (argp < argc && !strcmp(argv[argp], "-r")) {
        argp++;
        if (argp >= argc) {
            return false;
        }
        if (!str_to_int(argv[argp], repeat)) {
            return false;
        }
        argp++;
    }
    // algo
    if (argp >= argc) {
        return false;
    }
    auto it = str_algo_map.find(argv[argp]);
    if (it == str_algo_map.end()) {
        return false;
    }
    algo = it->second;
    argp++;
    // N
    switch (algo) {
    case Algo::BASIC_BATCH:
    case Algo::DW_BATCH:
    case Algo::DW_SPATIAL:
    case Algo::DSC_BATCH:
        N = 16;
        break;
    default:
        assert(false);
        return false;
    }
    if (argp < argc) {
        if (!str_to_int(argv[argp], N)) {
            return false;
        }
        argp++;
    }
    // batch_size
    switch (algo) {
    case Algo::BASIC_BATCH:
    case Algo::DW_BATCH:
    case Algo::DSC_BATCH:
        batch_size = (N > 64) ? 64 : N;
        break;
    case Algo::DW_SPATIAL:
        batch_size = (N > 16) ? 16 : N;
        break;
    default:
        assert(false);
        return false;
    }
    if (argp < argc) {
        if (!str_to_int(argv[argp], batch_size)) {
            return false;
        }
        argp++;
    }
    if (argp < argc) {
        return false;
    }
    return true;
}

void usage() {
    fprintf(stderr, "Usage: test_tanto [-r <repeat>] <op> [<N>] [<B>]\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "<op> is one of\n");
    fprintf(stderr, "    basic_batch\n");
    fprintf(stderr, "    dw_batch\n");
    fprintf(stderr, "    dw_spatial\n");
    fprintf(stderr, "    dsc_batch\n");
    fprintf(stderr, "\n");
}

int main(int argc, char **argv) {
    Algo algo = Algo(0);
    int N = 0;
    int batch_size = 0;
    int repeat = 0;
    if (!parse_args(argc, argv, algo, N, batch_size, repeat)) {
        usage();
        return 1;
    }
    if (!validate_args(algo, N, batch_size)) {
        fprintf(stderr, "Invalid combination of command line arguments\n");
        return 1;
    }
    try {
    switch (algo) {
        case Algo::BASIC_BATCH:
        case Algo::DW_BATCH:
        case Algo::DW_SPATIAL:
            run_group(algo, N, batch_size, repeat);
            break;
        case Algo::DSC_BATCH:
            run_dsc(algo, N, batch_size, repeat);
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

