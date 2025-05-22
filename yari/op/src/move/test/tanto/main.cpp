// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <array>
#include <unordered_map>
#include <exception>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/tanto/load_dist.hpp"
#include "host/tanto/store_dist.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::move;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

enum class Algo {
    LOAD_DIST
};

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

void compare(const std::vector<float> &y, const std::vector<float> &yref) {
    float rtol = 1.0e-1f;
    float atol = 1.0e-3f;
    float rtol_delta = 0.0f;
    float atol_delta = 0.0f;
    int num_outliers = 0;

    bool allclose = 
        util::comp_allclose(
            yref, 
            y, 
            rtol, 
            atol, 
            rtol_delta, 
            atol_delta, 
            num_outliers);
    printf("All close = %s\n", allclose ? "OK" : "FAIL");
    printf("Max ATOL delta: %g, max RTOL delta: %g, outliers: %d / %zd\n", 
        atol_delta, rtol_delta, num_outliers, y.size());

    float pcc = util::comp_pcc(yref, y);
    printf("Pcc = %s\n", (pcc >= 0.9999f) ? "OK" : "FAIL"); 
    printf("PCC: %g\n", pcc);
}

void run_load_dist(Algo algo, int N, int batch_size, int repeat) {
    int H = 56 * 56;
    int C = 64;
    printf("---- Batch [%d / %d] H [%d] C [%d]\n", N, batch_size, H, C);

    int xsize = N * H * C;
    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);

    tanto::LoadDist load(N, H, C, batch_size);
    tanto::StoreDist store(N, H, C, batch_size);

    std::vector<uint16_t> tx = float_to_u16b(load.transform_input(0, x));
    std::vector<uint16_t> ty(store.output_volume(0));

    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    uint32_t worker_x, worker_y;
    device.worker_grid_size(worker_x, worker_y);
    int num_cores = worker_x * worker_y;

    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, load.input_volume(0), log2_page_size);
    core::Global gy(device, T, store.output_volume(0), log2_page_size);
    core::Local ly(device, T, load.output_volume(0) / num_cores);
    core::Local lx(ly);

    load.init(device, gx, ly);
    store.init(device, lx, gy);

    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
    if (repeat <= 0) {
        load.run();
        store.run();
    } else {
        // warm up
        load.run();
        store.run();
        queue.finish();
        util::Timer timer;
        timer.start();
        for (int i = 0; i < repeat; i++) {
            load.run();
            store.run();
            queue.finish();
        }
        timer.stop();
        float t = timer.elapsed();
        printf("Elapsed time %g ms / %d iterations = %g\n", t, repeat, t / float(repeat));
    }

    queue.enqueue_read(gy, ty.data(), false);
    queue.finish();
    device.close();
    std::vector<float> y = store.transform_output(0, u16b_to_float(ty));

    compare(y, x);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"load_dist", Algo::LOAD_DIST}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N != batch_size) {
        return false;
    }
    switch (algo) {
    case Algo::LOAD_DIST:
        if (batch_size != 8 && batch_size != 16) {
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
    case Algo::LOAD_DIST:
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
    case Algo::LOAD_DIST:
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
    fprintf(stderr, "    load_dist\n");
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
        case Algo::LOAD_DIST:
            run_load_dist(algo, N, batch_size, repeat);
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

