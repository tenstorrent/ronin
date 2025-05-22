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

#include "host/tanto/fc_batch.hpp"

#include "host/ref/fc_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::fc;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;

enum class Algo {
    FC_BATCH
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

struct FCParam {
    int H;
    int C;
    int K;
};

std::vector<FCParam> param_config = {
    // H, C, K
    {1, 512, 1000}
};

template<typename SOLVER>
void run_fc(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        std::vector<float> &y,
        const FCParam &param,
        int N,
        int batch_size,
        int repeat) {
    SOLVER solver(N, param.H, param.C, param.K, batch_size);
    std::vector<uint16_t> tx = float_to_u16b(solver.transform_input(0, x));
    std::vector<uint16_t> tw = float_to_u16b(solver.transform_input(1, w));
    std::vector<uint16_t> tb = float_to_u16b(solver.transform_input(2, b));
    std::vector<uint16_t> ty(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, solver.input_volume(0), log2_page_size);
    core::Global gw(device, T, solver.input_volume(1), log2_page_size);
    core::Global gb(device, T, solver.input_volume(2), log2_page_size);
    core::Global gy(device, T, solver.output_volume(0), log2_page_size);
    solver.init(device, gx, gw, gb, gy);
    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
    queue.enqueue_write(gw, tw.data(), false);
    queue.enqueue_write(gb, tb.data(), false);
    if (repeat <= 0) {
        solver.run();
    } else {
        // warm up
        solver.run();
        queue.finish();
        util::Timer timer;
        timer.start();
        for (int i = 0; i < repeat; i++) {
            solver.run();
            queue.finish();
        }
        timer.stop();
        float t = timer.elapsed();
        printf("Elapsed time %g ms / %d iterations = %g\n", t, repeat, t / float(repeat));
    }
    queue.enqueue_read(gy, ty.data(), false);
    queue.finish();
    device.close();
    y = solver.transform_output(0, u16b_to_float(ty));
}

void run_fc_batch(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        std::vector<float> &y,
        const FCParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_fc<tanto::FCBatch>(x, w, b, y, param, N, batch_size, repeat);
}

void run_ref(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        std::vector<float> &y,
        const FCParam &param,
        int N) {
    y.resize(N * param.H * param.K);
    ref::FCRef solver(N, param.H, param.C, param.K);
    solver.init(x.data(), w.data(), b.data(), y.data());
    solver.run();
}

void run_algo(
        Algo algo,
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        std::vector<float> &y,
        const FCParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::FC_BATCH:
        run_fc_batch(x, w, b, y, param, N, batch_size, repeat);
        break;
    default:
        assert(false);
        break;
    }
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

void run(
        Algo algo, 
        const FCParam &param, 
        int N, 
        int batch_size,
        int repeat) {
    printf("---- Batch [%d / %d] H [%d] C [%d] K [%d]\n",
        N, batch_size, param.H, param.C, param.K);

    int xsize = N * param.H * param.C;
    int wsize = param.K * param.C;
    int ysize = N * param.H * param.K;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);
    std::vector<float> w = util::normal(0.0f, 0.1f, wsize);
    std::vector<float> b = util::normal(0.0f, 0.1f, param.K);

    std::vector<float> y;
    std::vector<float> yref;

    run_algo(algo, x, w, b, y, param, N, batch_size, repeat);
    run_ref(x, w, b, yref, param, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"fc_batch", Algo::FC_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::FC_BATCH:
        if (batch_size > 8 && 
                batch_size != 16 && 
                batch_size != 32 && 
                batch_size != 64) {
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
    case Algo::FC_BATCH:
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
    case Algo::FC_BATCH:
        batch_size = (N > 64) ? 64 : N;
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
    fprintf(stderr, "    fc_batch\n");
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
        for (FCParam &param: param_config) {
            run(algo, param, N, batch_size, repeat);
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

