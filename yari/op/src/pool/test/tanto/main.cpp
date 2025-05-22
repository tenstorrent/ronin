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

#include "host/tanto/pool2d_batch.hpp"

#include "host/ref/pool2d_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::pool;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;

enum class Algo {
    POOL2D_BATCH
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

enum class Pool2dOp {
    AVG,
    MAX
};

const char *pool2d_op_str(Pool2dOp op) {
    switch (op) {
    case Pool2dOp::AVG:
        return "avg";
    case Pool2dOp::MAX:
        return "max";
    default:
        assert(false);
        return "?";
    }
}

struct Pool2dParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int R;
    int S;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
};

std::vector<Pool2dOp> op_config = {
    Pool2dOp::AVG,
    Pool2dOp::MAX
};

// Based on ResNet50 pooling geometry

std::vector<Pool2dParam> param_config = {
    // H, W, C, P, Q, R, S, stride_h, stride_w, pad_h, pad_w
    {112, 112, 64, 56, 56, 3, 3, 2, 2, 1, 1},
    {56, 56, 256, 28, 28, 3, 3, 2, 2, 1, 1},
    {28, 28, 512, 14, 14, 3, 3, 2, 2, 1, 1},
    {14, 14, 1024, 7, 7, 3, 3, 2, 2, 1, 1},
    {112, 112, 64, 56, 56, 1, 1, 2, 2, 0, 0},
    {56, 56, 256, 28, 28, 1, 1, 2, 2, 0, 0},
    {28, 28, 512, 14, 14, 1, 1, 2, 2, 0, 0},
    {14, 14, 1024, 7, 7, 1, 1, 2, 2, 0, 0}
};

template<typename SOLVER>
void run_pool2d(
        const std::vector<float> &x,
        std::vector<float> &y,
        const Pool2dParam &param,
        int N,
        int batch_size,
        int repeat) {
    int dilation_h = 1;
    int dilation_w = 1;
    SOLVER solver(
        N,
        param.H,
        param.W,
        param.C,
        param.P,
        param.Q,
        param.R,
        param.S,
        param.pad_h,
        param.pad_w,
        param.stride_h,
        param.stride_w,
        dilation_h,
        dilation_w,
        batch_size);
    std::vector<uint16_t> tx = float_to_u16b(solver.transform_input(0, x));
    std::vector<uint16_t> ty(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, solver.input_volume(0), log2_page_size);
    core::Global gy(device, T, solver.output_volume(0), log2_page_size);
    solver.init(device, gx, gy);
    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
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

void run_pool2d_batch(
        const std::vector<float> &x,
        std::vector<float> &y,
        Pool2dOp op,
        const Pool2dParam &param,
        int N,
        int batch_size,
        int repeat) {
    if (op == Pool2dOp::AVG) {
        run_pool2d<tanto::AvgPool2dBatch>(x, y, param, N, batch_size, repeat);
    } else if (op == Pool2dOp::MAX) {
        run_pool2d<tanto::MaxPool2dBatch>(x, y, param, N, batch_size, repeat);
    } else {
        assert(false);
    }
}

void run_ref(
        const std::vector<float> &x,
        std::vector<float> &y,
        Pool2dOp op,
        const Pool2dParam &param,
        int N) {
    y.resize(N * param.P * param.Q * param.C);
    int dilation_h = 1;
    int dilation_w = 1;
    if (op == Pool2dOp::AVG) {
        ref::AvgPool2dRef solver(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            dilation_h,
            dilation_w);
        solver.init(x.data(), y.data());
        solver.run();
    } else if (op == Pool2dOp::MAX) {
        ref::MaxPool2dRef solver(
            N,
            param.H,
            param.W,
            param.C,
            param.P,
            param.Q,
            param.R,
            param.S,
            param.pad_h,
            param.pad_w,
            param.stride_h,
            param.stride_w,
            dilation_h,
            dilation_w);
        solver.init(x.data(), y.data());
        solver.run();
    } else {
        assert(false);
    }
}

void run_algo(
        Algo algo,
        const std::vector<float> &x,
        std::vector<float> &y,
        Pool2dOp op,
        const Pool2dParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::POOL2D_BATCH:
        run_pool2d_batch(x, y, op, param, N, batch_size, repeat);
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
        Pool2dOp op,
        const Pool2dParam &param, 
        int N, 
        int batch_size,
        int repeat) {
    printf(
        "---- Batch [%d / %d] op [%s] HWC [%d %d %d] PQ [%d %d] "
        "RS [%d %d] pad [%d %d] stride [%d %d]\n",
            N, batch_size, pool2d_op_str(op),
            param.H, param.W, param.C, 
            param.P, param.Q, 
            param.R, param.S, 
            param.pad_h, param.pad_w, 
            param.stride_h, param.stride_w);

    int xsize = N * param.H * param.W * param.C;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);

    std::vector<float> y;
    std::vector<float> yref;

    run_algo(algo, x, y, op, param, N, batch_size, repeat);
    run_ref(x, yref, op, param, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"pool2d_batch", Algo::POOL2D_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::POOL2D_BATCH:
        if (batch_size != 8 && 
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
    case Algo::POOL2D_BATCH:
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
    case Algo::POOL2D_BATCH:
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
    fprintf(stderr, "    pool2d_batch\n");
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
        for (Pool2dOp op: op_config) {
        for (Pool2dParam &param: param_config) {
            run(algo, op, param, N, batch_size, repeat);
        }
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

