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

#include "host/tanto/reduce_batch.hpp"

#include "host/ref/reduce_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::reduce;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;

enum class Algo {
    REDUCE_BATCH
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

enum class ReduceOp {
    MAX,
    MEAN,
    SUM
};

const char *reduce_op_str(ReduceOp op) {
    switch (op) {
    case ReduceOp::MAX:
        return "max";
    case ReduceOp::MEAN:
        return "mean";
    case ReduceOp::SUM:
        return "sum";
    default:
        assert(false);
        return "?";
    }
}

struct ReduceParam {
    int H;
    int W;
};

std::vector<ReduceOp> op_config = {
    ReduceOp::MAX,
    ReduceOp::MEAN,
    ReduceOp::SUM
};

std::vector<ReduceParam> param_config = {
    // H, W
    {7 * 7, 1024}
};

std::vector<int> axis_config = {1, 2};

template<typename SOLVER>
void run_reduce(
        const std::vector<float> &x,
        std::vector<float> &y,
        const ReduceParam &param,
        int axis,
        int N,
        int batch_size,
        int repeat) {
    SOLVER solver(N, param.H, param.W, axis, batch_size);
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

void run_reduce_batch(
        const std::vector<float> &x,
        std::vector<float> &y,
        ReduceOp op,
        const ReduceParam &param,
        int axis,
        int N,
        int batch_size,
        int repeat) {
    if (op == ReduceOp::MAX) {
        run_reduce<tanto::ReduceMaxBatch>(x, y, param, axis, N, batch_size, repeat);
    } else if (op == ReduceOp::MEAN) {
        run_reduce<tanto::ReduceMeanBatch>(x, y, param, axis, N, batch_size, repeat);
    } else if (op == ReduceOp::SUM) {
        run_reduce<tanto::ReduceSumBatch>(x, y, param, axis, N, batch_size, repeat);
    } else {
        assert(false);
    }
}

void run_ref(
        const std::vector<float> &x,
        std::vector<float> &y,
        ReduceOp op,
        const ReduceParam &param,
        int axis,
        int N) {
    if (axis == 1) {
        y.resize(N * param.W);
    } else {
        y.resize(N * param.H);
    }
    if (op == ReduceOp::MAX) {
        ref::ReduceMaxRef solver(N, param.H, param.W, axis);
        solver.init(x.data(), y.data());
        solver.run();
    } else if (op == ReduceOp::MEAN) {
        ref::ReduceMeanRef solver(N, param.H, param.W, axis);
        solver.init(x.data(), y.data());
        solver.run();
    } else if (op == ReduceOp::SUM) {
        ref::ReduceSumRef solver(N, param.H, param.W, axis);
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
        ReduceOp op,
        const ReduceParam &param,
        int axis,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::REDUCE_BATCH:
        run_reduce_batch(x, y, op, param, axis, N, batch_size, repeat);
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
        ReduceOp op,
        const ReduceParam &param, 
        int axis,
        int N, 
        int batch_size,
        int repeat) {
    printf(
        "---- Batch [%d / %d] op [%s] HW [%d %d] axis %d\n",
            N, batch_size, reduce_op_str(op), param.H, param.W, axis);

    int xsize = N * param.H * param.W;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);

    std::vector<float> y;
    std::vector<float> yref;

    run_algo(algo, x, y, op, param, axis, N, batch_size, repeat);
    run_ref(x, yref, op, param, axis, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"reduce_batch", Algo::REDUCE_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::REDUCE_BATCH:
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
    case Algo::REDUCE_BATCH:
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
    case Algo::REDUCE_BATCH:
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
    fprintf(stderr, "    reduce_batch\n");
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
        for (ReduceOp op: op_config) {
        for (ReduceParam &param: param_config) {
        for (int axis: axis_config) {
            run(algo, op, param, axis, N, batch_size, repeat);
        }
        }
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

