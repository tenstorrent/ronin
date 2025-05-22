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

#include "host/tanto/binary_batch.hpp"

#include "host/ref/binary_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::binary;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

enum class Algo {
    BINARY_BATCH
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

enum class BinaryOp {
    ADD,
    SUB,
    MUL
};

const char *binary_op_str(BinaryOp op) {
    switch (op) {
    case BinaryOp::ADD:
        return "add";
    case BinaryOp::SUB:
        return "sub";
    case BinaryOp::MUL:
        return "mul";
    default:
        assert(false);
        return "?";
    }
}

struct BinaryParam {
    int H;
    int C;
};

std::vector<BinaryOp> op_config = {
    BinaryOp::ADD,
    BinaryOp::SUB,
    BinaryOp::MUL
};

std::vector<BinaryParam> param_config = {
    // H, C
    {28 * 28, 256},
    {14 * 14, 512},
    {7 * 7, 1024}
};

std::vector<base::PostOpSpec> post_op_config = {
    base::PostOpSpec(base::PostOp::NONE),
    base::PostOpSpec(base::PostOp::RELU),
    base::PostOpSpec(base::PostOp::CLIP, 0.0f, 6.0f)
};

template<typename SOLVER>
void run_binary(
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c,
        const BinaryParam &param,
        const base::PostOpSpec &post_op,
        int N,
        int batch_size,
        int repeat) {
    SOLVER solver(N, param.H, param.C, post_op, batch_size);
    std::vector<uint16_t> ta = float_to_u16b(solver.transform_input(0, a));
    std::vector<uint16_t> tb = float_to_u16b(solver.transform_input(1, b));
    std::vector<uint16_t> tc(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global ga(device, T, solver.input_volume(0), log2_page_size);
    core::Global gb(device, T, solver.input_volume(1), log2_page_size);
    core::Global gc(device, T, solver.output_volume(0), log2_page_size);
    solver.init(device, ga, gb, gc);
    core::Queue queue(device, 0);
    queue.enqueue_write(ga, ta.data(), false);
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
    queue.enqueue_read(gc, tc.data(), false);
    queue.finish();
    device.close();
    c = solver.transform_output(0, u16b_to_float(tc));
}

void run_binary_batch(
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c,
        BinaryOp op,
        const BinaryParam &param,
        const base::PostOpSpec &post_op,
        int N,
        int batch_size,
        int repeat) {
    if (op == BinaryOp::ADD) {
        run_binary<tanto::AddBatch>(a, b, c, param, post_op, N, batch_size, repeat);
    } else if (op == BinaryOp::SUB) {
        run_binary<tanto::SubBatch>(a, b, c, param, post_op, N, batch_size, repeat);
    } else if (op == BinaryOp::MUL) {
        run_binary<tanto::MulBatch>(a, b, c, param, post_op, N, batch_size, repeat);
    } else {
        assert(false);
    }
}

void run_ref(
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c,
        BinaryOp op,
        const BinaryParam &param,
        const base::PostOpSpec &post_op,
        int N) {
    c.resize(N * param.H * param.C);
    if (op == BinaryOp::ADD) {
        ref::AddRef solver(N, param.H, param.C, post_op);
        solver.init(a.data(), b.data(), c.data());
        solver.run();
    } else if (op == BinaryOp::SUB) {
        ref::SubRef solver(N, param.H, param.C, post_op);
        solver.init(a.data(), b.data(), c.data());
        solver.run();
    } else if (op == BinaryOp::MUL) {
        ref::MulRef solver(N, param.H, param.C, post_op);
        solver.init(a.data(), b.data(), c.data());
        solver.run();
    } else {
        assert(false);
    }
}

void run_algo(
        Algo algo,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c,
        BinaryOp op,
        const BinaryParam &param,
        const base::PostOpSpec &post_op,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::BINARY_BATCH:
        run_binary_batch(a, b, c, op, param, post_op, N, batch_size, repeat);
        break;
    default:
        assert(false);
        break;
    }
}

void compare(const std::vector<float> &c, const std::vector<float> &cref) {
    float rtol = 1.0e-1f;
    float atol = 1.0e-3f;
    float rtol_delta = 0.0f;
    float atol_delta = 0.0f;
    int num_outliers = 0;

    bool allclose = 
        util::comp_allclose(
            cref, 
            c, 
            rtol, 
            atol, 
            rtol_delta, 
            atol_delta, 
            num_outliers);
    printf("All close = %s\n", allclose ? "OK" : "FAIL");
    printf("Max ATOL delta: %g, max RTOL delta: %g, outliers: %d / %zd\n", 
        atol_delta, rtol_delta, num_outliers, c.size());

    float pcc = util::comp_pcc(cref, c);
    printf("Pcc = %s\n", (pcc >= 0.9999f) ? "OK" : "FAIL"); 
    printf("PCC: %g\n", pcc);
}

void run(
        Algo algo, 
        BinaryOp op,
        const BinaryParam &param, 
        const base::PostOpSpec &post_op,
        int N, 
        int batch_size,
        int repeat) {
    printf(
        "---- Batch [%d / %d] op [%s] H [%d] C [%d] %s\n",
            N, batch_size, binary_op_str(op), param.H, param.C, post_op.str().c_str());

    int size = N * param.H * param.C;

    util::manual_seed(1234);
    std::vector<float> a = util::normal(0.0f, 0.1f, size);
    std::vector<float> b = util::normal(0.0f, 0.1f, size);

    std::vector<float> c;
    std::vector<float> cref;

    run_algo(algo, a, b, c, op, param, post_op, N, batch_size, repeat);
    run_ref(a, b, cref, op, param, post_op, N);

    compare(c, cref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"binary_batch", Algo::BINARY_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::BINARY_BATCH:
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
    case Algo::BINARY_BATCH:
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
    case Algo::BINARY_BATCH:
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
    fprintf(stderr, "    binary_batch\n");
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
        for (BinaryOp op: op_config) {
        for (BinaryParam &param: param_config) {
        for (base::PostOpSpec &post_op: post_op_config) {
            run(algo, op, param, post_op, N, batch_size, repeat);
        }
        }
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

