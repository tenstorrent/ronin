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

#include "host/tanto/deform_conv2d_basic_batch.hpp"

#include "host/ref/deform_conv2d_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::deform_conv;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

enum class Algo {
    BASIC_BATCH,
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

struct DeformConvParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    int K;
    int R;
    int S;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
};

std::vector<DeformConvParam> param_config = {
    // H, W, C, P, Q, K, R, S, stride_h, stride_w, pad_h, pad_w
    // ResNet50 top conv layers
    {14, 14, 256, 14, 14, 1024, 1, 1, 1, 1, 0, 0},    // kernel25
    {14, 14, 512, 14, 14, 1024, 1, 1, 1, 1, 0, 0},    // kernel26
    {14, 14, 1024, 14, 14, 256, 1, 1, 1, 1, 0, 0},    // kernel27
    {14, 14, 256, 7, 7, 256, 3, 3, 2, 2, 1, 1},       // kernel29
    {7, 7, 256, 7, 7, 1024, 1, 1, 1, 1, 0, 0},        // kernel30
    {7, 7, 1024, 7, 7, 512, 1, 1, 1, 1, 0, 0},        // kernel33
    {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1},         // kernel34
    {7, 7, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0},        // kernel35
    {7, 7, 1024, 7, 7, 2048, 1, 1, 1, 1, 0, 0},       // kernel36
    {7, 7, 2048, 7, 7, 512, 1, 1, 1, 1, 0, 0},        // kernel37
    {7, 7, 512, 7, 7, 2048, 1, 1, 1, 1, 0, 0}         // kernel38
};

struct DeformConvOpt {
    bool bias;
    bool add;
    base::PostOpSpec post_op;
};

base::PostOpSpec noop(base::PostOp::NONE);
//base::PostOpSpec relu(base::PostOp::RELU);

std::vector<DeformConvOpt> opt_config = {
    {true, false, noop},
//    {true, false, relu},
//    {true, true, relu}
};

std::vector<float> reorder_rskc_to_krsc(
        const std::vector<float> &x,
        int C,
        int K,
        int R, 
        int S) {
    std::vector<float> y(K * R * S * C);
    int KC = K * C;
    int RS = R * S;
    int iy = 0;
    for (int k = 0; k < K; k++) {
        for (int rs = 0; rs < RS; rs++) {
            for (int c = 0; c < C; c++) {
                int ix = rs * KC + k * C + c;
                y[iy] = x[ix];
                iy++;
            }
        }
    }
    return y;
}

template<typename SOLVER>
void run_conv(
        const std::vector<float> &x,
        const std::vector<float> &d,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DeformConvOpt &opt,
        const DeformConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    int dilation_h = 1;
    int dilation_w = 1;
    int deform_groups = 1;
    SOLVER solver(
        N,
        param.H,
        param.W,
        param.C,
        param.P,
        param.Q,
        param.K,
        param.R,
        param.S,
        param.pad_h,
        param.pad_w,
        param.stride_h,
        param.stride_w,
        dilation_h,
        dilation_w,
        deform_groups,
        opt.post_op,
        batch_size);
    std::vector<uint16_t> tx = float_to_u16b(solver.transform_input(0, x));
    std::vector<uint16_t> td = float_to_u16b(solver.transform_input(1, d));
    std::vector<uint16_t> tw = float_to_u16b(solver.transform_input(2, w));
    std::vector<uint16_t> tb;
    if (opt.bias) {
        tb = float_to_u16b(solver.transform_input(3, b));
    }
    std::vector<uint16_t> tz;
    if (opt.add) {
        tz = float_to_u16b(solver.transform_input(4, z));
    }
    std::vector<uint16_t> ty(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, solver.input_volume(0), log2_page_size);
    core::Global gd(device, T, solver.input_volume(1), log2_page_size);
    core::Global gw(device, T, solver.input_volume(2), log2_page_size);
    core::Global gb;
    if (opt.bias) {
        gb = core::Global(device, T, solver.input_volume(3), log2_page_size);
    }
    core::Global gz;
    if (opt.add) {
        gz = core::Global(device, T, solver.input_volume(4), log2_page_size);
    }
    core::Global gy(device, T, solver.output_volume(0), log2_page_size);
    solver.init(
        device,
        gx,
        gd,
        gw,
        gb,
        gz,
        gy);
    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
    queue.enqueue_write(gd, td.data(), false);
    queue.enqueue_write(gw, tw.data(), false);
    if (opt.bias) {
        queue.enqueue_write(gb, tb.data(), false);
    }
    if (opt.add) {
        queue.enqueue_write(gz, tz.data(), false);
    }
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

void run_basic_batch(
        const std::vector<float> &x,
        const std::vector<float> &d,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DeformConvOpt &opt,
        const DeformConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_conv<tanto::DeformConv2dBasicBatch>(x, d, w, b, z, y, opt, param, N, batch_size, repeat);
}

void run_ref(
        const std::vector<float> &x,
        const std::vector<float> &d,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DeformConvOpt &opt,
        const DeformConvParam &param,
        int N) {
    y.resize(N * param.P * param.Q * param.K);
    int dilation_h = 1;
    int dilation_w = 1;
    int deform_groups = 1;
    ref::DeformConv2dRef solver(
        N,
        param.H,
        param.W,
        param.C,
        param.P,
        param.Q,
        param.K,
        param.R,
        param.S,
        param.pad_h,
        param.pad_w,
        param.stride_h,
        param.stride_w,
        dilation_h,
        dilation_w,
        deform_groups,
        opt.post_op);
    solver.init(
        x.data(),
        d.data(),
        w.data(),
        opt.bias ? b.data() : nullptr,
        opt.add ? z.data() : nullptr,
        y.data());
    solver.run();
}

void run_algo(
        Algo algo,
        const std::vector<float> &x,
        const std::vector<float> &d,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DeformConvOpt &opt,
        const DeformConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::BASIC_BATCH:
        run_basic_batch(x, d, w, b, z, y, opt, param, N, batch_size, repeat);
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
        const DeformConvOpt &opt,
        const DeformConvParam &param, 
        int N, 
        int batch_size,
        int repeat) {
    printf(
        "---- Batch [%d / %d] HWC [%d %d %d] PQK [%d %d %d] "
        "RS [%d %d] pad [%d %d] stride [%d %d] bias [%d] add [%d] %s\n",
            N, batch_size,
            param.H, param.W, param.C, 
            param.P, param.Q, param.K, 
            param.R, param.S, 
            param.pad_h, param.pad_w, 
            param.stride_h, param.stride_w,
            int(opt.bias), int(opt.add), opt.post_op.str().c_str());

    int xsize = N * param.H * param.W * param.C;
    // assume deform_groups = 1
    int dsize = N * param.P * param.Q * param.R * param.S * 2;
    int wsize = param.K * param.R * param.S * param.C;
    int ysize = N * param.P * param.Q * param.K;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);
    std::vector<float> d = util::normal(0.0f, 5.0f, dsize);
    std::vector<float> w = util::normal(0.0f, 0.1f, wsize);
    std::vector<float> b;
    if (opt.bias) {
        b = util::normal(0.0f, 0.1f, param.K);
    }
    std::vector<float> z;
    if (opt.add) {
        z = util::normal(0.0f, 0.1f, ysize);
    }

    std::vector<float> y;
    std::vector<float> yref;

    run_algo(algo, x, d, w, b, z, y, opt, param, N, batch_size, repeat);
    run_ref(x, d, w, b, z, yref, opt, param, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"basic_batch", Algo::BASIC_BATCH},
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::BASIC_BATCH:
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
    case Algo::BASIC_BATCH:
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
    fprintf(stderr, "    basic_batch\n");
    fprintf(stderr, "\n");
}

void run(Algo algo, int N, int batch_size, int repeat) {
    for (DeformConvOpt &opt: opt_config) {
        for (DeformConvParam &param: param_config) {
            run(algo, opt, param, N, batch_size, repeat);
//return;
        }
    }
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
        run(algo, N, batch_size, repeat);
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

