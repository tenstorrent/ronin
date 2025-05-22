// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <vector>

#include "host/core/api.hpp"

#include "host/base/post_op.hpp"

#include "host/tanto/ds_conv2d_batch.hpp"

#include "host/ref/ds_conv2d_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

#include "test/tanto/run.hpp"

namespace {

using namespace ronin::op::group_conv;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

struct DSConvParam {
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

std::vector<DSConvParam> param_config = {
    // H, W, C, P, Q, K, R, S, stride_h, stride_w, pad_h, pad_w
    // MobileNetV2_050 DW + PW layers
    {112, 112, 16, 112, 112, 8, 3, 3, 1, 1, 1, 1}, // conv3 + conv5
    {112, 112, 48, 56, 56, 16, 3, 3, 2, 2, 1, 1},  // conv8 + conv10
    {56, 56, 96, 56, 56, 16, 3, 3, 1, 1, 1, 1},    // conv13 + conv15
    {56, 56, 96, 28, 28, 16, 3, 3, 2, 2, 1, 1},    // conv19 + conv21
    {28, 28, 96, 28, 28, 16, 3, 3, 1, 1, 1, 1},    // conv24 + conv26
    {28, 28, 96, 14, 14, 32, 3, 3, 2, 2, 1, 1},    // conv36 + conv38
    {14, 14, 192, 14, 14, 32, 3, 3, 1, 1, 1, 1},   // conv41 + conv43
    {14, 14, 288, 14, 14, 48, 3, 3, 1, 1, 1, 1},   // conv64 + conv66
    {14, 14, 288, 7, 7, 80, 3, 3, 2, 2, 1, 1},     // conv76 + conv78
    {7, 7, 480, 7, 7, 80, 3, 3, 1, 1, 1, 1}        // conv81 + conv83
};

struct DSConvOpt {
    bool add;
    base::PostOpSpec post_op;
};

base::PostOpSpec relu6(base::PostOp::CLIP, 0.0f, 6.0f);

std::vector<DSConvOpt> opt_config = {
    {false, relu6},
    {true, relu6}
};

template<typename SOLVER>
void run_conv(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &w2,
        const std::vector<float> &b2,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DSConvOpt &opt,
        const DSConvParam &param,
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
        param.K,
        param.R,
        param.S,
        param.pad_h,
        param.pad_w,
        param.stride_h,
        param.stride_w,
        dilation_h,
        dilation_w,
        opt.post_op,
        batch_size);
    std::vector<uint16_t> tx = float_to_u16b(solver.transform_input(0, x));
    std::vector<uint16_t> tw = float_to_u16b(solver.transform_input(1, w));
    std::vector<uint16_t> tb = float_to_u16b(solver.transform_input(2, b));
    std::vector<uint16_t> tw2 = float_to_u16b(solver.transform_input(3, w2));
    std::vector<uint16_t> tb2 = float_to_u16b(solver.transform_input(4, b2));
    std::vector<uint16_t> tz;
    if (opt.add) {
        tz = float_to_u16b(solver.transform_input(5, z));
    }
    std::vector<uint16_t> ty(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, solver.input_volume(0), log2_page_size);
    core::Global gw(device, T, solver.input_volume(1), log2_page_size);
    core::Global gb(device, T, solver.input_volume(2), log2_page_size);
    core::Global gw2(device, T, solver.input_volume(3), log2_page_size);
    core::Global gb2(device, T, solver.input_volume(4), log2_page_size);
    core::Global gz;
    if (opt.add) {
        gz = core::Global(device, T, solver.input_volume(5), log2_page_size);
    }
    core::Global gy(device, T, solver.output_volume(0), log2_page_size);
    solver.init(
        device,
        gx,
        gw,
        gb,
        gw2,
        gb2,
        gz,
        gy);
    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
    queue.enqueue_write(gw, tw.data(), false);
    queue.enqueue_write(gb, tb.data(), false);
    queue.enqueue_write(gw2, tw2.data(), false);
    queue.enqueue_write(gb2, tb2.data(), false);
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

void run_batch(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &w2,
        const std::vector<float> &b2,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DSConvOpt &opt,
        const DSConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_conv<tanto::DSConv2dBatch>(x, w, b, w2, b2, z, y, opt, param, N, batch_size, repeat);
}

void run_ref(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &w2,
        const std::vector<float> &b2,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DSConvOpt &opt,
        const DSConvParam &param,
        int N) {
    y.resize(N * param.P * param.Q * param.K);
    int dilation_h = 1;
    int dilation_w = 1;
    ref::DSConv2dRef solver(
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
        opt.post_op);
    solver.init(
        x.data(),
        w.data(),
        b.data(),
        w2.data(),
        b2.data(),
        opt.add ? z.data() : nullptr,
        y.data());
    solver.run();
}

void run_algo(
        Algo algo,
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &w2,
        const std::vector<float> &b2,
        const std::vector<float> &z,
        std::vector<float> &y,
        const DSConvOpt &opt,
        const DSConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::DSC_BATCH:
        run_batch(x, w, b, w2, b2, z, y, opt, param, N, batch_size, repeat);
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
        const DSConvOpt &opt,
        const DSConvParam &param, 
        int N, 
        int batch_size,
        int repeat) {
    printf(
        "---- Batch [%d / %d] HWC [%d %d %d] PQK [%d %d %d] "
        "RS [%d %d] pad [%d %d] stride [%d %d] add [%d] %s\n",
            N, batch_size,
            param.H, param.W, param.C, 
            param.P, param.Q, param.K, 
            param.R, param.S, 
            param.pad_h, param.pad_w, 
            param.stride_h, param.stride_w,
            int(opt.add), opt.post_op.str().c_str());

    int xsize = N * param.H * param.W * param.C;
    int wsize = param.R * param.S * param.C;
    int w2size = param.K * param.C;
    int ysize = N * param.P * param.Q * param.K;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);
    std::vector<float> w = util::normal(0.0f, 0.1f, wsize);
    std::vector<float> b = util::normal(0.0f, 0.1f, param.C);
    std::vector<float> w2 = util::normal(0.0f, 0.1f, w2size);
    std::vector<float> b2 = util::normal(0.0f, 0.1f, param.K);
    std::vector<float> z;
    if (opt.add) {
        z = util::normal(0.0f, 0.1f, ysize);
    }

    std::vector<float> y;
    std::vector<float> yref;

    run_algo(algo, x, w, b, w2, b2, z, y, opt, param, N, batch_size, repeat);
    run_ref(x, w, b, w2, b2, z, yref, opt, param, N);

    compare(y, yref);
}

} // namespace

void run_dsc(Algo algo, int N, int batch_size, int repeat) {
    for (DSConvOpt &opt: opt_config) {
        for (DSConvParam &param: param_config) {
            run(algo, opt, param, N, batch_size, repeat);
        }
    }
}

