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

#include "host/tanto/conv2d_basic_batch.hpp"
#include "host/tanto/conv2d_basic_split.hpp"
#include "host/tanto/conv2d_basic_spatial.hpp"
#include "host/tanto/conv2d_image_batch.hpp"

#include "host/ref/conv2d_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

//#define MOBILENETV2_050

using namespace ronin::op::conv;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;
namespace base = ronin::op::common::base;

enum class Algo {
    BASIC_BATCH,
    BASIC_SPLIT,
    BASIC_SPATIAL,
    IMAGE_BATCH
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

struct ConvParam {
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

#if defined MOBILENETV2_050
std::vector<ConvParam> basic_param_config = {
    // H, W, C, P, Q, K, R, S, stride_h, stride_w, pad_h, pad_w
    // MobileNetV2_050 conv layers
    {112, 112, 16, 112, 112, 8, 1, 1, 1, 1, 0, 0},    // conv5
    {112, 112, 8, 112, 112, 48, 1, 1, 1, 1, 0, 0},    // conv6
    {56, 56, 48, 56, 56, 16, 1, 1, 1, 1, 0, 0},       // conv10
    {56, 56, 16, 56, 56, 96, 1, 1, 1, 1, 0, 0},       // conv11
    {56, 56, 96, 56, 56, 16, 1, 1, 1, 1, 0, 0},       // conv15
    {28, 28, 96, 28, 28, 16, 1, 1, 1, 1, 0, 0},       // conv21
    {28, 28, 16, 28, 28, 96, 1, 1, 1, 1, 0, 0},       // conv22
    {28, 28, 96, 28, 28, 16, 1, 1, 1, 1, 0, 0},       // conv26
    {14, 14, 96, 14, 14, 32, 1, 1, 1, 1, 0, 0},       // conv38
    {14, 14, 32, 14, 14, 192, 1, 1, 1, 1, 0, 0},      // conv39
    {14, 14, 192, 14, 14, 32, 1, 1, 1, 1, 0, 0},      // conv43
    {14, 14, 192, 14, 14, 48, 1, 1, 1, 1, 0, 0},      // conv61
    {14, 14, 48, 14, 14, 288, 1, 1, 1, 1, 0, 0},      // conv62
    {14, 14, 288, 14, 14, 48, 1, 1, 1, 1, 0, 0},      // conv66
    {7, 7, 288, 7, 7, 80, 1, 1, 1, 1, 0, 0},          // conv78
    {7, 7, 80, 7, 7, 480, 1, 1, 1, 1, 0, 0},          // conv79
    {7, 7, 480, 7, 7, 80, 1, 1, 1, 1, 0, 0},          // conv83
    {7, 7, 480, 7, 7, 160, 1, 1, 1, 1, 0, 0},         // conv95
    {7, 7, 160, 7, 7, 1280, 1, 1, 1, 1, 0, 0}         // conv96
};

#else
std::vector<ConvParam> basic_param_config = {
    // H, W, C, P, Q, K, R, S, stride_h, stride_w, pad_h, pad_w
    // ResNet18 conv layers
    {56, 56, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1},      // kernel03
    {56, 56, 64, 28, 28, 128, 1, 1, 2, 2, 0, 0},     // kernel05
    {56, 56, 64, 28, 28, 128, 3, 3, 2, 2, 1, 1},     // kernel06
    {28, 28, 128, 28, 28, 128, 3, 3, 1, 1, 1, 1},    // kernel07
    {28, 28, 128, 14, 14, 256, 3, 3, 2, 2, 1, 1},    // kernel09
    {14, 14, 256, 14, 14, 256, 3, 3, 1, 1, 1, 1},    // kernel10
    {28, 28, 128, 14, 14, 256, 1, 1, 2, 2, 0, 0},    // kernel11
    {14, 14, 256, 7, 7, 512, 3, 3, 2, 2, 1, 1},      // kernel14
    {7, 7, 512, 7, 7, 512, 3, 3, 1, 1, 1, 1},        // kernel15
    {14, 14, 256, 7, 7, 512, 1, 1, 2, 2, 0, 0}       // kernel16
};
#endif

std::vector<ConvParam> image_param_config = {
    // H, W, C, P, K, Q, R, S, stride_h, stride_w, pad_h, pad_w
    {224, 224, 3, 55, 55, 64, 11, 11, 4, 4, 2, 2},    // AlexNet
    {224, 224, 3, 112, 112, 64, 7, 7, 2, 2, 3, 3},    // DenseNet, GoogleNet, ResNet
    {299, 299, 3, 149, 149, 32, 3, 3, 2, 2, 0, 0},    // InceptionV3
    {224, 224, 3, 112, 112, 16, 3, 3, 2, 2, 1, 1},    // MnasNet 0.5
    {224, 224, 3, 112, 112, 32, 3, 3, 2, 2, 1, 1},    // MnasNet 1.0, MobileNetV2
    {224, 224, 3, 112, 112, 24, 3, 3, 2, 2, 1, 1},    // ShuffleNetV2
    {224, 224, 3, 109, 109, 96, 7, 7, 2, 2, 0, 0},    // SqueezeNet v1.0
    {224, 224, 3, 111, 111, 64, 3, 3, 2, 2, 0, 0},    // SqueezeNet v1.1
    {224, 224, 3, 224, 224, 64, 3, 3, 1, 1, 1, 1}     // VGG
};

struct ConvOpt {
    bool bias;
    bool add;
    base::PostOpSpec post_op;
};

base::PostOpSpec noop(base::PostOp::NONE);
base::PostOpSpec relu(base::PostOp::RELU);

std::vector<ConvOpt> basic_opt_config = {
    {true, false, noop},
    {true, false, relu},
    {true, true, relu}
};

std::vector<ConvOpt> image_opt_config = {
    {true, false, noop},
    {true, false, relu}
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
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
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
    std::vector<uint16_t> tb;
    if (opt.bias) {
        tb = float_to_u16b(solver.transform_input(2, b));
    }
    std::vector<uint16_t> tz;
    if (opt.add) {
        tz = float_to_u16b(solver.transform_input(3, z));
    }
    std::vector<uint16_t> ty(solver.output_volume(0));
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    core::DataFormat T = core::DataFormat::BFLOAT16;
    uint32_t log2_page_size = 10; // 2 ^ 10 = 1024
    core::Global gx(device, T, solver.input_volume(0), log2_page_size);
    core::Global gw(device, T, solver.input_volume(1), log2_page_size);
    core::Global gb;
    if (opt.bias) {
        gb = core::Global(device, T, solver.input_volume(2), log2_page_size);
    }
    core::Global gz;
    if (opt.add) {
        gz = core::Global(device, T, solver.input_volume(3), log2_page_size);
    }
    core::Global gy(device, T, solver.output_volume(0), log2_page_size);
    solver.init(
        device,
        gx,
        gw,
        gb,
        gz,
        gy);
    core::Queue queue(device, 0);
    queue.enqueue_write(gx, tx.data(), false);
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
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_conv<tanto::Conv2dBasicBatch>(x, w, b, z, y, opt, param, N, batch_size, repeat);
}

void run_basic_split(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_conv<tanto::Conv2dBasicSplit>(x, w, b, z, y, opt, param, N, batch_size, repeat);
}

void run_basic_spatial(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_conv<tanto::Conv2dBasicSpatial>(x, w, b, z, y, opt, param, N, batch_size, repeat);
}

void run_image_batch(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    std::vector<float> w2 = reorder_rskc_to_krsc(w, param.C, param.K, param.R, param.S);
    run_conv<tanto::Conv2dImageBatch>(x, w2, b, z, y, opt, param, N, batch_size, repeat);
}

void run_ref(
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N) {
    y.resize(N * param.P * param.Q * param.K);
    int dilation_h = 1;
    int dilation_w = 1;
    ref::Conv2dRef solver(
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
        opt.bias ? b.data() : nullptr,
        opt.add ? z.data() : nullptr,
        y.data());
    solver.run();
}

void run_algo(
        Algo algo,
        const std::vector<float> &x,
        const std::vector<float> &w,
        const std::vector<float> &b,
        const std::vector<float> &z,
        std::vector<float> &y,
        const ConvOpt &opt,
        const ConvParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::BASIC_BATCH:
        run_basic_batch(x, w, b, z, y, opt, param, N, batch_size, repeat);
        break;
    case Algo::BASIC_SPLIT:
        run_basic_split(x, w, b, z, y, opt, param, N, batch_size, repeat);
        break;
    case Algo::BASIC_SPATIAL:
        run_basic_spatial(x, w, b, z, y, opt, param, N, batch_size, repeat);
        break;
    case Algo::IMAGE_BATCH:
        run_image_batch(x, w, b, z, y, opt, param, N, batch_size, repeat);
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
        const ConvOpt &opt,
        const ConvParam &param, 
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
    int wsize = param.K * param.R * param.S * param.C;
    int ysize = N * param.P * param.Q * param.K;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);
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

    run_algo(algo, x, w, b, z, y, opt, param, N, batch_size, repeat);
    run_ref(x, w, b, z, yref, opt, param, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"basic_batch", Algo::BASIC_BATCH},
    {"basic_split", Algo::BASIC_SPLIT},
    {"basic_spatial", Algo::BASIC_SPATIAL},
    {"image_batch", Algo::IMAGE_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::BASIC_BATCH:
    case Algo::IMAGE_BATCH:
        if (batch_size > 8 && 
                batch_size != 16 && 
                batch_size != 32 && 
                batch_size != 64) {
            return false;
        }
        break;
    case Algo::BASIC_SPLIT:
        if (batch_size != 8 && batch_size != 16) {
            return false;
        }
        break;
    case Algo::BASIC_SPATIAL:
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
    case Algo::BASIC_SPLIT:
    case Algo::BASIC_SPATIAL:
    case Algo::IMAGE_BATCH:
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
    case Algo::IMAGE_BATCH:
        batch_size = (N > 64) ? 64 : N;
        break;
    case Algo::BASIC_SPLIT:
    case Algo::BASIC_SPATIAL:
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
    fprintf(stderr, "    basic_split\n");
    fprintf(stderr, "    basic_spatial\n");
    fprintf(stderr, "    image_batch\n");
    fprintf(stderr, "\n");
}

void run_basic(Algo algo, int N, int batch_size, int repeat) {
    for (ConvOpt &opt: basic_opt_config) {
        for (ConvParam &param: basic_param_config) {
            if (algo == Algo::BASIC_SPLIT) {
                // ACHTUNG: Limit of 64 cores is Wormhole-specific
                int block_size = 64 / batch_size;
                if (param.K % (block_size * 32) != 0) {
                    continue;
                }
                if (param.C * param.K * param.R * param.S > (256 + 32) * 1024 * block_size) {
printf("@@@ SKIP C %d K %d R %d S %d\n", param.C, param.K, param.R, param.S);
                    continue;
                }
            }
            run(algo, opt, param, N, batch_size, repeat);
        }
    }
}

void run_image(Algo algo, int N, int batch_size, int repeat) {
    for (ConvOpt &opt: image_opt_config) {
        for (ConvParam &param: image_param_config) {
            run(algo, opt, param, N, batch_size, repeat);
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
        switch (algo) {
        case Algo::BASIC_BATCH:
        case Algo::BASIC_SPLIT:
        case Algo::BASIC_SPATIAL:
            run_basic(algo, N, batch_size, repeat);
            break;
        case Algo::IMAGE_BATCH:
            run_image(algo, N, batch_size, repeat);
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

