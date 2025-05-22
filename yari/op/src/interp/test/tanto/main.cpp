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

#include "host/tanto/interp2d_linear_batch.hpp"

#include "host/ref/interp2d_linear_ref.hpp"

#include "host/util/transform.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/timer.hpp"

using namespace ronin::op::interp;
using namespace ronin::op::common::util;
using namespace ronin::op::common::test;

namespace core = ronin::tanto::host;

enum class Algo {
    INTERP2D_LINEAR_BATCH
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

const char *coord_transform_mode_str(tanto::CoordTransformMode mode) {
    switch (mode) {
    case tanto::CoordTransformMode::HALF_PIXEL:
        return "half_pixel";
    case tanto::CoordTransformMode::PYTORCH_HALF_PIXEL:
        return "pytorch_half_pixel";
    case tanto::CoordTransformMode::ASYMMETRIC:
        return "asymmetric";
    case tanto::CoordTransformMode::TF_HALF_PIXEL_FOR_NN:
        return "tf_half_pixel_for_nn";
    case tanto::CoordTransformMode::ALIGN_CORNERS:
        return "align_corners";
    default:
        assert(false);
        return "?";
    }
}

ref::CoordTransformMode coord_transform_mode_ref(tanto::CoordTransformMode mode) {
    switch (mode) {
    case tanto::CoordTransformMode::HALF_PIXEL:
        return ref::CoordTransformMode::HALF_PIXEL;
    case tanto::CoordTransformMode::PYTORCH_HALF_PIXEL:
        return ref::CoordTransformMode::PYTORCH_HALF_PIXEL;
    case tanto::CoordTransformMode::ASYMMETRIC:
        return ref::CoordTransformMode::ASYMMETRIC;
    case tanto::CoordTransformMode::TF_HALF_PIXEL_FOR_NN:
        return ref::CoordTransformMode::TF_HALF_PIXEL_FOR_NN;
    case tanto::CoordTransformMode::ALIGN_CORNERS:
        return ref::CoordTransformMode::ALIGN_CORNERS;
    default:
        assert(false);
        return ref::CoordTransformMode(0);
    }
}

struct Interp2dLinearParam {
    int H;
    int W;
    int C;
    int P;
    int Q;
    float scale_h;
    float scale_w;
    tanto::CoordTransformMode coord_transform_mode;
};

using tanto::CoordTransformMode::HALF_PIXEL;

std::vector<Interp2dLinearParam> param_config {
    // H, W, C, P, Q, scale_h, scale_w, coord_transform_mode
    {7, 7, 128, 28, 28, 4.0f, 4.0f, HALF_PIXEL},
    {28, 28, 21, 224, 224, 8.0f, 8.0f, HALF_PIXEL}
};

template<typename SOLVER>
void run_interp2d_linear(
        const std::vector<float> &x,
        std::vector<float> &y,
        const Interp2dLinearParam &param,
        int N,
        int batch_size,
        int repeat) {
    SOLVER solver(
        N, 
        param.H, 
        param.W,
        param.C,
        param.P, 
        param.Q,
        param.scale_h,
        param.scale_w,
        param.coord_transform_mode,
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

void run_interp2d_linear_batch(
        const std::vector<float> &x,
        std::vector<float> &y,
        const Interp2dLinearParam &param,
        int N,
        int batch_size,
        int repeat) {
    run_interp2d_linear<tanto::Interp2dLinearBatch>(x, y, param, N, batch_size, repeat);
}

void run_interp2d_linear_ref(
        const std::vector<float> &x,
        std::vector<float> &y,
        const Interp2dLinearParam &param,
        int N) {
    y.resize(N * param.P * param.Q * param.C);
    ref::Interp2dLinearRef solver(
        N, 
        param.H, 
        param.W,
        param.C,
        param.P, 
        param.Q,
        param.scale_h,
        param.scale_w,
        coord_transform_mode_ref(param.coord_transform_mode));
    solver.init(x.data(), y.data());
    solver.run();
}

void run_interp2d_linear_algo(
        Algo algo,
        const std::vector<float> &x,
        std::vector<float> &y,
        const Interp2dLinearParam &param,
        int N,
        int batch_size,
        int repeat) {
    switch (algo) {
    case Algo::INTERP2D_LINEAR_BATCH:
        run_interp2d_linear_batch(x, y, param, N, batch_size, repeat);
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

void run_interp2d_linear(
        Algo algo, 
        const Interp2dLinearParam &param, 
        int N, 
        int batch_size,
        int repeat) {
    printf("---- Batch [%d / %d] HW [%d %d] C [%d] PQ [%d %d] scale [%g %g] %s\n",
        N, batch_size, param.H, param.W, param.C, param.P, param.Q,
        param.scale_h, param.scale_w, coord_transform_mode_str(param.coord_transform_mode));

    int xsize = N * param.H * param.W * param.C;
    int ysize = N * param.P * param.Q * param.C;

    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, xsize);

    std::vector<float> y;
    std::vector<float> yref;

    run_interp2d_linear_algo(algo, x, y, param, N, batch_size, repeat);
    run_interp2d_linear_ref(x, yref, param, N);

    compare(y, yref);
}

std::unordered_map<std::string, Algo> str_algo_map = {
    {"interp2d_linear_batch", Algo::INTERP2D_LINEAR_BATCH}
};

bool validate_args(Algo algo, int N, int batch_size) {
    if (N % batch_size != 0) {
        return false;
    }
    switch (algo) {
    case Algo::INTERP2D_LINEAR_BATCH:
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
    case Algo::INTERP2D_LINEAR_BATCH:
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
    case Algo::INTERP2D_LINEAR_BATCH:
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
    fprintf(stderr, "    interp2d_linear_batch\n");
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
        for (Interp2dLinearParam &param: param_config) {
            run_interp2d_linear(algo, param, N, batch_size, repeat);
        }
    } catch (std::exception &e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}

