// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/transpose_wh.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/tiles.hpp"

#include "test/ref/transpose_wh_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

void run_algo(
        int N,
        int H,
        int W,
        const std::vector<float> &x,
        std::vector<float> &y) {
    std::vector<float> x2 = util::tilize(x, W / 32);
    std::vector<uint16_t> tx = float_to_u16b(x2);
    std::vector<uint16_t> ty(y.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::TransposeWh solver;
    solver.init(device, N, H, W);
    solver.run(tx.data(), ty.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    y = u16b_to_float(ty);
    y = util::untilize(y, H / 32);
}

void run_ref(
        int N,
        int H,
        int W,
        const std::vector<float> &x,
        std::vector<float> &y) {
    TransposeWhRef solver;
    solver.init(N, H, W);
    solver.run(x.data(), y.data());
}

void run() {
    printf("---- Transpose WH\n");
    int N = 3;
    int H = 3 * 32;
    int W = 4 * 32;
    int size = N * H * W;
    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, size);
    std::vector<float> y(size);
    std::vector<float> yref(size);
    run_algo(N, H, W, x, y);
    run_ref(N, H, W, x, yref);
    compare(y, yref);
}

} // namespace

void main_transpose_wh() {
    run();
}

