// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/unpack_tilize.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"

#include "test/ref/unpack_tilize_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

void run_algo(
        int H,
        int W,
        const std::vector<float> &x,
        std::vector<float> &y) {
    std::vector<uint16_t> tx = float_to_u16b(x);
    std::vector<uint16_t> ty(y.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::UnpackTilize solver;
    solver.init(device, H, W);
    solver.run(tx.data(), ty.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    y = u16b_to_float(ty);
}

void run_ref(
        int H,
        int W,
        const std::vector<float> &x,
        std::vector<float> &y) {
    UnpackTilizeRef solver;
    solver.init(H, W);
    solver.run(x.data(), y.data());
}

void run() {
    printf("---- Unpack tilize\n");
    int H = 8 * 32;
    int W = 16 * 32;
    int size = H * W;
    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, size);
    std::vector<float> y(size);
    std::vector<float> yref(size);
    run_algo(H, W, x, y);
    run_ref(H, W, x, yref);
    compare(y, yref);
}

} // namespace

void main_unpack_tilize() {
    run();
}

