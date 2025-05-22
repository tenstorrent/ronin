// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/matmul_single.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/tiles.hpp"

#include "test/ref/matmul_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

void run_algo(
        int batch,
        int M,
        int N,
        int K,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    std::vector<float> a2 = util::tilize(a, K / 32);
    std::vector<float> b2 = util::tilize(b, N / 32);
    std::vector<uint16_t> ta = float_to_u16b(a2);
    std::vector<uint16_t> tb = float_to_u16b(b2);
    std::vector<uint16_t> tc(c.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::MatmulSingle solver;
    solver.init(device, batch, M, N, K);
    solver.run(ta.data(), tb.data(), tc.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    c = u16b_to_float(tc);
    c = util::untilize(c, N / 32);
}

void run_ref(
        int batch,
        int M,
        int N,
        int K,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    MatmulRef solver;
    solver.init(batch, M, N, K);
    solver.run(a.data(), b.data(), c.data());
}

void run() {
    printf("---- Matmul single\n");
    int batch = 4;
    int M = 10 * 32;
    int N = 20 * 32;
    int K = 30 * 32;
    util::manual_seed(1234);
    std::vector<float> a = util::normal(0.0f, 0.1f, batch * M * K);
    std::vector<float> b = util::normal(0.0f, 0.1f, batch * K * N);
    std::vector<float> c(batch * M * N);
    std::vector<float> cref(batch * M * N);
    run_algo(batch, M, N, K, a, b, c);
    run_ref(batch, M, N, K, a, b, cref);
    compare(c, cref);
}

} // namespace

void main_matmul_single() {
    run();
}

