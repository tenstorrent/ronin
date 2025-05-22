// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/eltwise_binary.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"

#include "test/ref/eltwise_binary_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

std::vector<tanto::EltwiseBinaryOp> config = {
    tanto::EltwiseBinaryOp::Add,
    tanto::EltwiseBinaryOp::Sub,
    tanto::EltwiseBinaryOp::Mul
};

std::string op_to_str(tanto::EltwiseBinaryOp op) {
    switch (op) {
    case tanto::EltwiseBinaryOp::Add:
        return "add";
    case tanto::EltwiseBinaryOp::Sub:
        return "sub";
    case tanto::EltwiseBinaryOp::Mul:
        return "mul";
    default:
        assert(false);
        return "invalid";
    }
}

EltwiseBinaryRefOp op_to_ref(tanto::EltwiseBinaryOp op) {
    switch (op) {
    case tanto::EltwiseBinaryOp::Add:
        return EltwiseBinaryRefOp::Add;
    case tanto::EltwiseBinaryOp::Sub:
        return EltwiseBinaryRefOp::Sub;
    case tanto::EltwiseBinaryOp::Mul:
        return EltwiseBinaryRefOp::Mul;
    default:
        assert(false);
        return EltwiseBinaryRefOp(0);
    }
}

void run_algo(
        tanto::EltwiseBinaryOp op,
        int N,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    std::vector<uint16_t> ta = float_to_u16b(a);
    std::vector<uint16_t> tb = float_to_u16b(b);
    std::vector<uint16_t> tc(c.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::EltwiseBinary solver;
    solver.init(device, op, N);
    solver.run(ta.data(), tb.data(), tc.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    c = u16b_to_float(tc);
}

void run_ref(
        EltwiseBinaryRefOp op,
        int N,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    EltwiseBinaryRef solver;
    solver.init(op, N);
    solver.run(a.data(), b.data(), c.data());
}

void run(tanto::EltwiseBinaryOp op) {
    std::string str_op = op_to_str(op);
    printf("---- Op %s\n", str_op.c_str());
    int N = 1024 * 1024;
    util::manual_seed(1234);
    std::vector<float> a = util::normal(0.0f, 0.1f, N);
    std::vector<float> b = util::normal(0.0f, 0.1f, N);
    std::vector<float> c(N);
    std::vector<float> cref(N);
    EltwiseBinaryRefOp ref_op = op_to_ref(op);
    run_algo(op, N, a, b, c);
    run_ref(ref_op, N, a, b, cref);
    compare(c, cref);
}

} // namespace

void main_eltwise_binary() {
    for (tanto::EltwiseBinaryOp op: config) {
        run(op);
    }
}

