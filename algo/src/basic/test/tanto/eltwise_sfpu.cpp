// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>

#include "host/core/api.hpp"

#include "host/tanto/eltwise_sfpu.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"

#include "test/ref/eltwise_sfpu_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

using tanto::EltwiseSfpuOp;

struct Param {
    EltwiseSfpuOp op;
    uint32_t iparam;
    float fparam;
};

std::vector<Param> param_config = {
    {EltwiseSfpuOp::Exp, 0, 0.0f},
    {EltwiseSfpuOp::Gelu, 0, 0.0f},
    {EltwiseSfpuOp::Log, 0, 0.0f},
    {EltwiseSfpuOp::Recip, 0, 0.0f},
    {EltwiseSfpuOp::Relu, 0, 0.0f},
    {EltwiseSfpuOp::Sigmoid, 0, 0.0f},
    {EltwiseSfpuOp::Sqrt, 0, 0.0f},
    {EltwiseSfpuOp::Tanh, 0, 0.0f}
};

std::unordered_map<EltwiseSfpuOp, std::string> op_str_map = {
    {EltwiseSfpuOp::Abs, "abs"},
    {EltwiseSfpuOp::Acos, "acos"},
    {EltwiseSfpuOp::Asin, "asin"},
    {EltwiseSfpuOp::Atan, "atan"},
    {EltwiseSfpuOp::Cos, "cos"},
    {EltwiseSfpuOp::Elu, "elu"},
    {EltwiseSfpuOp::Eqz, "eqz"},
    {EltwiseSfpuOp::Erf, "erf"},
    {EltwiseSfpuOp::Erfc, "erfc"},
    {EltwiseSfpuOp::Erfinv, "erfinv"},
    {EltwiseSfpuOp::Exp, "exp"},
    {EltwiseSfpuOp::Exp2, "exp2"},
    {EltwiseSfpuOp::Expm1, "expm1"},
    {EltwiseSfpuOp::Gelu, "gelu"},
    {EltwiseSfpuOp::Gez, "gez"},
    {EltwiseSfpuOp::Gtz, "gtz"},
    {EltwiseSfpuOp::Heaviside, "heaviside"},
    {EltwiseSfpuOp::I0, "i0"},
    {EltwiseSfpuOp::Isfinite, "isfinite"},
    {EltwiseSfpuOp::Isinf, "isinf"},
    {EltwiseSfpuOp::Isnan, "isnan"},
    {EltwiseSfpuOp::Isneginf, "isneginf"},
    {EltwiseSfpuOp::Isposinf, "isposinf"},
    {EltwiseSfpuOp::LeakyRelu, "leaky_relu"},
    {EltwiseSfpuOp::Lez, "lez"},
    {EltwiseSfpuOp::Log, "log"},
    {EltwiseSfpuOp::LogWithBase, "log_with_base"},
    {EltwiseSfpuOp::LogicalNot, "logical_not"},
    {EltwiseSfpuOp::Ltz, "ltz"},
    {EltwiseSfpuOp::Nez, "nez"},
    {EltwiseSfpuOp::Power, "power"},
    {EltwiseSfpuOp::Recip, "recip"},
    {EltwiseSfpuOp::Relu, "relu"},
    {EltwiseSfpuOp::ReluMax, "relu_max"},
    {EltwiseSfpuOp::ReluMin, "relu_min"},
    {EltwiseSfpuOp::Rsqrt, "rsqrt"},
    {EltwiseSfpuOp::Sigmoid, "sigmoid"},
    {EltwiseSfpuOp::Sign, "sign"},
    {EltwiseSfpuOp::Signbit, "signbit"},
    {EltwiseSfpuOp::Sin, "sin"},
    {EltwiseSfpuOp::Sqrt, "sqrt"},
    {EltwiseSfpuOp::Square, "square"},
    {EltwiseSfpuOp::Tan, "tan"},
    {EltwiseSfpuOp::Tanh, "tanh"}
};

std::string op_to_str(EltwiseSfpuOp op) {
    auto it = op_str_map.find(op);
    if (it == op_str_map.end()) {
        assert(false);
        return "invalid";
    }
    return it->second;
}

EltwiseSfpuRefOp op_to_ref(EltwiseSfpuOp op) {
    // both enums must have identical member lists
    return EltwiseSfpuRefOp(op);
}

void run_algo(
        EltwiseSfpuOp op,
        uint32_t iparam,
        float fparam,
        int N,
        const std::vector<float> &x,
        std::vector<float> &y) {
    std::vector<uint16_t> tx = float_to_u16b(x);
    std::vector<uint16_t> ty(y.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::EltwiseSfpu solver;
    solver.init(device, op, iparam, fparam, N);
    solver.run(tx.data(), ty.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    y = u16b_to_float(ty);
}

void run_ref(
        EltwiseSfpuRefOp op,
        uint32_t iparam,
        float fparam,
        int N,
        const std::vector<float> &x,
        std::vector<float> &y) {
    EltwiseSfpuRef solver;
    solver.init(op, iparam, fparam, N);
    solver.run(x.data(), y.data());
}

void run(const Param &param) {
    EltwiseSfpuOp op = param.op;
    uint32_t iparam = param.iparam;
    float fparam = param.fparam;
    std::string str_op = op_to_str(op);
    printf("---- Op %s iparam %d fparam %g\n", str_op.c_str(), iparam, fparam);
    int N = 1024 * 1024;
    util::manual_seed(1234);
    std::vector<float> x;
    switch (op) {
    case EltwiseSfpuOp::Sqrt:
    case EltwiseSfpuOp::Log:
        x = util::uniform(0.0001f, 4.0f, N);
        break;
    case EltwiseSfpuOp::Exp:
    case EltwiseSfpuOp::Gelu:
    case EltwiseSfpuOp::Recip:
        x = util::uniform_from_vector({-1.0f, -0.5f, 0.5f, 1.0f}, N);
        break;
    default:
        x = util::uniform(-1.0f, 1.0f, N);
        break;
    }
    std::vector<float> y(N);
    std::vector<float> yref(N);
    EltwiseSfpuRefOp ref_op = op_to_ref(op);
    run_algo(op, iparam, fparam, N, x, y);
    run_ref(ref_op, iparam, fparam, N, x, yref);
    compare(y, yref);
}

} // namespace

void main_eltwise_sfpu() {
    for (const Param &param: param_config) {
        run(param);
    }
}

