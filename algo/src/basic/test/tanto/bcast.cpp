// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/bcast.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/tiles.hpp"

#include "test/ref/bcast_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

using tanto::BcastOp;
using tanto::BcastDim;

struct Param {
    BcastOp op;
    BcastDim dim;
};

std::vector<Param> param_config = {
    {BcastOp::Add, BcastDim::Rows},
    {BcastOp::Sub, BcastDim::Rows},
    {BcastOp::Mul, BcastDim::Rows},
    {BcastOp::Add, BcastDim::Cols},
    {BcastOp::Sub, BcastDim::Cols},
    {BcastOp::Mul, BcastDim::Cols},
    {BcastOp::Add, BcastDim::Scalar},
    {BcastOp::Sub, BcastDim::Scalar},
    {BcastOp::Mul, BcastDim::Scalar}
};

std::string op_to_str(BcastOp op) {
    switch (op) {
    case BcastOp::Add:
        return "add";
    case BcastOp::Sub:
        return "sub";
    case BcastOp::Mul:
        return "mul";
    default:
        assert(false);
        return "invalid";
    }
}

std::string dim_to_str(BcastDim dim) {
    switch (dim) {
    case BcastDim::Rows:
        return "rows";
    case BcastDim::Cols:
        return "cols";
    case BcastDim::Scalar:
        return "scalar";
    default:
        assert(false);
        return "invalid";
    }
}

BcastRefOp op_to_ref(BcastOp op) {
    switch (op) {
    case BcastOp::Add:
        return BcastRefOp::Add;
    case BcastOp::Sub:
        return BcastRefOp::Sub;
    case BcastOp::Mul:
        return BcastRefOp::Mul;
    default:
        assert(false);
        return BcastRefOp(0);
    }
}

BcastRefDim dim_to_ref(BcastDim dim) {
    switch (dim) {
    case BcastDim::Rows:
        return BcastRefDim::Rows;
    case BcastDim::Cols:
        return BcastRefDim::Cols;
    case BcastDim::Scalar:
        return BcastRefDim::Scalar;
    default:
        assert(false);
        return BcastRefDim(0);
    }
}

int get_bcast_size(
        BcastDim dim,
        int N,
        int C,
        int H,
        int W) {
    switch (dim) {
    case BcastDim::Rows:
        return N * C * W;
    case BcastDim::Cols:
        return N * C * H;
    case BcastDim::Scalar:
        return N * C;
    default:
        assert(false);
        return 0;
    }
}

std::vector<float> make_bcast_data(
        BcastDim dim,
        int N,
        int C,
        int H,
        int W) {
    int size = get_bcast_size(dim, N, C, H, W);
    std::vector<float> result(size);
    float v0 = 10.0f;
    for (int i = 0; i < size; i++) {
        // add something not too large but different between tiles
        result[i] = v0 + i % 7;
    }
    return result;
}

std::vector<float> expand_bcast_data_rows(
        const std::vector<float> &data,
        int N,
        int C,
        int W) {
    int size = int(data.size());
    assert(N * C * W == size);
    std::vector<float> result(size * 32, 0.0f);
    int NC = N * C;
    int p = 0;
    int q_start = 0;
    for (int nc = 0; nc < NC; nc++) {
        int q = q_start;
        for (int w = 0; w < W; w++) {
            result[q] = data[p];
            p++;
            q++;
        }
        q_start += W * 32;
    }
    return result;
}

std::vector<float> expand_bcast_data_cols(
        const std::vector<float> &data,
        int N,
        int C,
        int H) {
    int size = int(data.size());
    assert(N * C * H == size);
    std::vector<float> result(size * 32, 0.0f);
    int NC = N * C;
    int p = 0;
    int q = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < H; h++) {
            result[q] = data[p];
            p++;
            q += 32;
        }
    }
    return result;
}

std::vector<float> expand_bcast_data_scalar(
        const std::vector<float> &data,
        int N,
        int C) {
    int size = int(data.size());
    assert(N * C == size);
    std::vector<float> result(size * 1024, 0.0f);
    int NC = N * C;
    int p = 0;
    int q = 0;
    for (int nc = 0; nc < NC; nc++) {
        result[q] = data[p];
        p++;
        q += 1024;
    }
    return result;
}

std::vector<float> expand_bcast_data(
        const std::vector<float> &data,
        BcastDim dim,
        int N,
        int C,
        int H,
        int W) {
    switch (dim) {
    case BcastDim::Rows:
        return expand_bcast_data_rows(data, N, C, W);
    case BcastDim::Cols:
        return expand_bcast_data_cols(data, N, C, H);
    case BcastDim::Scalar:
        return expand_bcast_data_scalar(data, N, C);
    default:
        assert(false);
        return {};
    }
}

void run_algo(
        BcastOp op,
        BcastDim dim,
        int N,
        int C,
        int H,
        int W,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    std::vector<float> a2 = util::tilize(a, W / 32);
    std::vector<float> b2 = expand_bcast_data(b, dim, N, C, H, W);
    if (dim == BcastDim::Rows) {
        b2 = util::tilize(b2, W / 32);
    } else {
        b2 = util::tilize(b2, 1);
    }
    std::vector<uint16_t> ta = float_to_u16b(a2);
    std::vector<uint16_t> tb = float_to_u16b(b2);
    std::vector<uint16_t> tc(c.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::Bcast solver;
    solver.init(device, op, dim, N, C, H, W);
    solver.run(ta.data(), tb.data(), tc.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    c = u16b_to_float(tc);
    c = util::untilize(c, W / 32);
}

void run_ref(
        BcastRefOp op,
        BcastRefDim dim,
        int N,
        int C,
        int H,
        int W,
        const std::vector<float> &a,
        const std::vector<float> &b,
        std::vector<float> &c) {
    BcastRef solver;
    solver.init(op, dim, N, C, H, W);
    solver.run(a.data(), b.data(), c.data());
}

void run(const Param &param) {
    BcastOp op = param.op;
    BcastDim dim = param.dim;
    std::string str_op = op_to_str(op);
    std::string str_dim = dim_to_str(dim);
    printf("---- Op %s dim %s\n", str_op.c_str(), str_dim.c_str());
    int N = 2;
    int C = 4;
    int H = 64;
    int W = 96;
    int NCHW = N * C * H * W;
    BcastRefOp ref_op = op_to_ref(op);
    BcastRefDim ref_dim = dim_to_ref(dim);
    util::manual_seed(1234);
    std::vector<float> a = util::normal(0.0f, 10.0f, NCHW);
    std::vector<float> b = make_bcast_data(dim, N, C, H, W);
    std::vector<float> c(NCHW);
    std::vector<float> cref(NCHW);
    run_algo(op, dim, N, C, H, W, a, b, c);
    run_ref(ref_op, ref_dim, N, C, H, W, a, b, cref);
    compare(c, cref);
}

} // namespace

void main_bcast() {
    for (const Param &param: param_config) {
        run(param);
    }
}

