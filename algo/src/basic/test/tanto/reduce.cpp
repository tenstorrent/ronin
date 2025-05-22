// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <string>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/reduce.hpp"

#include "test/util/gen.hpp"
#include "test/util/comp.hpp"
#include "test/util/tiles.hpp"

#include "test/ref/reduce_ref.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic;
using namespace ronin::algo::basic::test;
using namespace ronin::algo::basic::test::ref;

namespace core = ronin::tanto::host;

namespace {

using tanto::ReduceOp;
using tanto::ReduceDim;

struct Param {
    ReduceOp op;
    ReduceDim dim;
};

std::vector<Param> param_config = {
    {ReduceOp::Max, ReduceDim::Rows},
    {ReduceOp::Sum, ReduceDim::Rows},
    {ReduceOp::Max, ReduceDim::Cols},
    {ReduceOp::Sum, ReduceDim::Cols},
    {ReduceOp::Max, ReduceDim::Scalar},
    {ReduceOp::Sum, ReduceDim::Scalar}
};

std::string op_to_str(ReduceOp op) {
    switch (op) {
    case ReduceOp::Max:
        return "max";
    case ReduceOp::Sum:
        return "sum";
    default:
        assert(false);
        return "invalid";
    }
}

std::string dim_to_str(ReduceDim dim) {
    switch (dim) {
    case ReduceDim::Rows:
        return "rows";
    case ReduceDim::Cols:
        return "cols";
    case ReduceDim::Scalar:
        return "scalar";
    default:
        assert(false);
        return "invalid";
    }
}

ReduceRefOp op_to_ref(ReduceOp op) {
    switch (op) {
    case ReduceOp::Max:
        return ReduceRefOp::Max;
    case ReduceOp::Sum:
        return ReduceRefOp::Sum;
    default:
        assert(false);
        return ReduceRefOp(0);
    }
}

ReduceRefDim dim_to_ref(ReduceDim dim) {
    switch (dim) {
    case ReduceDim::Rows:
        return ReduceRefDim::Rows;
    case ReduceDim::Cols:
        return ReduceRefDim::Cols;
    case ReduceDim::Scalar:
        return ReduceRefDim::Scalar;
    default:
        assert(false);
        return ReduceRefDim(0);
    }
}

int get_output_size(
        ReduceDim dim,
        int N,
        int C,
        int H,
        int W) {
    switch (dim) {
    case ReduceDim::Rows:
        // reduce rows collapses W dimension
        return N * C * H * 32;
    case ReduceDim::Cols:
        // reduce cols collapses H dimension
        return N * C * 32 * W;
    case ReduceDim::Scalar:
        // reduce scalar collapses H and W dimensions
        return N * C * 1024;
    default:
        assert(false);
        return 0;
    }
}

int get_output_ref_size(
        ReduceRefDim dim,
        int N,
        int C,
        int H,
        int W) {
    switch (dim) {
    case ReduceRefDim::Rows:
        // reduce rows collapses W dimension
        return N * C * H;
    case ReduceRefDim::Cols:
        // reduce cols collapses H dimension
        return N * C * W;
    case ReduceRefDim::Scalar:
        // reduce scalar collapses H and W dimensions
        return N * C;
    default:
        assert(false);
        return 0;
    }
}

std::vector<float> compact_rows(
        const std::vector<float> &data,
        int N, 
        int C, 
        int H) {
    int NCH = N * C * H;
    std::vector<float> result(NCH);
    int p = 0;
    for (int q = 0; q < NCH; q++) {
        result[q] = data[p];
        p += 32;
    }
    return result;
}

std::vector<float> compact_cols(
        const std::vector<float> &data,
        int N, 
        int C, 
        int W) {
    int NC = N * C;
    std::vector<float> result(NC * W);
    int p = 0;
    int q = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int w = 0; w < W; w++) {
            result[q] = data[p + w];
            q++;
        }
        p += W * 32;
    }
    return result;
}

std::vector<float> compact_scalar(
        const std::vector<float> &data,
        int N, 
        int C) {
    int NC = N * C;
    std::vector<float> result(NC);
    int p = 0;
    for (int q = 0; q < NC; q++) {
        result[q] = data[p];
        p += 1024;
    }
    return result;
}

std::vector<float> compact(
        const std::vector<float> &data,
        ReduceDim dim, 
        int N, 
        int C, 
        int H, 
        int W) {
    switch (dim) {
    case ReduceDim::Rows:
        return compact_rows(data, N, C, H);
    case ReduceDim::Cols:
        return compact_cols(data, N, C, W);
    case ReduceDim::Scalar:
        return compact_scalar(data, N, C);
    default:
        assert(false);
        return {};
    }
}

void run_algo(
        ReduceOp op,
        ReduceDim dim,
        int N,
        int C,
        int H,
        int W,
        const std::vector<float> &x,
        float scalar,
        std::vector<float> &y) {
    std::vector<float> x2 = util::tilize(x, W / 32);
    std::vector<uint16_t> tx = float_to_u16b(x2);
    std::vector<uint16_t> ty(y.size());
    core::Platform platform = core::Platform::get_default();
    core::Device device(platform, 0);
    tanto::Reduce solver;
    solver.init(device, op, dim, N, C, H, W);
    solver.run(tx.data(), scalar, ty.data());
    core::Queue queue(device, 0);
    queue.finish();
    device.close();
    y = u16b_to_float(ty);
    if (dim == ReduceDim::Cols) {
        y = util::untilize(y, W / 32);
    } else {
        // W dimension is collapsed
        y = util::untilize(y, 1);
    }
}

void run_ref(
        ReduceRefOp op,
        ReduceRefDim dim,
        int N,
        int C,
        int H,
        int W,
        const std::vector<float> &x,
        float scalar,
        std::vector<float> &y) {
    ReduceRef solver;
    solver.init(op, dim, N, C, H, W);
    solver.run(x.data(), scalar, y.data());
}

void run(const Param &param) {
    ReduceOp op = param.op;
    ReduceDim dim = param.dim;
    std::string str_op = op_to_str(op);
    std::string str_dim = dim_to_str(dim);
    printf("---- Op %s dim %s\n", str_op.c_str(), str_dim.c_str());
    int N = 16;
    int C = 4;
    int H = 128;
    int W = 128;
    int NCHW = N * C * H * W;
    float scaler = 1.0f;
    ReduceRefOp ref_op = op_to_ref(op);
    ReduceRefDim ref_dim = dim_to_ref(dim);
    int ysize = get_output_size(dim, N, C, H, W);
    int ysize_ref = get_output_ref_size(ref_dim, N, C, H, W);
    util::manual_seed(1234);
    std::vector<float> x = util::normal(0.0f, 0.1f, NCHW);
    std::vector<float> y(ysize);
    std::vector<float> yref(ysize_ref);
    run_algo(op, dim, N, C, H, W, x, scaler, y);
    run_ref(ref_op, ref_dim, N, C, H, W, x, scaler, yref);
    std::vector<float> ycomp = compact(y, dim, N, C, H, W);
    compare(ycomp, yref);
}

} // namespace

void main_reduce() {
    for (const Param &param: param_config) {
        run(param);
    }
}

