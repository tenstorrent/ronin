// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <algorithm>
#include <limits>

#include "test/ref/reduce_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    ReduceRef
//

ReduceRef::ReduceRef() { }

ReduceRef::~ReduceRef() { }

void ReduceRef::init(
        ReduceRefOp op, 
        ReduceRefDim dim,
        int N,
        int C,
        int H,
        int W) {
    m_op = op;
    m_dim = dim;
    m_N = N;
    m_C = C;
    m_H = H;
    m_W = W;
}

void ReduceRef::run(
        const float *x,
        float scaler,
        float *y) {
    switch (m_op) {
    case ReduceRefOp::Max:
        switch (m_dim) {
        case ReduceRefDim::Rows:
            max_rows(x, scaler, y);
            break;
        case ReduceRefDim::Cols:
            max_cols(x, scaler, y);
            break;
        case ReduceRefDim::Scalar:
            max_scalar(x, scaler, y);
            break;
        default:
            assert(false);
            break;
        }
        break;
    case ReduceRefOp::Sum:
        switch (m_dim) {
        case ReduceRefDim::Rows:
            sum_rows(x, scaler, y);
            break;
        case ReduceRefDim::Cols:
            sum_cols(x, scaler, y);
            break;
        case ReduceRefDim::Scalar:
            sum_scalar(x, scaler, y);
            break;
        default:
            assert(false);
            break;
        }
        break;
    default:
        assert(false);
        break;
    }
}

void ReduceRef::max_rows(
        const float *x,
        float scaler,
        float *y) {
    float init = std::numeric_limits<float>::lowest();
    int NC = m_N * m_C;
    int ix = 0;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            float acc = init;
            for (int w = 0; w < m_W; w++) {
                acc = std::max(acc, x[ix]);
                ix++;
            }
            y[iy] = acc * scaler;
            iy++;
        }
    }
}

void ReduceRef::sum_rows(
        const float *x,
        float scaler,
        float *y) {
    int NC = m_N * m_C;
    int ix = 0;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            float acc = 0.0f;
            for (int w = 0; w < m_W; w++) {
                acc += x[ix];
                ix++;
            }
            y[iy] = acc * scaler;
            iy++;
        }
    }
}

void ReduceRef::max_cols(
        const float *x,
        float scaler,
        float *y) {
    float init = std::numeric_limits<float>::lowest();
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int w = 0; w < m_W; w++) {
            float acc = init;
            for (int h = 0; h < m_H; h++) {
                int ix = nc * HW + h * m_W + w;
                acc = std::max(acc, x[ix]);
            }
            y[iy] = acc * scaler;
            iy++;
        }
    }
}

void ReduceRef::sum_cols(
        const float *x,
        float scaler,
        float *y) {
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int w = 0; w < m_W; w++) {
            float acc = 0.0f;
            for (int h = 0; h < m_H; h++) {
                int ix = nc * HW + h * m_W + w;
                acc += x[ix];
            }
            y[iy] = acc * scaler;
            iy++;
        }
    }
}

void ReduceRef::max_scalar(
        const float *x,
        float scaler,
        float *y) {
    float init = std::numeric_limits<float>::lowest();
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int ix = 0;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        float acc = init;
        for (int hw = 0; hw < HW; hw++) {
            acc = std::max(acc, x[ix]);
            ix++;
        }
        y[iy] = acc * scaler;
        iy++;
    }
}

void ReduceRef::sum_scalar(
        const float *x,
        float scaler,
        float *y) {
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int ix = 0;
    int iy = 0;
    for (int nc = 0; nc < NC; nc++) {
        float acc = 0.0f;
        for (int hw = 0; hw < HW; hw++) {
            acc += x[ix];
            ix++;
        }
        y[iy] = acc * scaler;
        iy++;
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

