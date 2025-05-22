// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "test/ref/bcast_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    BcastRef
//

BcastRef::BcastRef() { }

BcastRef::~BcastRef() { }

void BcastRef::init(
        BcastRefOp op, 
        BcastRefDim dim,
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

void BcastRef::run(
        const float *a,
        const float *b,
        float *c) {
    switch (m_op) {
    case BcastRefOp::Add:
        switch (m_dim) {
        case BcastRefDim::Rows:
            add_rows(a, b, c);
            break;
        case BcastRefDim::Cols:
            add_cols(a, b, c);
            break;
        case BcastRefDim::Scalar:
            add_scalar(a, b, c);
            break;
        default:
            assert(false);
            break;
        }
        break;
    case BcastRefOp::Sub:
        switch (m_dim) {
        case BcastRefDim::Rows:
            sub_rows(a, b, c);
            break;
        case BcastRefDim::Cols:
            sub_cols(a, b, c);
            break;
        case BcastRefDim::Scalar:
            sub_scalar(a, b, c);
            break;
        default:
            assert(false);
            break;
        }
        break;
    case BcastRefOp::Mul:
        switch (m_dim) {
        case BcastRefDim::Rows:
            mul_rows(a, b, c);
            break;
        case BcastRefDim::Cols:
            mul_cols(a, b, c);
            break;
        case BcastRefDim::Scalar:
            mul_scalar(a, b, c);
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

void BcastRef::add_rows(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib_start = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            int ib = ib_start;
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] + b[ib];
                ia++;
                ib++;
            }
        }
        ib_start += m_W;
    }
}

void BcastRef::add_cols(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] + b[ib];
                ia++;
            }
            ib++;
        }
    }
}

void BcastRef::add_scalar(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int hw = 0; hw < HW; hw++) {
            c[ia] = a[ia] + b[ib];
            ia++;
        }
        ib++;
    }
}

void BcastRef::sub_rows(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib_start = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            int ib = ib_start;
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] - b[ib];
                ia++;
                ib++;
            }
        }
        ib_start += m_W;
    }
}

void BcastRef::sub_cols(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] - b[ib];
                ia++;
            }
            ib++;
        }
    }
}

void BcastRef::sub_scalar(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int hw = 0; hw < HW; hw++) {
            c[ia] = a[ia] - b[ib];
            ia++;
        }
        ib++;
    }
}

void BcastRef::mul_rows(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib_start = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            int ib = ib_start;
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] * b[ib];
                ia++;
                ib++;
            }
        }
        ib_start += m_W;
    }
}

void BcastRef::mul_cols(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int h = 0; h < m_H; h++) {
            for (int w = 0; w < m_W; w++) {
                c[ia] = a[ia] * b[ib];
                ia++;
            }
            ib++;
        }
    }
}

void BcastRef::mul_scalar(
        const float *a,
        const float *b,
        float *c) {
    int NC = m_N * m_C;
    int HW = m_H * m_W;
    int ia = 0;
    int ib = 0;
    for (int nc = 0; nc < NC; nc++) {
        for (int hw = 0; hw < HW; hw++) {
            c[ia] = a[ia] * b[ib];
            ia++;
        }
        ib++;
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

