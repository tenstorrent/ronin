// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "host/ref/fc_ref.hpp"

namespace ronin {
namespace op {
namespace fc {
namespace ref {

//
//    FCRef
//

FCRef::FCRef(
        int N,
        int H,
        int C,
        int K):
            m_N(N),
            m_H(H),
            m_C(C),
            m_K(K) { }

FCRef::~FCRef() { }

void FCRef::init(
        const float *x,
        const float *w,
        const float *b,
        float *y) {
    m_x = x;
    m_w = w;
    m_b = b;
    m_y = y;
}

void FCRef::run() {
    // Y = X * Wt + B
    int NH = m_N * m_H;
    for (int nh = 0; nh < NH; nh++) {
        for (int k = 0; k < m_K; k++) {
            float acc = 0.0f;
            for (int c = 0; c < m_C; c++) {
                acc += m_x[nh * m_C + c] * m_w[k * m_C + c];
            }
            if (m_b != nullptr) {
                acc += m_b[k];
            }
            m_y[nh * m_K + k] = acc;
        }
    }
}

int FCRef::input_volume(int index) {
    switch (index) {
    case 0:
        return m_N * m_H * m_C;
    case 1:
        return m_K * m_C;
    case 2:
        return m_K;
    default:
        assert(false);
        return 0;
    }
}

int FCRef::output_volume(int index) {
    assert(index == 0);
    return m_N * m_H * m_K;
}

} // namespace ref
} // namespace fc
} // namespace op
} // namespace ronin

