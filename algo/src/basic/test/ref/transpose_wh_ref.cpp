// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test/ref/transpose_wh_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    TransposeWhRef
//

TransposeWhRef::TransposeWhRef() { }

TransposeWhRef::~TransposeWhRef() { }

void TransposeWhRef::init(
        int N,
        int H,
        int W) {
    m_N = N;
    m_H = H;
    m_W = W;
}

void TransposeWhRef::run(const float *x, float *y) {
    int HW = m_H * m_W;
    int pos = 0;
    for (int n = 0; n < m_N; n++) {
        for (int h = 0; h < m_H; h++) {
            for (int w = 0; w < m_W; w++) {
                int ix = pos + h * m_W + w;
                int iy = pos + w * m_H + h;
                y[iy] = x[ix]; 
            }
        }
        pos += HW;
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

