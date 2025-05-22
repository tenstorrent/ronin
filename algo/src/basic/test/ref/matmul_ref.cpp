// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test/ref/matmul_ref.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

//
//    MatmulRef
//

MatmulRef::MatmulRef() { }

MatmulRef::~MatmulRef() { }

void MatmulRef::init(int batch, int M, int N, int K) {
    m_batch = batch;
    m_M = M;
    m_N = N;
    m_K = K;
}

void MatmulRef::run(
        const float *a,
        const float *b,
        float *c) {
    int ia = 0;
    int ib = 0;
    int ic = 0;
    for (int nb = 0; nb < m_batch; nb++) {
        for (int m = 0; m < m_M; m++) {
            for (int n = 0; n < m_N; n++) {
                float acc = 0.0f;
                for (int k = 0; k < m_K; k++) {
                    acc += a[ia + m * m_K + k] * b[ib + k * m_N + n];
                }
                c[ic + m * m_N + n] = acc;
            }
        }
        ia += m_M * m_K;
        ib += m_K * m_N;
        ic += m_M * m_N;
    }
}

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

