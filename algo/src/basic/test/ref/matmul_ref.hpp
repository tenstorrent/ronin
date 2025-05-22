// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace ref {

class MatmulRef {
public:
    MatmulRef();
    ~MatmulRef();
public:
    void init(int batch, int M, int N, int K);
    void run(
        const float *a,
        const float *b,
        float *c);
private:
    int m_batch = 0;
    int m_M = 0;
    int m_N = 0;
    int m_K = 0;
};

} // namespace ref
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

