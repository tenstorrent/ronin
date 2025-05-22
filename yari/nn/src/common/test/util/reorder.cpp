// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "test/util/reorder.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

std::vector<float> reorder_nchw_to_nhwc(
        const std::vector<float> &x,
        int N,
        int H,
        int W,
        int C) {
    std::vector<float> y(N * H * W * C);
    int HW = H * W;
    int CHW = C * HW;
    int iy = 0;
    for (int n = 0; n < N; n++) {
        for (int hw = 0; hw < HW; hw++) {
            for (int c = 0; c < C; c++) {
                int ix = n * CHW + c * HW + hw; 
                y[iy] = x[ix];
                iy++;
            }
        }
    }
    return y;
}

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

