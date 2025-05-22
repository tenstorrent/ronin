// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

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
    int C);

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

