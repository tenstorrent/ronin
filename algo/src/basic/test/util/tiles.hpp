// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace util {

void tilize(const float *px, float *py, int H, int W);
void untilize(const float *px, float *py, int H, int W);

std::vector<float> tilize(const std::vector<float> &x, int block);
std::vector<float> untilize(const std::vector<float> &x, int block);

} // namespace util
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

