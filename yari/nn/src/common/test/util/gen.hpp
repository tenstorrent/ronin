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

void manual_seed(int seed);
std::vector<float> normal(float mean, float std, int size);

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

