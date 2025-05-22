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

void manual_seed(int seed);
std::vector<float> uniform(float a, float b, int size);
std::vector<float> uniform_from_vector(const std::vector<float> &v, int size);
std::vector<float> normal(float mean, float std, int size);

} // namespace util
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

