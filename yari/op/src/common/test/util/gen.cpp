// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <functional>
#include <random>

#include "test/util/gen.hpp"

namespace ronin {
namespace op {
namespace common {
namespace test {
namespace util {

namespace {

std::mt19937 g_engine(0);

} // namespace

void manual_seed(int seed) {
    g_engine.seed(seed);
}

std::vector<float> normal(float mean, float std, int size) {
    std::vector<float> result(size);
    std::normal_distribution<float> dist(mean, std);
    for (int i = 0; i < size; i++) {
        result[i] = dist(g_engine);
    }
    return result;
}

} // namespace util
} // namespace test
} // namespace common
} // namespace op
} // namespace ronin

