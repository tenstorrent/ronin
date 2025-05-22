// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <vector>
#include <functional>
#include <random>

#include "test/util/gen.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace test {
namespace util {

namespace {

std::mt19937 g_engine(0);

} // namespace

void manual_seed(int seed) {
    g_engine.seed(seed);
}

std::vector<float> uniform(float a, float b, int size) {
    std::vector<float> result(size);
    std::uniform_real_distribution<float> dist(a, b);
    for (int i = 0; i < size; i++) {
        result[i] = dist(g_engine);
    }
    return result;
}

std::vector<float> uniform_from_vector(const std::vector<float> &v, int size) {
    int n = int(v.size());
    assert(n > 0);
    std::vector<float> result(size);
    if (n <= 1) {
        float x = (n > 0) ? v[0] : 0.0f;
        for (int i = 0; i < size; i++) {
            result[i] = x;
        }
    } else {
        std::uniform_int_distribution<int> dist(0, n - 1);
        for (int i = 0; i < size; i++) {
            int k = dist(g_engine);
            result[i] = v[k];
        }
    }
    return result;
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
} // namespace basic
} // namespace algo
} // namespace ronin

