// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>

#include "test/util/comp.hpp"

namespace ronin {
namespace nn {
namespace common {
namespace test {
namespace util {

bool comp_allclose(
        const std::vector<float> &golden,
        const std::vector<float> &calculated,
        float rtol,
        float atol,
        float &atol_delta,
        float &rtol_delta,
        int &num_outliers) {
    size_t size = golden.size();
    assert(calculated.size() == size);
    rtol_delta = 0.0f;
    atol_delta = 0.0f;
    num_outliers = 0;
    for (size_t i = 0; i < size; i++) {
//printf("@@@ [%zd] GOLDEN %g CALC %g\n", i, golden[i], calculated[i]);
        float a = std::abs(golden[i] - calculated[i]);
        if (a > atol_delta) {
            atol_delta = a;
        }
        float c = std::abs(calculated[i]);
        float r = a / c;
        if (!std::isinf(r) && r > rtol_delta) {
            rtol_delta = r;
        }
        if (a > atol + rtol * c) {
            num_outliers++;
        }
    }
    return (num_outliers == 0);
}

float comp_pcc(
        const std::vector<float> &golden,
        const std::vector<float> &calculated) {
    size_t size = golden.size();
    assert(calculated.size() == size);
    if (size <= 1) {
        return 0.0f;
    }
    double mean_golden = 0.0;
    double mean_calculated = 0.0;
    for (size_t i = 0; i < size; i++) {
        mean_golden += golden[i];
        mean_calculated += calculated[i];
    }
    mean_golden /= double(size);
    mean_calculated /= double(size);
    double gg = 0.0;
    double cc = 0.0;
    double gc = 0.0;
    for (size_t i = 0; i < size; i++) {
        double g = golden[i] - mean_golden;
        double c = calculated[i] - mean_calculated;
        gg += g * g;
        cc += c * c;
        gc += g * c;
    }
    if (gg == 0.0 || cc == 0.0) {
        return 0.0f;
    }
    return float(gc / std::sqrt(gg * cc));
}

} // namespace util
} // namespace test
} // namespace common
} // namespace nn
} // namespace ronin

