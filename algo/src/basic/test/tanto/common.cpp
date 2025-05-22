// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>

#include "test/util/comp.hpp"

#include "test/tanto/common.hpp"

using namespace ronin::algo::basic::test;

namespace {

union U32 {
    float f;
    uint32_t i;
};

} // namespace

std::vector<uint16_t> float_to_u16b(const std::vector<float> &x) {
    U32 u32;
    size_t n = x.size();
    std::vector<uint16_t> y(n);
    for (size_t i = 0; i < n; i++) {
        u32.f = x[i];
        y[i] = uint16_t(u32.i >> 16);
    }
    return y;
}

std::vector<float> u16b_to_float(const std::vector<uint16_t> &x) {
    U32 u32;
    size_t n = x.size();
    std::vector<float> y(n);
    for (size_t i = 0; i < n; i++) {
        u32.i = uint32_t(x[i]) << 16;
        y[i] = u32.f;
    }
    return y;
}

void compare(const std::vector<float> &got, const std::vector<float> &want) {
    float rtol = 1.0e-1f;
    float atol = 1.0e-3f;
    float rtol_delta = 0.0f;
    float atol_delta = 0.0f;
    int num_outliers = 0;

    bool allclose = 
        util::comp_allclose(
            want, 
            got, 
            rtol, 
            atol, 
            rtol_delta, 
            atol_delta, 
            num_outliers);
    printf("All close = %s\n", allclose ? "OK" : "FAIL");
    printf("Max ATOL delta: %g, max RTOL delta: %g, outliers: %d / %zd\n", 
        atol_delta, rtol_delta, num_outliers, got.size());

    float pcc = util::comp_pcc(want, got);
    printf("Pcc = %s\n", (pcc >= 0.9995f) ? "OK" : "FAIL"); 
    printf("PCC: %g\n", pcc);
}

