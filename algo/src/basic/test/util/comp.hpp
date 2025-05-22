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

bool comp_allclose(
    const std::vector<float> &golden,
    const std::vector<float> &calculated,
    float rtol,
    float atol,
    float &atol_delta,
    float &rtol_delta,
    int &num_outliers);
float comp_pcc(
    const std::vector<float> &golden,
    const std::vector<float> &calculated);

} // namespace util
} // namespace test
} // namespace basic
} // namespace algo
} // namespace ronin

