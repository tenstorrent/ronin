// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace ronin {
namespace op {
namespace common {
namespace util {

std::string diag_data_stats(const std::vector<float> &x);

} // namespace util
} // namespace common
} // namespace op
} // namespace ronin

