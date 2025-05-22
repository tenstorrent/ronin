// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <utility>

#include "core/common.hpp"

namespace ronin {
namespace tanto {
namespace front {

struct FrontendArgs {
    FrontendMode mode;
    std::vector<std::pair<std::string, std::string>> defines;
    std::vector<std::pair<uint32_t, uint32_t>> params;
};

bool run_frontend(
    const FrontendArgs &args, 
    const std::string &input_code,
    std::string &output_code,
    std::vector<std::string> &errors);

} // namespace front
} // namespace tanto
} // namespace ronin

