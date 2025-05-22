// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "host/core/api.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

namespace core = ronin::tanto::host;

void matmul_split(
    const core::Program &program,
    uint32_t num_cores_x,
    uint32_t num_cores_y,
    uint32_t units_to_divide, 
    bool row_wise,
    uint32_t &target_num_cores, 
    core::Grid &all_cores, 
    core::Grid &core_group_1, 
    core::Grid &core_group_2, 
    uint32_t &units_per_core_group_1, 
    uint32_t &units_per_core_group_2);

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

