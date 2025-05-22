// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"

namespace ronin {
namespace tanto {
namespace host {

bool is_pow2(uint32_t n);
uint32_t u32_log2(uint32_t n);

uint32_t get_item_bytes(DataFormat data_format);

bool range_overlap(const Range &range1, const Range &range2);

void validate_range_coord(const Range &range);
void validate_program_grid(
    const std::shared_ptr<ProgramImpl> &program, 
    const std::shared_ptr<GridImpl> &grid); 

} // namespace host
} // namespace tanto
} // namespace ronin

