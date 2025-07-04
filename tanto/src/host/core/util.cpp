// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <cmath>
#include <memory>

#include "core/api.hpp"
#include "core/impl.hpp"
#include "core/util.hpp"

namespace ronin {
namespace tanto {
namespace host {

bool is_pow2(uint32_t n) {
    return (n != 0 && (n & (n - 1)) == 0);
}

uint32_t u32_log2(uint32_t n) {
    assert(is_pow2(n));
    return uint32_t(std::log2(double(n)));
}

uint32_t get_item_bytes(DataFormat data_format) {
    switch (data_format) {
    case DataFormat::UINT16:
        return 2;
    case DataFormat::UINT32:
        return 4;
    case DataFormat::FLOAT32:
        return 4;
    case DataFormat::BFLOAT16:
        return 2;
    default:
        throw Error("Unsupported data format");
        return 0;
    }
}

bool range_overlap(const Range &range1, const Range &range2) {
    return (range1.x_start <= range2.x_end && range1.x_end >= range2.x_start &&
        range1.y_start <= range2.y_end && range1.y_end >= range2.y_start);
}

void validate_range_coord(const Range &range) {
    if (range.x_start > range.x_end || range.y_start > range.y_end) {
        throw Error("Invalid range coordinated");
    }
}

void validate_program_grid(
        const std::shared_ptr<ProgramImpl> &program, 
        const std::shared_ptr<GridImpl> &grid) {
    if (program != grid->program()) {
        throw Error("Mismatched program and grid");
    }
}

} // namespace host
} // namespace tanto
} // namespace ronin

