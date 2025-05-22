// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <vector>

#include "host/core/api.hpp"

#include "host/tanto/matmul_split.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

namespace {

void add_range(
        std::vector<core::Range> &ranges,
        uint32_t x_start,
        uint32_t y_start,
        uint32_t x_end,
        uint32_t y_end) {
    core::Range range{x_start, y_start, x_end, y_end};
    ranges.push_back(range);
}

std::vector<core::Range> num_cores_to_ranges(
        uint32_t target_num_cores, 
        uint32_t num_cores_x,
        uint32_t num_cores_y,
        bool row_wise) {
    assert(target_num_cores <= num_cores_x * num_cores_y &&
        "Target number of cores is greater than total number of cores");
    std::vector<core::Range> ranges;
    if (row_wise) {
        if (target_num_cores > num_cores_x) {
            // start block
            add_range(
                ranges,
                0, 
                0, 
                num_cores_x - 1, 
                target_num_cores / num_cores_x - 1);
            uint32_t leftover_stick_size = target_num_cores % num_cores_x;
            if (leftover_stick_size > 0) {
                uint32_t leftover_start_y = target_num_cores / num_cores_x;
                // leftover block
                add_range(
                    ranges,
                    0, 
                    leftover_start_y, 
                    leftover_stick_size - 1, 
                    leftover_start_y);
            }
        } else {
            // start block
            add_range(
                ranges,
                0, 
                0, 
                target_num_cores - 1,
                0);
        }
    } else {
        if (target_num_cores > num_cores_y) {
            // start block
            add_range(
                ranges,
                0, 
                0, 
                target_num_cores / num_cores_y - 1, 
                num_cores_y - 1);
            uint32_t leftover_stick_size = target_num_cores % num_cores_y;
            if (leftover_stick_size > 0) {
                uint32_t leftover_start_x = target_num_cores / num_cores_y;
                // leftover block
                add_range(
                    ranges,
                    leftover_start_x, 
                    0, 
                    leftover_start_x, 
                    leftover_stick_size - 1);
            }
        } else {
            // start block
            add_range(
                ranges,
                0, 
                0, 
                0, 
                target_num_cores - 1);
        }
    }
    return ranges;
}

} // namespace

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
        uint32_t &units_per_core_group_2) {
    target_num_cores = std::min(units_to_divide, num_cores_x * num_cores_y);
    std::vector<core::Range> all_ranges = 
        num_cores_to_ranges(
            target_num_cores, 
            num_cores_x, 
            num_cores_y, 
            row_wise);
    all_cores = core::Grid(program, all_ranges);

    std::vector<core::Range> core_group_1_ranges;
    std::vector<core::Range> core_group_2_ranges;
    units_per_core_group_1 = units_to_divide / target_num_cores;
    units_per_core_group_2 = 0;

    if (units_to_divide % target_num_cores == 0) {
        // Evenly divided units to all target cores
        core_group_1_ranges = all_ranges;
    } else {
        // Uneven division of units across cores
        // This case should only be hit when there are more units of work than
        // a full grid of cores
        // which is implicitly assumed in the following logic
        // Set 1: Group of cores that do more work
        core_group_1_ranges = 
            num_cores_to_ranges(
                units_to_divide % target_num_cores, 
                num_cores_x, 
                num_cores_y, 
                row_wise);
        core::Range &last_block_group_1 = core_group_1_ranges.back();
        core::Range &last_block_all_cores = all_ranges.back();
        if (row_wise) {
            // Case where only the last row is divided between core group 1 and 2
            if (last_block_group_1.y_end == last_block_all_cores.y_end && 
                    last_block_group_1.x_end != last_block_all_cores.x_end) {
                // leftover block
                add_range(
                    core_group_2_ranges,
                    last_block_group_1.x_end + 1, 
                    last_block_group_1.y_end, 
                    last_block_all_cores.x_end,
                    last_block_all_cores.y_end);
            } else {
                // Case where a middle row is divided between core group 1 and 2
                if (last_block_group_1.x_end != num_cores_x - 1) {
                    // leftover stick
                    add_range(
                        core_group_2_ranges,
                        last_block_group_1.x_end + 1, 
                        last_block_group_1.y_end,
                        num_cores_x - 1, 
                        last_block_group_1.y_end);
                }
                // Remaining rows of cores that does less work
                // leftover block
                add_range(
                    core_group_2_ranges,
                    0, 
                    last_block_group_1.y_end + 1, 
                    last_block_all_cores.x_end,
                    last_block_all_cores.y_end);
            }
        } else {
            // Case where only the last column is divided between core group 1 and 2
            if (last_block_group_1.x_end == last_block_all_cores.x_end && 
                    last_block_group_1.y_end != last_block_all_cores.y_end) {
                // leftover block
                add_range(
                    core_group_2_ranges,
                    last_block_group_1.x_end, 
                    last_block_group_1.y_end + 1, 
                    last_block_all_cores.x_end,
                    last_block_all_cores.y_end);
            } else {
                // Case where a middle column is divided between core group 1 and 2
                if (last_block_group_1.y_end != num_cores_y - 1) {
                    // leftover stick
                    add_range(
                        core_group_2_ranges,
                        last_block_group_1.x_end, 
                        last_block_group_1.y_end + 1,
                        last_block_group_1.x_end, 
                        num_cores_y - 1);
                }
                // Remaining columns of cores that does less work
                // lefrover block
                add_range(
                    core_group_2_ranges,
                    last_block_group_1.x_end + 1, 
                    0, 
                    last_block_all_cores.x_end,
                    last_block_all_cores.y_end);
            }
        }
        units_per_core_group_2 = units_per_core_group_1;
        units_per_core_group_1++;
    }
    core_group_1 = core::Grid(program, core_group_1_ranges);
    core_group_2 = core::Grid(program, core_group_2_ranges);
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

