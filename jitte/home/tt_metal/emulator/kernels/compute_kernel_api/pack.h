// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"
#include "llk_defs.h"

namespace ckernel {

// ACHTUNG: emulator-specific
API void pack_relu_config(uint32_t config);

API void pack_tile(
    uint32_t ifrom_dst, 
    uint32_t icb, 
    uint32_t output_tile_index,
    bool out_of_order_output);

template <bool out_of_order_output = false>
void pack_tile(uint32_t ifrom_dst, uint32_t icb, uint32_t output_tile_index = 0) {
    pack_tile(ifrom_dst, icb, output_tile_index, out_of_order_output);
}

API void matmul_pack_tile(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles);

inline void pack_reconfig_data_format(uint32_t new_operand) {
    // SKIPPED
}

inline void pack_reconfig_data_format(uint32_t old_operand, uint32_t new_operand) {
    // SKIPPED
}

// ACHTUNG: New in 0.48
inline void pack_reconfig_l1_acc(uint32_t l1_acc_en) {
    // SKIPPED
}

} // namespace ckernel

