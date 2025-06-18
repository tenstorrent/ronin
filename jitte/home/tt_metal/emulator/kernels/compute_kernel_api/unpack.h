// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

inline void unpack_reconfig_data_format(
        uint32_t srca_new_operand, uint32_t srcb_new_operand) {
    // SKIPPED
}

inline  void unpack_reconfig_data_format(
    uint32_t srca_old_operand, 
    uint32_t srca_new_operand, 
    uint32_t srcb_old_operand, 
    uint32_t srcb_new_operand) {
    // SKIPPED
}

inline void unpack_reconfig_data_format_srca(uint32_t srca_new_operand) {
    // SKIPPED
}

inline void unpack_reconfig_data_format_srca(
        uint32_t srca_old_operand, uint32_t srca_new_operand) {
    // SKIPPED
}

inline void unpack_reconfig_data_format_srcb(uint32_t srcb_new_operand) {
    // SKIPPED
}

inline void unpack_reconfig_data_format_srcb(
        uint32_t srcb_old_operand, uint32_t srcb_new_operand) {
    // SKIPPED
}

} // ckernel

