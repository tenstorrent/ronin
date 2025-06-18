// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void untilize_init(uint32_t icb, uint32_t ocb = 16);
API void untilize_init_short(uint32_t icb);

API void untilize_block(
    int N,
    uint32_t icb, 
    uint32_t block, 
    uint32_t ocb);

template <int N = 1>
void untilize_block(uint32_t icb, uint32_t block, uint32_t ocb) {
    untilize_block(N, icb, block, ocb);
}

API void untilize_uninit(uint32_t icb);

} // ckernels

