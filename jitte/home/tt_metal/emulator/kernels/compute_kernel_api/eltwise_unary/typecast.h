// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void typecast_tile(uint32_t in_dtype, uint32_t out_dtype, uint32_t idst);

template <uint32_t IN_DTYPE, uint32_t OUT_DTYPE>
void typecast_tile(uint32_t idst) {
    typecast_tile(IN_DTYPE, OUT_DTYPE, idst);
}

API void typecast_tile_init();

} // namespace ckernel

