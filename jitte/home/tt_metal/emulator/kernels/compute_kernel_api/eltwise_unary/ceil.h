// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void ceil_tile_init();
API void ceil_tile(uint32_t idst);
API void ceil_tile_float32(uint32_t idst);

} // namespace ckernel

