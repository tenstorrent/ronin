// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void fill_tile_bitcast(uint32_t idst, uint32_t param0);
API void fill_tile_init();

} // namespace ckernel

