// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void sin_tile_init();
API void sin_tile(uint32_t idst);
API void cos_tile_init();
API void cos_tile(uint32_t idst);
API void tan_tile_init();
API void tan_tile(uint32_t idst);

} // namespace ckernel

