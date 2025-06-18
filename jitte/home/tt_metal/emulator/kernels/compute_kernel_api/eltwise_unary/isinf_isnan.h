// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void isinf_tile(uint32_t idst);
API void isinf_tile_init();
API void isposinf_tile(uint32_t idst);
API void isposinf_tile_init();
API void isneginf_tile(uint32_t idst);
API void isneginf_tile_init();
API void isnan_tile(uint32_t idst);
API void isnan_tile_init();
API void isfinite_tile(uint32_t idst);
API void isfinite_tile_init();

} // namespace ckernel

