// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void add_binary_tile(uint32_t idst0, uint32_t idst1);
API void sub_binary_tile(uint32_t idst0, uint32_t idst1);
API void mul_binary_tile(uint32_t idst0, uint32_t idst1);
API void div_binary_tile(uint32_t idst0, uint32_t idst1);
API void rsub_binary_tile(uint32_t idst0, uint32_t idst1);
API void power_binary_tile(uint32_t idst0, uint32_t idst1);
API void add_binary_tile_init();

} // namespace ckernel

