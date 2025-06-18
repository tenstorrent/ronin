// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void logical_not_unary_tile(uint32_t idst);
API void logical_not_unary_tile_init();

} // namespace ckernel

