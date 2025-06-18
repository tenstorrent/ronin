// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

// TODO: Implement in Compute
API void rsub_tile(uint32_t idst, uint32_t param0);
API void rsub_tile_init();

} // namespace ckernel

