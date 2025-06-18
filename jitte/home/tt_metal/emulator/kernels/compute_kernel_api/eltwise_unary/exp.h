// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void exp_tile_init(bool fast_and_approx = false);
API void exp_tile(uint32_t idst, bool fast_and_approx = false);

} // namespace ckernel

