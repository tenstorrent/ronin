// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void copy_dest_values(uint32_t idst0, uint32_t idst1);
API void copy_dest_values_init();

} // namespace ckernel

