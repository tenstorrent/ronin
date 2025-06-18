// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void unary_op_init_common(uint32_t icb);
API void init_sfpu(uint32_t icb);

} // namespace ckernel

