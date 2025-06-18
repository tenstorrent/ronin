// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"

namespace ckernel {

API void add_unary_tile(uint32_t idst, uint32_t param0);
API void sub_unary_tile(uint32_t idst, uint32_t param0);
API void mul_unary_tile(uint32_t idst, uint32_t param0);
API void div_unary_tile(uint32_t idst, uint32_t param0);
API void rsub_unary_tile(uint32_t idst, uint32_t param0);
API void binop_with_scalar_tile_init();

} // namespace ckernel

