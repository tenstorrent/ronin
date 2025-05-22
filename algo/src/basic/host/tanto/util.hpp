// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

bool is_pow2(uint32_t n);
uint32_t u32_log2(uint32_t n);
uint32_t u32_log2_up(uint32_t n);
uint32_t float_as_u32(float x);
uint16_t float_as_u16b(float x);

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

