// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "host/core/api.hpp"

namespace ronin {
namespace op {
namespace group_conv {
namespace tanto {

namespace core = ronin::tanto::host;

bool is_pow2(uint32_t n);
uint32_t u32_log2(uint32_t n);
uint32_t u32_log2_up(uint32_t n);
uint32_t u32_align(uint32_t a, uint32_t b);
uint32_t get_item_bytes(core::DataFormat data_format);
void init_fastdiv_u16(uint32_t d, uint32_t &m, uint32_t &s);

} // namespace tanto
} // namespace group_conv
} // namespace op
} // namespace ronin

