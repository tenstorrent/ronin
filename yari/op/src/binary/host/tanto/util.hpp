// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "host/core/api.hpp"

namespace ronin {
namespace op {
namespace binary {
namespace tanto {

namespace core = ronin::tanto::host;

bool is_pow2(uint32_t n);
uint32_t u32_log2(uint32_t n);
uint32_t u32_log2_up(uint32_t n);
uint32_t u32_align(uint32_t a, uint32_t b);
uint32_t get_item_bytes(core::DataFormat data_format);

} // namespace tanto
} // namespace binary
} // namespace op
} // namespace ronin

