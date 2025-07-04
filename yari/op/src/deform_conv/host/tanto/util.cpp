// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>
#include <cmath>

#include "host/tanto/util.hpp"

namespace ronin {
namespace op {
namespace deform_conv {
namespace tanto {

bool is_pow2(uint32_t n) {
    return (n != 0 && (n & (n - 1)) == 0);
}

uint32_t u32_log2(uint32_t n) {
    assert(is_pow2(n));
    return uint32_t(std::log2(double(n)));
}

uint32_t u32_log2_up(uint32_t n) {
    return uint32_t(std::ceil(std::log2(double(n))));
}

uint32_t u32_align(uint32_t a, uint32_t b) {
    return ((a + b - 1) / b) * b;
}

uint32_t get_item_bytes(core::DataFormat data_format) {
    switch (data_format) {
    case core::DataFormat::UINT32:
        return 4;
    case core::DataFormat::FLOAT32:
        return 4;
    case core::DataFormat::BFLOAT16:
        return 2;
    default:
        assert(false);
        return 0;
    }
}

// unused (reserved)
void init_fastdiv_u16(uint32_t d, uint32_t &m, uint32_t &s) {
    s = 15 + u32_log2_up(d);
    m = ((1 << s) + d - 1) / d;
}

} // namespace tanto
} // namespace deform_conv
} // namespace op
} // namespace ronin

