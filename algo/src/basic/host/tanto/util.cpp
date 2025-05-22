// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cmath>
#include <cassert>

#include "host/tanto/util.hpp"

namespace ronin {
namespace algo {
namespace basic {
namespace tanto {

//
//    Numeric utilities
//

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

uint32_t float_as_u32(float x) {
    union U32 {
        float f;
        uint32_t i;
    };
    U32 u32;
    u32.f = x;
    return u32.i;
}

uint16_t float_as_u16b(float x) {
    union U32 {
        float f;
        uint32_t i;
    };
    U32 u32;
    u32.f = x;
    return uint16_t(u32.i >> 16);
}

} // namespace tanto
} // namespace basic
} // namespace algo
} // namespace ronin

