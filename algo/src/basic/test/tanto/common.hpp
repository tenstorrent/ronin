// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

enum class Algo {
    ELTWISE_BINARY,
    ELTWISE_SFPU,
    BCAST,
    MATMUL_SINGLE,
    MATMUL_MULTI,
    REDUCE,
    TRANSPOSE_WH,
    UNPACK_TILIZE,
    UNPACK_UNTILIZE
};

std::vector<uint16_t> float_to_u16b(const std::vector<float> &x);
std::vector<float> u16b_to_float(const std::vector<uint16_t> &x);

void compare(const std::vector<float> &got, const std::vector<float> &want);

void main_eltwise_binary();
void main_eltwise_sfpu();
void main_bcast();
void main_matmul_single();
void main_matmul_multi();
void main_reduce();
void main_transpose_wh();
void main_unpack_tilize();
void main_unpack_untilize();

