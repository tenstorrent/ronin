// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

API void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim);

template <uint32_t block_ct_dim = 8>
void pack_untilize_init(uint32_t icb, uint32_t ocb) {
    pack_untilize_init(icb, ocb, block_ct_dim);
}

API void pack_untilize_block(
    uint32_t icb, 
    uint32_t block_rt_dim, 
    uint32_t ocb,
    uint32_t block_ct_dim);

template <uint32_t block_ct_dim = 8>
void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb) {
    pack_untilize_block(icb, block_rt_dim, ocb, block_ct_dim);
}

API void pack_untilize_uninit(uint32_t ocb = 16);

API void pack_untilize_dst_init_short(
    uint32_t ocb, 
    uint32_t face_r_dim, 
    uint32_t num_faces,
    uint32_t block_ct_dim, 
    uint32_t full_ct_dim, 
    bool diagonal);

template<
    uint32_t block_ct_dim = 8, 
    uint32_t full_ct_dim = block_ct_dim, 
    bool diagonal = false>
void pack_untilize_dst_init_short(
        uint32_t ocb, 
        uint32_t face_r_dim = 16, 
        uint32_t num_faces = 4) {
    pack_untilize_dst_init_short(
        ocb, 
        face_r_dim, 
        num_faces,
        block_ct_dim, 
        full_ct_dim, 
        diagonal);
}

API void pack_untilize_dst(
    uint32_t ocb, 
    uint32_t block_rt_dim, 
    uint32_t block_c_index, 
    uint32_t face_r_dim, 
    uint32_t num_faces,
    uint32_t block_ct_dim, 
    uint32_t full_ct_dim, 
    bool diagonal);

template<
    uint32_t block_ct_dim = 8, 
    uint32_t full_ct_dim = block_ct_dim, 
    bool diagonal = false>
void pack_untilize_dst(
        uint32_t ocb, 
        uint32_t block_rt_dim = 1, 
        uint32_t block_c_index = 0, 
        uint32_t face_r_dim = 16, 
        uint32_t num_faces = 4) {
    pack_untilize_dst(
        ocb, 
        block_rt_dim, 
        block_c_index, 
        face_r_dim, 
        num_faces,
        block_ct_dim, 
        full_ct_dim, 
        diagonal);
}

API void pack_untilize_init_short(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim);

template <uint32_t block_ct_dim = 8>
void pack_untilize_init_short(uint32_t icb, uint32_t ocb) {
    pack_untilize_init_short(icb, ocb, block_ct_dim);
}

} // namespace ckernel

