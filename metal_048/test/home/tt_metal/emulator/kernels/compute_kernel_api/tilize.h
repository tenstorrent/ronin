#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

API void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb = 16);
/* TODO
API void tilizeA_B_reduce_init(
    uint32_t icb0, 
    uint32_t icb1_scaler, 
    uint32_t block, 
    uint32_t ocb = 16, 
    uint32_t num_faces = 4, 
    uint32_t face_r_dim = 16);
API void tilizeA_B_dot_product_init(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t block, 
    uint32_t ocb = 16, 
    uint32_t num_faces = 4, 
    uint32_t face_r_dim = 16);
*/
API void tilize_init_short(uint32_t icb, uint32_t block);
/* TODO
API void tilize_init_unpack(uint32_t icb, uint32_t block);
API void tilizeA_B_init_unpack(uint32_t icb0, uint32_t icb1, uint32_t block);
*/
API void tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block);
API void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb);
/* TODO
API void unpack_tilize_block(uint32_t icb, uint32_t block);
API void unpack_tilizeA_B_block(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t block, 
    uint32_t tile_idx_b, 
    uint32_t num_faces = 4);
API void unpack_tilizeA_B_dot_product_block(
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t block, 
    uint32_t tile_idx_b, 
    uint32_t num_faces = 4);
*/
API void tilize_uninit(uint32_t icb);
API void tilize_uninit_with_dt(uint32_t old_icb = 0, uint32_t new_icb = 1);

} // namespace ckernel

