#pragma once

#include "llk_defs.h"

#include "compute_kernel_api/common.h"

namespace ckernel {

// ACHTUNG: Ugly design of entire module

API void reduce_init(
    PoolType reduce_type, 
    ReduceDim reduce_dim, 
    bool at_start,
    uint32_t icb, 
    uint32_t icb_scaler, 
    uint32_t ocb);

template<
    bool at_start, 
    PoolType reduce_type = REDUCE_OP, 
    ReduceDim reduce_dim = REDUCE_DIM>
void reduce_init(
        PoolType reduce_op, 
        ReduceDim dim, 
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb = 16) {
    reduce_init(
        reduce_type, 
        reduce_dim, 
        at_start,
        icb, 
        icb_scaler, 
        ocb);
}

API void reduce_init_short(
    PoolType reduce_type, 
    ReduceDim reduce_dim,
    uint32_t icb, 
    uint32_t icb_scaler, 
    uint32_t ocb);

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
void reduce_init_short(uint32_t icb, uint32_t icb_scaler, uint32_t ocb = 16) {
    reduce_init_short(
        reduce_type, 
        reduce_dim,
        icb, 
        icb_scaler, 
        ocb);
}

API void reduce_init_delta(
    PoolType reduce_type, 
    ReduceDim reduce_dim,
    bool at_start, 
    uint32_t ocb, 
    uint32_t icb0, 
    uint32_t icb1);

template<
    bool at_start, 
    PoolType reduce_type = REDUCE_OP, 
    ReduceDim reduce_dim = REDUCE_DIM>
void reduce_init_delta(
        PoolType reduce_op, 
        ReduceDim dim, 
        uint32_t ocb = 16, 
        uint32_t icb0 = 0, 
        uint32_t icb1 = 1) {
    reduce_init_delta(
        reduce_type, 
        reduce_dim,
        at_start, 
        ocb, 
        icb0, 
        icb1);
}

/* TODO
API void reduce_init_delta_no_pack(
    PoolType reduce_type, 
    ReduceDim reduce_dim,
    uint32_t icb0, 
    uint32_t icb1);

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
void reduce_init_delta_no_pack(uint32_t icb0 = 0, uint32_t icb1 = 1) {
    reduce_init_delta_no_pack(reduce_type, reduce_dim, icb0, icb1);
}

API void reduce_init_delta_math(PoolType reduce_type, ReduceDim reduce_dim);

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
void reduce_init_delta_math() {
    reduce_init_delta_math(reduce_type, reduce_dim);
}
*/

API void reduce_revert_delta(ReduceDim reduce_dim, uint32_t ocb);

template<ReduceDim reduce_dim = REDUCE_DIM>
void reduce_revert_delta(uint32_t ocb = 16) {
    reduce_revert_delta(reduce_dim, ocb);
}

API void reduce_tile(
    PoolType reduce_type, 
    ReduceDim reduce_dim,
    uint32_t icb0, 
    uint32_t icb1, 
    uint32_t itile0, 
    uint32_t itile1, 
    uint32_t idst);

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
void reduce_tile(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    reduce_tile(
        reduce_type, 
        reduce_dim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

/* TODO
API void reduce_tile_math(
    PoolType reduce_type, 
    ReduceDim reduce_dim,
    uint32_t idst, 
    uint32_t num_faces);

template<PoolType reduce_type = REDUCE_OP, ReduceDim reduce_dim = REDUCE_DIM>
void reduce_tile_math(uint32_t idst, uint32_t num_faces = 4) {
    reduce_tile_math(reduce_type, reduce_dim, idst, num_faces);
}
*/

} // namespace ckernel

