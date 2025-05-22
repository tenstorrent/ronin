// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Temporary definitions (TODO: Design regular solution)

#ifndef REDUCE_OP
#define REDUCE_OP PoolType::MAX
#endif

#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/untilize.h"

#include "compute_kernel_api.h"

#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/elu.h"
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#include "compute_kernel_api/eltwise_unary/erfinv.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/i0.h"
#include "compute_kernel_api/eltwise_unary/isinf_isnan.h"
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/relu.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"

// used by LLK helpers
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_datacopy.h"
#endif

// temporary solution
#define FAST_AND_APPROX false

typedef uint32_t uint32;

struct Global {
    uint32 addr;
    uint32 log2_page_size;
};

struct Local {
    uint32 addr;
};

struct Pipe {
    uint32 cb_id;
    uint32 frame_size;
};

struct Semaphore {
    uint32 addr;
};

// LLK helpers

#ifdef TRISC_MATH
// based on "llk_math_unary_datacopy_api.h"
template <DataCopyType type, BroadcastType src_b_bcast_type = BroadcastType::NONE, bool is_fp32_dest_acc_en = false>
// within_face_16x16_transpose is used by unpacker, math does not transpose
inline void tanto_llk_math_eltwise_unary_datacopy_init(
        uint32_t transpose_of_faces = 0,
        uint32_t within_face_16x16_transpose = 0) {
    // introduced to avoid use of "icb" in MATH init section
    // omit "num_faces" argument: GS - absent, WH - default 4
    _llk_math_eltwise_unary_datacopy_init_<type, src_b_bcast_type, is_fp32_dest_acc_en>(
        transpose_of_faces, within_face_16x16_transpose);
}
#endif

// wrappers

ALWI void tanto_max_tile(uint32_t idst) {
    max_tile(idst, idst + 1);
}

// common

ALWI void tanto_compute_init() {
    // ACHTUNG: Cheating - requires same data format of all CBs
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated(0, 0) ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));
    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(0) ));
}

// math

ALWI void tanto_copy_init() {
    // also serves as common init for all SFPU primitives
    // transpose_of_faces = 0, within_face_16x16_transpose = 0
    MATH(( tanto_llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(0, 0) ));
}

ALWI void tanto_add_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE>() )); 
}

ALWI void tanto_sub_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::NONE>() )); 
}

ALWI void tanto_mul_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MATH_FIDELITY>() )); 
}

ALWI void tanto_add_bcast_rows_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::ROW>() ));
}

ALWI void tanto_sub_bcast_rows_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::ROW>() ));
}

ALWI void tanto_mul_bcast_rows_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::ROW, MATH_FIDELITY>() ));
}

ALWI void tanto_add_bcast_cols_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::COL>() ));
}

ALWI void tanto_sub_bcast_cols_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::COL>() ));
}

ALWI void tanto_mul_bcast_cols_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>() ));
}

ALWI void tanto_add_bcast_scalar_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::SCALAR>() ));
}

ALWI void tanto_sub_bcast_scalar_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWSUB, BroadcastType::SCALAR>() ));
}

ALWI void tanto_mul_bcast_scalar_init() { 
    MATH(( llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::SCALAR, MATH_FIDELITY>() ));
}

ALWI void tanto_matmul_init(bool transpose) { 
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, transpose) ));
}

// ACHTUNG: New version (TODO: Unify)
ALWI void tanto_matmul_init(bool transpose, uint32 ct_dim, uint32 rt_dim, uint32 kt_dim) { 
    MATH(( llk_math_matmul_init<MATH_FIDELITY>(0, 1, transpose, ct_dim, rt_dim, kt_dim) ));
}

ALWI void tanto_reduce_max_rows_init() { 
    MATH(( llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, MATH_FIDELITY>() ));
}

ALWI void tanto_reduce_max_cols_init() { 
    MATH(( llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL, MATH_FIDELITY>() ));
}

ALWI void tanto_reduce_max_scalar_init() { 
    MATH(( llk_math_reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR, MATH_FIDELITY>() ));
}

ALWI void tanto_reduce_sum_rows_init() { 
    MATH(( llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, MATH_FIDELITY>() ));
}

ALWI void tanto_reduce_sum_cols_init() { 
    MATH(( llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_COL, MATH_FIDELITY>() ));
}

ALWI void tanto_reduce_sum_scalar_init() { 
    MATH(( llk_math_reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, MATH_FIDELITY>() ));
}

ALWI void tanto_transpose_init() { 
    // transpose_of_faces = 1, within_face_16x16_transpose = 1
    MATH(( tanto_llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(1, 1) ));
}

ALWI void tanto_tilize_block_init() {
    // transpose_of_faces = 0, within_face_16x16_transpose = 0
    MATH(( tanto_llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(0, 0) ));
}

ALWI void tanto_untilize_block_init() { 
    // transpose_of_faces = 0, within_face_16x16_transpose = 0
    MATH(( tanto_llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(0, 0) ));
}

// sfpu

ALWI void tanto_abs_init() {
    MATH(( llk_math_eltwise_unary_sfpu_abs_init<APPROX>() ));
}

ALWI void tanto_acos_init() {
    MATH(( llk_math_eltwise_unary_sfpu_acos_init<true>() ));
}

ALWI void tanto_asin_init() {
    MATH(( llk_math_eltwise_unary_sfpu_asin_init<true>() ));
}

ALWI void tanto_atan_init() {
    MATH(( llk_math_eltwise_unary_sfpu_atan_init<true>() ));
}

ALWI void tanto_binary_scalar_init() {
    MATH((llk_math_eltwise_unary_sfpu_binop_with_scalar_init<APPROX>()));
}

ALWI void tanto_cos_init() {
    MATH((llk_math_eltwise_unary_sfpu_cosine_init<APPROX>()));
}

ALWI void tanto_elu_init() {
    MATH(( llk_math_eltwise_unary_sfpu_elu_init<APPROX>() ));
}

ALWI void tanto_eqz_init() {
    MATH(( llk_math_eltwise_unary_sfpu_eqz_init<APPROX>() ));
}

ALWI void tanto_erf_init() {
    // originally fast_and_approx
    MATH((llk_math_eltwise_unary_sfpu_erf_init<true>()));
}

ALWI void tanto_erfc_init() {
    // originally fast_and_approx
    MATH((llk_math_eltwise_unary_sfpu_erfc_init<true>()));
}

ALWI void tanto_erfinv_init() {
    MATH((llk_math_eltwise_unary_sfpu_erfinv_init<APPROX>() ));
}

ALWI void tanto_exp_init() {
    // originally fast_and_approx
    MATH(( llk_math_eltwise_unary_sfpu_exponential_init<false>() ));
}

ALWI void tanto_exp2_init() {
    MATH(( llk_math_eltwise_unary_sfpu_exp2_init<true>() ));
}

ALWI void tanto_expm1_init() {
    MATH(( llk_math_eltwise_unary_sfpu_expm1_init<true>() ));
}

ALWI void tanto_gelu_init() {
    // originally fast_and_approx
    MATH(( llk_math_eltwise_unary_sfpu_gelu_init<true>() ));
}

ALWI void tanto_gez_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gez_init<APPROX>() ));
}

ALWI void tanto_gtz_init() {
    MATH(( llk_math_eltwise_unary_sfpu_gtz_init<APPROX>() ));
}

ALWI void tanto_heaviside_init() {
    MATH(( llk_math_eltwise_unary_sfpu_heaviside_init<APPROX>() ));
}

ALWI void tanto_i0_init() {
    MATH((llk_math_eltwise_unary_sfpu_i0_init<APPROX>() ));
}

ALWI void tanto_isfinite_init() {
    MATH((llk_math_eltwise_unary_sfpu_isfinite_init<APPROX>() ));
}

ALWI void tanto_isinf_init() {
    MATH((llk_math_eltwise_unary_sfpu_isinf_init<APPROX>() ));
}

ALWI void tanto_isnan_init() {
    MATH((llk_math_eltwise_unary_sfpu_isnan_init<APPROX>() ));
}

ALWI void tanto_isneginf_init() {
    MATH((llk_math_eltwise_unary_sfpu_isneginf_init<APPROX>() ));
}

ALWI void tanto_isposinf_init() {
    MATH((llk_math_eltwise_unary_sfpu_isposinf_init<APPROX>() ));
}

ALWI void tanto_leaky_relu_init() {
    MATH(( llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>() ));
}

ALWI void tanto_lez_init() {
    MATH(( llk_math_eltwise_unary_sfpu_lez_init<APPROX>() ));
}

ALWI void tanto_log_init() {
    MATH(( llk_math_eltwise_unary_sfpu_log_init<APPROX>() ));
}

ALWI void tanto_log_with_base_init() {
    MATH((llk_math_eltwise_unary_sfpu_log_with_base_init<APPROX>()));
}

ALWI void tanto_logical_not_init() {
    MATH((llk_math_eltwise_unary_sfpu_logical_not_unary_init<APPROX>() ));
}

ALWI void tanto_ltz_init() {
    MATH(( llk_math_eltwise_unary_sfpu_ltz_init<APPROX>() ));
}

ALWI void tanto_nez_init() {
    MATH(( llk_math_eltwise_unary_sfpu_nez_init<APPROX>() ));
}

ALWI void tanto_max_init() {
    MATH(( llk_math_eltwise_unary_sfpu_max_init<APPROX>() ));
}

ALWI void tanto_power_init() {
    MATH(( llk_math_eltwise_unary_sfpu_power_init<APPROX>() ));
}

ALWI void tanto_recip_init() {
    MATH(( llk_math_eltwise_unary_sfpu_reciprocal_init<APPROX>() ));
}

ALWI void tanto_relu_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_init<APPROX>() ));
}

ALWI void tanto_relu_max_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>() ));
}

ALWI void tanto_relu_min_init() {
    MATH(( llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>() ));
}

ALWI void tanto_rsqrt_init() {
    // originally fast_and_approx
    MATH(( llk_math_eltwise_unary_sfpu_rsqrt_init<true>() ));
}

ALWI void tanto_sigmoid_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sigmoid_init<APPROX>() ));
}

ALWI void tanto_sign_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sign_init<APPROX>() ));
}

ALWI void tanto_signbit_init() {
    MATH(( llk_math_eltwise_unary_sfpu_signbit_init<APPROX>() ));
}

ALWI void tanto_sin_init() {
    MATH((llk_math_eltwise_unary_sfpu_sine_init<APPROX>()));
}

ALWI void tanto_sqrt_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sqrt_init<APPROX>() ));
}

ALWI void tanto_square_init() {
    MATH(( llk_math_eltwise_unary_sfpu_square_init<APPROX>() ));
}

ALWI void tanto_tan_init() {
    MATH((llk_math_eltwise_unary_sfpu_tan_init<APPROX>()));
}

ALWI void tanto_tanh_init() {
    MATH(( llk_math_eltwise_unary_sfpu_tanh_init<APPROX>() ));
}

// unpack

ALWI void tanto_unpack_binary_init(uint32 icb0, uint32 icb1) { 
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1) ));
}

ALWI void tanto_unpack_bcast_rows_init(uint32 icb0, uint32 icb1) { 
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::ROW>(icb0, icb1) ));
}

ALWI void tanto_unpack_bcast_cols_init(uint32 icb0, uint32 icb1) { 
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::COL>(icb0, icb1) ));
}

ALWI void tanto_unpack_bcast_scalar_init(uint32 icb0, uint32 icb1) { 
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_init<BroadcastType::SCALAR>(icb0, icb1) ));
}

ALWI void tanto_unpack_matmul_init(uint32 icb0, uint32 icb1, bool transpose) { 
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_matmul_init(icb0, icb1, transpose) ));
}

// ACHTUNG: New version (TODO: Unify)
ALWI void tanto_unpack_matmul_init(uint32 icb0, uint32 icb1, bool transpose, uint32 ct_dim, uint32 rt_dim, uint32 kt_dim) { 
    UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1) ));
    UNPACK(( llk_unpack_AB_matmul_init(icb0, icb1, transpose, ct_dim, rt_dim, kt_dim) ));
}

ALWI void tanto_unpack_unary_init(uint32 icb) {
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE>(icb) ));
    // transpose_of_faces = 0, within_face_16x16_transpose = 0
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(0, 0, icb) ));
}

ALWI void tanto_unpack_reduce_rows_init(uint32 icb, uint32 icb_scaler) {
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));
    UNPACK(( llk_unpack_AB_reduce_init<ReduceDim::REDUCE_ROW>(icb, icb_scaler) ));
}

ALWI void tanto_unpack_reduce_cols_init(uint32 icb, uint32 icb_scaler) {
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));
    UNPACK(( llk_unpack_AB_reduce_init<ReduceDim::REDUCE_COL>(icb, icb_scaler) ));
}

ALWI void tanto_unpack_reduce_scalar_init(uint32 icb, uint32 icb_scaler) {
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, icb_scaler) ));
    UNPACK(( llk_unpack_AB_reduce_init<ReduceDim::REDUCE_SCALAR>(icb, icb_scaler) ));
}

ALWI void tanto_unpack_tilize_block_init(uint32 icb, uint32 block) {
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated<DST_ACCUM_MODE>(icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tanto_unpack_transpose_init(uint32 icb) {
    // ACHTUNG: Originally "0", replaced by "icb"
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE>(icb, true) ));
    // transpose_of_faces = 1, within_face_16x16_transpose = 1
    // added explicit "icb" argument
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(1, 1, icb) ));
}

ALWI void tanto_unpack_untilize_block_init(uint32 icb) {
    UNPACK(( llk_unpack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE>(icb) ));
    UNPACK(( llk_unpack_untilize_init(icb) )); // init must be after configure
}

// pack

ALWI void tanto_pack_init(uint32 ocb) { 
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_reduce_config_v2<ReduceDim::REDUCE_ROW, false, true, DST_ACCUM_MODE>(ocb) ));
}

ALWI void tanto_pack_row_init(uint32 ocb) { 
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_reduce_config_v2<ReduceDim::REDUCE_ROW, false, false, DST_ACCUM_MODE>(ocb) ));
}

ALWI void tanto_pack_col_init(uint32 ocb) { 
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_reduce_config_v2<ReduceDim::REDUCE_COL, false, false, DST_ACCUM_MODE>(ocb) ));
}

ALWI void tanto_pack_scalar_init(uint32 ocb) { 
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_pack_reduce_config_v2<ReduceDim::REDUCE_SCALAR, false, false, DST_ACCUM_MODE>(ocb) ));
}


