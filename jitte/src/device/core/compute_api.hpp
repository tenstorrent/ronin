// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"

namespace tt {
namespace metal {
namespace device {

class Compute {
public:
    Compute() { }
    virtual ~Compute() { }
public:
    virtual void reset() = 0;
    virtual uint32_t get_arg_uint32(int arg_idx) = 0;
    // reg_api
    virtual void acquire_dst(DstMode mode) = 0;
    virtual void tile_regs_acquire() = 0;
    virtual void tile_regs_wait() = 0;
    virtual void release_dst(DstMode mode) = 0;
    virtual void tile_regs_commit() = 0;
    virtual void tile_regs_release() = 0;
    // pack
    virtual void pack_relu_config(uint32_t config) = 0;
    virtual void pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t output_tile_index,
        bool out_of_order_output) = 0;
    virtual void matmul_pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t ntiles) = 0;
    virtual void pack_reconfig_data_format(uint32_t new_operand) = 0;
    // unpack
    virtual void unpack_reconfig_data_format(
        uint32_t srca_new_operand, 
        uint32_t srcb_new_operand) = 0;
    // cb_api
    virtual void cb_wait_front(uint32_t cbid, uint32_t ntiles) = 0;
    virtual void cb_pop_front(uint32_t cbid, uint32_t ntiles) = 0;
    virtual void cb_reserve_back(uint32_t cbid, uint32_t ntiles) = 0;
    virtual void cb_push_back(uint32_t cbid, uint32_t ntiles) = 0;
    virtual uint32_t get_read_ptr(uint32_t cbid) = 0;
    virtual uint32_t get_write_ptr(uint32_t cbid) = 0;
    virtual void set_read_ptr(uint32_t cbid, uint32_t ptr) = 0;
    virtual void set_write_ptr(uint32_t cbid, uint32_t ptr) = 0;
    // bcast
    virtual void sub_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void mul_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void mul_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void add_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void add_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void init_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t ocb) = 0;
    virtual void any_tiles_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void add_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void sub_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void mul_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) = 0;
    virtual void add_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) = 0;
    virtual void mul_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) = 0;
    virtual void mul_tiles_bcast_scalar(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) = 0;
    virtual void mul_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) = 0;
    virtual void sub_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) = 0;
    // eltwise_binary
    virtual void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb) = 0;
    virtual void mul_tiles_init_f() = 0;
    virtual void mul_tiles_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void add_tiles_init_nof() = 0;
    virtual void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) = 0;
    virtual void sub_tiles_init_nof() = 0;
    virtual void sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) = 0;
    virtual void mul_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void add_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void sub_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    virtual void binary_op_specific_init(
        bool full_init, 
        EltwiseBinaryType eltwise_binary_op_type) = 0;
    // eltwise_unary
    virtual void unary_op_init_common(uint32_t icb) = 0;
    virtual void init_sfpu(uint32_t icb) = 0;
    // matmul
    virtual void mm_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose) = 0;
    virtual void matmul_tiles(
        uint32_t c_in0, 
        uint32_t c_in1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst, 
        uint32_t transpose) = 0;
    virtual void mm_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose) = 0;
    virtual void mm_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t c_in_old_srca, 
        uint32_t transpose) = 0;
    virtual void mm_block_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) = 0;
    virtual void matmul_block(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t in0_tile_index, 
        uint32_t in1_tile_index, 
        uint32_t idst, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) = 0;
    virtual void mm_block_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) = 0;
    virtual void mm_block_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t old_in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) = 0;
    // reduce
    virtual void reduce_init(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) = 0;
    virtual void reduce_init_short(
        PoolType reduce_type, 
        ReduceDim reduce_dim,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) = 0;
    virtual void reduce_init_delta(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t ocb, 
        uint32_t icb0, 
        uint32_t icb1) = 0;
    virtual void reduce_revert_delta(ReduceDim reduce_dim, uint32_t ocb) = 0;
    virtual void reduce_tile(
        PoolType reduce_type, 
        ReduceDim reduce_dim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) = 0;
    // tile_move_copy
    virtual void copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose) = 0;
    virtual void copy_tile_init() = 0;
    virtual void copy_tile_to_dst_init_short_with_dt(
        uint32_t old_cbid, 
        uint32_t new_cbid, 
        uint32_t transpose) = 0;
    virtual void copy_tile(
        uint32_t in_cb_id, 
        uint32_t in_tile_index, 
        uint32_t dst_tile_index) = 0;
    virtual void copy_block_matmul_partials(
        uint32_t in_cb_id, 
        uint32_t start_in_tile_index, 
        uint32_t start_dst_tile_index, 
        uint32_t ntiles) = 0;
    // tilize
    virtual void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) = 0;
    virtual void tilize_init_short(uint32_t icb, uint32_t block) = 0;
    virtual void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb) = 0;
    virtual void tilize_uninit(uint32_t icb) = 0;
    // transpose_wh
    virtual void transpose_wh_init(uint32_t icb, uint32_t ocb) = 0;
    virtual void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst) = 0;
    // untilize
    virtual void untilize_init(uint32_t icb, uint32_t ocb) = 0;
    virtual void untilize_init_short(uint32_t icb) = 0;
    virtual void untilize_block(uint32_t N, uint32_t icb, uint32_t block, uint32_t ocb) = 0;
    virtual void untilize_uninit(uint32_t icb) = 0;
    // pack_untilize
    virtual void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim) = 0;
    virtual void pack_untilize_block(
        uint32_t icb, 
        uint32_t block_rt_dim, 
        uint32_t ocb,
        uint32_t block_ct_dim) = 0;
    virtual void pack_untilize_uninit(uint32_t ocb) = 0;
    virtual void pack_untilize_dst_init_short(
        uint32_t ocb, 
        uint32_t face_r_dim, 
        uint32_t num_faces,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim, 
        bool diagonal) = 0;
    virtual void pack_untilize_dst(
        uint32_t ocb, 
        uint32_t block_rt_dim, 
        uint32_t block_c_index, 
        uint32_t face_r_dim, 
        uint32_t num_faces,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim, 
        bool diagonal) = 0;
    virtual void pack_untilize_init_short(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim) = 0;
    // copy_dest_values
    virtual void copy_dest_values_init() = 0;
    virtual void copy_dest_values(uint32_t idst0, uint32_t idst1) = 0;
    // eltwise_binary_sfpu
    virtual void add_binary_tile_init() = 0;
    virtual void add_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void sub_binary_tile_init() = 0;
    virtual void sub_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void mul_binary_tile_init() = 0;
    virtual void mul_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void div_binary_tile_init() = 0;
    virtual void div_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void rsub_binary_tile_init() = 0;
    virtual void rsub_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void power_binary_tile_init() = 0;
    virtual void power_binary_tile(uint32_t idst0, uint32_t idst1) = 0;
    // eltwise_unary_sfpu
    virtual void rsqrt_tile_init() = 0;
    virtual void rsqrt_tile(uint32_t idst, bool fast_and_approx) = 0;
    virtual void sigmoid_tile_init() = 0;
    virtual void sigmoid_tile(uint32_t idst) = 0;
    virtual void log_tile_init() = 0;
    virtual void log_tile(uint32_t idst) = 0;
    virtual void log_with_base_tile_init() = 0;
    virtual void log_with_base_tile(uint32_t idst, uint32_t base_scale) = 0;
    virtual void tanh_tile_init() = 0;
    virtual void tanh_tile(uint32_t idst) = 0;
    virtual void signbit_tile_init() = 0;
    virtual void signbit_tile(uint32_t idst) = 0;
    virtual void abs_tile_init() = 0;
    virtual void abs_tile(uint32_t idst) = 0;
    virtual void sign_tile_init() = 0;
    virtual void sign_tile(uint32_t idst) = 0;
    virtual void square_tile_init() = 0;
    virtual void square_tile(uint32_t idst) = 0;
    virtual void ltz_tile_init() = 0;
    virtual void ltz_tile(uint32_t idst) = 0;
    virtual void eqz_tile_init() = 0;
    virtual void eqz_tile(uint32_t idst) = 0;
    virtual void lez_tile_init() = 0;
    virtual void lez_tile(uint32_t idst) = 0;
    virtual void gtz_tile_init() = 0;
    virtual void gtz_tile(uint32_t idst) = 0;
    virtual void nez_tile_init() = 0;
    virtual void nez_tile(uint32_t idst) = 0;
    virtual void gez_tile_init() = 0;
    virtual void gez_tile(uint32_t idst) = 0;
    virtual void power_tile_init() = 0;
    virtual void power_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void max_tile_init() = 0;
    virtual void max_tile(uint32_t idst0, uint32_t idst1) = 0;
    virtual void exp2_tile_init() = 0;
    virtual void exp2_tile(uint32_t idst) = 0;
    virtual void heaviside_tile_init() = 0;
    virtual void heaviside_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void expm1_tile_init() = 0;
    virtual void expm1_tile(uint32_t idst) = 0;
    virtual void asin_tile_init() = 0;
    virtual void asin_tile(uint32_t idst) = 0;
    virtual void atan_tile_init() = 0;
    virtual void atan_tile(uint32_t idst) = 0;
    virtual void acos_tile_init() = 0;
    virtual void acos_tile(uint32_t idst) = 0;
    // eltwise_unary/binop_with_scalar
    virtual void binop_with_scalar_tile_init() = 0;
    virtual void add_unary_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void sub_unary_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void mul_unary_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void div_unary_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void rsub_unary_tile(uint32_t idst, uint32_t param0) = 0;
    // eltwise_unary/ceil
    virtual void ceil_tile_init() = 0;
    virtual void ceil_tile(uint32_t idst) = 0;
    virtual void ceil_tile_float32(uint32_t idst) = 0;
    // eltwise_unary/elu
    virtual void elu_tile_init() = 0;
    virtual void elu_tile(uint32_t idst, uint32_t param0) = 0;
    // eltwise_unary/erf_erfc
    virtual void erf_tile_init() = 0; 
    virtual void erf_tile(uint32_t idst, bool fast_and_approx) = 0;
    virtual void erfc_tile_init() = 0; 
    virtual void erfc_tile(uint32_t idst, bool fast_and_approx) = 0;
    // eltwise_unary/erfinv
    virtual void erfinv_tile_init() = 0;
    virtual void erfinv_tile(uint32_t idst) = 0;
    // eltwise_unary/exp
    virtual void exp_tile_init() = 0;
    virtual void exp_tile(uint32_t idst) = 0;
    // eltwise_unary/fill
    virtual void fill_tile_init() = 0;
    virtual void fill_tile_bitcast(uint32_t idst, uint32_t param) = 0;
    // eltwise_unary/floor
    virtual void floor_tile_init() = 0;
    virtual void floor_tile(uint32_t idst) = 0;
    virtual void floor_tile_float32(uint32_t idst) = 0;
    // eltwise_unary/gelu
    virtual void gelu_tile_init() = 0;
    virtual void gelu_tile(uint32_t idst, bool fast_and_approx) = 0;
    // eltwise_unary/i0
    virtual void i0_tile_init() = 0;
    virtual void i0_tile(uint32_t idst) = 0;
    // eltwise_unary/isinf_isnan
    virtual void isinf_tile_init() = 0;
    virtual void isinf_tile(uint32_t idst) = 0;
    virtual void isposinf_tile_init() = 0;
    virtual void isposinf_tile(uint32_t idst) = 0;
    virtual void isneginf_tile_init() = 0;
    virtual void isneginf_tile(uint32_t idst) = 0;
    virtual void isnan_tile_init() = 0;
    virtual void isnan_tile(uint32_t idst) = 0;
    virtual void isfinite_tile_init() = 0;
    virtual void isfinite_tile(uint32_t idst) = 0;
    // eltwise_unary/logical_not_noti
    virtual void logical_not_unary_tile_init() = 0;
    virtual void logical_not_unary_tile(uint32_t idst) = 0;
    // eltwise_unary/recip
    virtual void recip_tile_init() = 0;
    virtual void recip_tile(uint32_t idst) = 0;
    // eltwise_unary/relu
    virtual void relu_max_tile_init() = 0;
    virtual void relu_max_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void relu_min_tile_init() = 0;
    virtual void relu_min_tile(uint32_t idst, uint32_t param0) = 0;
    virtual void relu_tile_init() = 0;
    virtual void relu_tile(uint32_t idst) = 0;
    virtual void leaky_relu_tile_init() = 0;
    virtual void leaky_relu_tile(uint32_t idst, uint32_t param0) = 0;
    // eltwise_unary/sqrt
    virtual void sqrt_tile_init() = 0;
    virtual void sqrt_tile(uint32_t idst) = 0;
    // eltwise_unary/trigonometry
    virtual void sin_tile_init() = 0;
    virtual void sin_tile(uint32_t idst) = 0;
    virtual void cos_tile_init() = 0;
    virtual void cos_tile(uint32_t idst) = 0;
    virtual void tan_tile_init() = 0;
    virtual void tan_tile(uint32_t idst) = 0;
    // eltwise_unary/typecast.h
    virtual void typecast_tile_init() = 0;
    virtual void typecast_tile(uint32_t in_dtype, uint32_t out_dtype, uint32_t idst) = 0;
    // Tanto extensions: math
    virtual void tanto_compute_init() = 0;
    virtual void tanto_copy_init() = 0;
    virtual void tanto_add_init() = 0;
    virtual void tanto_sub_init() = 0;
    virtual void tanto_mul_init() = 0;
    virtual void tanto_add_bcast_rows_init() = 0;
    virtual void tanto_sub_bcast_rows_init() = 0;
    virtual void tanto_mul_bcast_rows_init() = 0;
    virtual void tanto_add_bcast_cols_init() = 0;
    virtual void tanto_sub_bcast_cols_init() = 0;
    virtual void tanto_mul_bcast_cols_init() = 0;
    virtual void tanto_add_bcast_scalar_init() = 0;
    virtual void tanto_sub_bcast_scalar_init() = 0;
    virtual void tanto_mul_bcast_scalar_init() = 0;
    virtual void tanto_matmul_init(bool transpose) = 0;
    virtual void tanto_reduce_max_rows_init() = 0;
    virtual void tanto_reduce_max_cols_init() = 0;
    virtual void tanto_reduce_max_scalar_init() = 0;
    virtual void tanto_reduce_sum_rows_init() = 0;
    virtual void tanto_reduce_sum_cols_init() = 0;
    virtual void tanto_reduce_sum_scalar_init() = 0;
    virtual void tanto_transpose_init() = 0;
    virtual void tanto_tilize_block_init() = 0;
    virtual void tanto_untilize_block_init() = 0;
    virtual void tanto_copy_dst_init() = 0;
    virtual void tanto_add_dst_init() = 0;
    virtual void tanto_sub_dst_init() = 0;
    virtual void tanto_rsub_dst_init() = 0;
    virtual void tanto_mul_dst_init() = 0;
    virtual void tanto_div_dst_init() = 0;
    virtual void tanto_power_dst_init() = 0;
    virtual void tanto_abs_init() = 0;
    virtual void tanto_acos_init() = 0;
    virtual void tanto_asin_init() = 0;
    virtual void tanto_atan_init() = 0;
    virtual void tanto_binary_scalar_init() = 0;
    virtual void tanto_cast_init() = 0;
    virtual void tanto_ceil_init() = 0;
    virtual void tanto_cos_init() = 0;
    virtual void tanto_elu_init() = 0;
    virtual void tanto_eqz_init() = 0;
    virtual void tanto_erf_init() = 0;
    virtual void tanto_erfc_init() = 0;
    virtual void tanto_erfinv_init() = 0;
    virtual void tanto_exp_init() = 0;
    virtual void tanto_exp2_init() = 0;
    virtual void tanto_expm1_init() = 0;
    virtual void tanto_fill_init() = 0;
    virtual void tanto_floor_init() = 0;
    virtual void tanto_gelu_init() = 0;
    virtual void tanto_gez_init() = 0;
    virtual void tanto_gtz_init() = 0;
    virtual void tanto_heaviside_init() = 0;
    virtual void tanto_i0_init() = 0;
    virtual void tanto_isfinite_init() = 0;
    virtual void tanto_isinf_init() = 0;
    virtual void tanto_isnan_init() = 0;
    virtual void tanto_isneginf_init() = 0;
    virtual void tanto_isposinf_init() = 0;
    virtual void tanto_leaky_relu_init() = 0;
    virtual void tanto_lez_init() = 0;
    virtual void tanto_log_init() = 0;
    virtual void tanto_log_with_base_init() = 0;
    virtual void tanto_logical_not_init() = 0;
    virtual void tanto_ltz_init() = 0;
    virtual void tanto_max_init() = 0;
    virtual void tanto_nez_init() = 0;
    virtual void tanto_power_init() = 0;
    virtual void tanto_recip_init() = 0;
    virtual void tanto_relu_init() = 0;
    virtual void tanto_relu_max_init() = 0;
    virtual void tanto_relu_min_init() = 0;
    virtual void tanto_rsqrt_init() = 0;
    virtual void tanto_sigmoid_init() = 0;
    virtual void tanto_sign_init() = 0;
    virtual void tanto_signbit_init() = 0;
    virtual void tanto_sin_init() = 0;
    virtual void tanto_sqrt_init() = 0;
    virtual void tanto_square_init() = 0;
    virtual void tanto_tan_init() = 0;
    virtual void tanto_tanh_init() = 0;
    // Tanto extensions: unpack
    virtual void tanto_unpack_binary_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_bcast_rows_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_bcast_cols_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_bcast_scalar_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_matmul_init(uint32_t icb0, uint32_t icb1, bool transpose) = 0;
    virtual void tanto_unpack_unary_init(uint32_t icb) = 0;
    virtual void tanto_unpack_reduce_rows_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_reduce_cols_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_reduce_scalar_init(uint32_t icb0, uint32_t icb1) = 0;
    virtual void tanto_unpack_tilize_block_init(uint32_t icb, uint32_t block) = 0;
    virtual void tanto_unpack_transpose_init(uint32_t icb) = 0;
    virtual void tanto_unpack_untilize_block_init(uint32_t icb) = 0;
    // Tanto extensions: pack
    virtual void tanto_pack_init(uint32_t ocb) = 0;
    virtual void tanto_pack_row_init(uint32_t ocb) = 0;
    virtual void tanto_pack_col_init(uint32_t ocb) = 0;
    virtual void tanto_pack_scalar_init(uint32_t ocb) = 0;
};

} // namespace device
} // namespace metal
} // namespace tt

