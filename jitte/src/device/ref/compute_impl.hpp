// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/compute_api.hpp"

#include "ref/llk.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

class ComputeImpl: public Compute {
public:
    ComputeImpl(Memory *l1, CB *cb);
    ~ComputeImpl();
public:
    void reset() override;
    uint32_t get_arg_uint32(int arg_idx) override;
    // reg_api
    void acquire_dst(DstMode mode) override;
    void tile_regs_acquire() override;
    void tile_regs_wait() override;
    void release_dst(DstMode mode) override;
    void tile_regs_commit() override;
    void tile_regs_release() override;
    // pack
    void pack_relu_config(uint32_t config) override;
    void pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t output_tile_index,
        bool out_of_order_output) override;
    void matmul_pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t ntiles) override;
    void pack_reconfig_data_format(uint32_t new_operand) override;
    // unpack
    void unpack_reconfig_data_format(
        uint32_t srca_new_operand, 
        uint32_t srcb_new_operand) override;
    // cb_api
    void cb_wait_front(uint32_t cbid, uint32_t ntiles) override;
    void cb_pop_front(uint32_t cbid, uint32_t ntiles) override;
    void cb_reserve_back(uint32_t cbid, uint32_t ntiles) override;
    void cb_push_back(uint32_t cbid, uint32_t ntiles) override;
    uint32_t get_read_ptr(uint32_t cbid) override;
    uint32_t get_write_ptr(uint32_t cbid) override;
    void set_read_ptr(uint32_t cbid, uint32_t ptr) override;
    void set_write_ptr(uint32_t cbid, uint32_t ptr) override;
    // bcast
    void sub_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void mul_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void mul_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void add_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void add_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void init_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t ocb) override;
    void any_tiles_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void add_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void sub_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void mul_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) override;
    void add_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) override;
    void mul_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) override;
    void mul_tiles_bcast_scalar(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) override;
    void mul_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) override;
    void sub_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) override;
    // eltwise_binary
    void binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb) override;
    void mul_tiles_init_f() override;
    void mul_tiles_init(uint32_t icb0, uint32_t icb1) override;
    void add_tiles_init_nof() override;
    void add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) override;
    void sub_tiles_init_nof() override;
    void sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) override;
    void mul_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void add_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void sub_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    void binary_op_specific_init(
        bool full_init, 
        EltwiseBinaryType eltwise_binary_op_type) override;
    // eltwise_unary
    void unary_op_init_common(uint32_t icb) override;
    void init_sfpu(uint32_t icb) override;
    // matmul
    void mm_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose) override;
    void matmul_tiles(
        uint32_t c_in0, 
        uint32_t c_in1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst, 
        uint32_t transpose) override;
    void mm_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose) override;
    void mm_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t c_in_old_srca, 
        uint32_t transpose) override;
    void mm_block_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) override;
    void matmul_block(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t in0_tile_index, 
        uint32_t in1_tile_index, 
        uint32_t idst, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) override;
    void mm_block_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) override;
    void mm_block_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t old_in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) override;
    // reduce
    void reduce_init(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) override;
    void reduce_init_short(
        PoolType reduce_type, 
        ReduceDim reduce_dim,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) override;
    void reduce_init_delta(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t ocb, 
        uint32_t icb0, 
        uint32_t icb1) override;
    void reduce_revert_delta(ReduceDim reduce_dim, uint32_t ocb) override;
    void reduce_tile(
        PoolType reduce_type, 
        ReduceDim reduce_dim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) override;
    // tile_move_copy
    void copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose) override;
    void copy_tile_init() override;
    void copy_tile_to_dst_init_short_with_dt(
        uint32_t old_cbid, 
        uint32_t new_cbid, 
        uint32_t transpose) override;
    void copy_tile(
        uint32_t in_cb_id, 
        uint32_t in_tile_index, 
        uint32_t dst_tile_index) override;
    void copy_block_matmul_partials(
        uint32_t in_cb_id, 
        uint32_t start_in_tile_index, 
        uint32_t start_dst_tile_index, 
        uint32_t ntiles) override;
    // tilize
    void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) override;
    void tilize_init_short(uint32_t icb, uint32_t block) override;
    void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb) override;
    void tilize_uninit(uint32_t icb) override;
    // transpose_wh
    void transpose_wh_init(uint32_t icb, uint32_t ocb) override;
    void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst) override;
    // untilize
    void untilize_init(uint32_t icb, uint32_t ocb) override;
    void untilize_init_short(uint32_t icb) override;
    void untilize_block(uint32_t N, uint32_t icb, uint32_t block, uint32_t ocb) override;
    void untilize_uninit(uint32_t icb) override;
    // pack_untilize
    void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim) override;
    void pack_untilize_block(
        uint32_t icb, 
        uint32_t block_rt_dim, 
        uint32_t ocb,
        uint32_t block_ct_dim) override;
    void pack_untilize_uninit(uint32_t ocb) override;
    void pack_untilize_dst_init_short(
        uint32_t ocb, 
        uint32_t face_r_dim, 
        uint32_t num_faces,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim, 
        bool diagonal) override;
    void pack_untilize_dst(
        uint32_t ocb, 
        uint32_t block_rt_dim, 
        uint32_t block_c_index, 
        uint32_t face_r_dim, 
        uint32_t num_faces,
        uint32_t block_ct_dim, 
        uint32_t full_ct_dim, 
        bool diagonal) override;
    void pack_untilize_init_short(uint32_t icb, uint32_t ocb, uint32_t block_ct_dim) override;
    // eltwise_unary_sfpu
    void rsqrt_tile_init() override;
    void rsqrt_tile(uint32_t idst, bool fast_and_approx) override;
    void sigmoid_tile_init() override;
    void sigmoid_tile(uint32_t idst) override;
    void log_tile_init() override;
    void log_tile(uint32_t idst) override;
    void log_with_base_tile_init() override;
    void log_with_base_tile(uint32_t idst, uint32_t base_scale) override;
    void tanh_tile_init() override;
    void tanh_tile(uint32_t idst) override;
    void signbit_tile_init() override;
    void signbit_tile(uint32_t idst) override;
    void abs_tile_init() override;
    void abs_tile(uint32_t idst) override;
    void sign_tile_init() override;
    void sign_tile(uint32_t idst) override;
    void square_tile_init() override;
    void square_tile(uint32_t idst) override;
    void ltz_tile_init() override;
    void ltz_tile(uint32_t idst) override;
    void eqz_tile_init() override;
    void eqz_tile(uint32_t idst) override;
    void lez_tile_init() override;
    void lez_tile(uint32_t idst) override;
    void gtz_tile_init() override;
    void gtz_tile(uint32_t idst) override;
    void nez_tile_init() override;
    void nez_tile(uint32_t idst) override;
    void gez_tile_init() override;
    void gez_tile(uint32_t idst) override;
    void power_tile_init() override;
    void power_tile(uint32_t idst, uint32_t param0) override;
    void max_tile_init() override;
    void max_tile(uint32_t idst0, uint32_t idst1) override;
    void exp2_tile_init() override;
    void exp2_tile(uint32_t idst) override;
    void heaviside_tile_init() override;
    void heaviside_tile(uint32_t idst, uint32_t param0) override;
    void expm1_tile_init() override;
    void expm1_tile(uint32_t idst) override;
    void asin_tile_init() override;
    void asin_tile(uint32_t idst) override;
    void atan_tile_init() override;
    void atan_tile(uint32_t idst) override;
    void acos_tile_init() override;
    void acos_tile(uint32_t idst) override;
    // eltwise_unary/binop_with_scalar
    void binop_with_scalar_tile_init() override;
    void add_unary_tile(uint32_t idst, uint32_t param0) override;
    void sub_unary_tile(uint32_t idst, uint32_t param0) override;
    void mul_unary_tile(uint32_t idst, uint32_t param0) override;
    void div_unary_tile(uint32_t idst, uint32_t param0) override;
    void rsub_unary_tile(uint32_t idst, uint32_t param0) override;
    // eltwise_unary/elu
    void elu_tile_init() override;
    void elu_tile(uint32_t idst, uint32_t param0) override;
    // eltwise_unary/erf_erfc
    void erf_tile_init() override; 
    void erf_tile(uint32_t idst, bool fast_and_approx) override;
    void erfc_tile_init() override; 
    void erfc_tile(uint32_t idst, bool fast_and_approx) override;
    // eltwise_unary/erfinv
    void erfinv_tile_init() override;
    void erfinv_tile(uint32_t idst) override;
    // eltwise_unary/exp
    void exp_tile_init() override;
    void exp_tile(uint32_t idst) override;
    // eltwise_unary/gelu
    void gelu_tile_init() override;
    void gelu_tile(uint32_t idst, bool fast_and_approx) override;
    // eltwise_unary/i0
    void i0_tile_init() override;
    void i0_tile(uint32_t idst) override;
    // eltwise_unary/isinf_isnan
    void isinf_tile_init() override;
    void isinf_tile(uint32_t idst) override;
    void isposinf_tile_init() override;
    void isposinf_tile(uint32_t idst) override;
    void isneginf_tile_init() override;
    void isneginf_tile(uint32_t idst) override;
    void isnan_tile_init() override;
    void isnan_tile(uint32_t idst) override;
    void isfinite_tile_init() override;
    void isfinite_tile(uint32_t idst) override;
    // eltwise_unary/logical_not_noti
    void logical_not_unary_tile_init() override;
    void logical_not_unary_tile(uint32_t idst) override;
    // eltwise_unary/recip
    void recip_tile_init() override;
    void recip_tile(uint32_t idst) override;
    // eltwise_unary/relu
    void relu_max_tile_init() override;
    void relu_max_tile(uint32_t idst, uint32_t param0) override;
    void relu_min_tile_init() override;
    void relu_min_tile(uint32_t idst, uint32_t param0) override;
    void relu_tile_init() override;
    void relu_tile(uint32_t idst) override;
    void leaky_relu_tile_init() override;
    void leaky_relu_tile(uint32_t idst, uint32_t param0) override;
    // eltwise_unary/sqrt
    void sqrt_tile_init() override;
    void sqrt_tile(uint32_t idst) override;
    // eltwise_unary/trigonometry
    void sin_tile_init() override;
    void sin_tile(uint32_t idst) override;
    void cos_tile_init() override;
    void cos_tile(uint32_t idst) override;
    void tan_tile_init() override;
    void tan_tile(uint32_t idst) override;
    // Tanto extensions: math
    void tanto_compute_init() override;
    void tanto_copy_init() override;
    void tanto_add_init() override;
    void tanto_sub_init() override;
    void tanto_mul_init() override;
    void tanto_add_bcast_rows_init() override;
    void tanto_sub_bcast_rows_init() override;
    void tanto_mul_bcast_rows_init() override;
    void tanto_add_bcast_cols_init() override;
    void tanto_sub_bcast_cols_init() override;
    void tanto_mul_bcast_cols_init() override;
    void tanto_add_bcast_scalar_init() override;
    void tanto_sub_bcast_scalar_init() override;
    void tanto_mul_bcast_scalar_init() override;
    void tanto_matmul_init(bool transpose) override;
    void tanto_reduce_max_rows_init() override;
    void tanto_reduce_max_cols_init() override;
    void tanto_reduce_max_scalar_init() override;
    void tanto_reduce_sum_rows_init() override;
    void tanto_reduce_sum_cols_init() override;
    void tanto_reduce_sum_scalar_init() override;
    void tanto_transpose_init() override;
    void tanto_tilize_block_init() override;
    void tanto_untilize_block_init() override;
    void tanto_abs_init() override;
    void tanto_acos_init() override;
    void tanto_asin_init() override;
    void tanto_atan_init() override;
    void tanto_binary_scalar_init() override;
    void tanto_cos_init() override;
    void tanto_elu_init() override;
    void tanto_eqz_init() override;
    void tanto_erf_init() override;
    void tanto_erfc_init() override;
    void tanto_erfinv_init() override;
    void tanto_exp_init() override;
    void tanto_exp2_init() override;
    void tanto_expm1_init() override;
    void tanto_gelu_init() override;
    void tanto_gez_init() override;
    void tanto_gtz_init() override;
    void tanto_heaviside_init() override;
    void tanto_i0_init() override;
    void tanto_isfinite_init() override;
    void tanto_isinf_init() override;
    void tanto_isnan_init() override;
    void tanto_isneginf_init() override;
    void tanto_isposinf_init() override;
    void tanto_leaky_relu_init() override;
    void tanto_lez_init() override;
    void tanto_log_init() override;
    void tanto_log_with_base_init() override;
    void tanto_logical_not_init() override;
    void tanto_ltz_init() override;
    void tanto_max_init() override;
    void tanto_nez_init() override;
    void tanto_power_init() override;
    void tanto_recip_init() override;
    void tanto_relu_init() override;
    void tanto_relu_max_init() override;
    void tanto_relu_min_init() override;
    void tanto_rsqrt_init() override;
    void tanto_sigmoid_init() override;
    void tanto_sign_init() override;
    void tanto_signbit_init() override;
    void tanto_sin_init() override;
    void tanto_sqrt_init() override;
    void tanto_square_init() override;
    void tanto_tan_init() override;
    void tanto_tanh_init() override;
    // Tanto extensions: unpack
    void tanto_unpack_binary_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_bcast_rows_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_bcast_cols_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_bcast_scalar_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_matmul_init(uint32_t icb0, uint32_t icb1, bool transpose) override;
    void tanto_unpack_unary_init(uint32_t icb) override;
    void tanto_unpack_reduce_rows_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_reduce_cols_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_reduce_scalar_init(uint32_t icb0, uint32_t icb1) override;
    void tanto_unpack_tilize_block_init(uint32_t icb, uint32_t block) override;
    void tanto_unpack_transpose_init(uint32_t icb) override;
    void tanto_unpack_untilize_block_init(uint32_t icb) override;
    // Tanto extensions: pack
    void tanto_pack_init(uint32_t ocb) override;
    void tanto_pack_row_init(uint32_t ocb) override;
    void tanto_pack_col_init(uint32_t ocb) override;
    void tanto_pack_scalar_init(uint32_t ocb) override;
private:
    Memory *m_l1;
    CB *m_cb;
    LLK m_llk;
};

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

