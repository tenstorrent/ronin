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
#if 0 // TODO: Revise this
    virtual void mm_init_once() = 0;
#endif
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
#if 0 // TODO: Revise this
    virtual void copy_tile_matmul_partials_init_short_with_dt(uint32_t cbid) = 0;
#endif
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
#if 0 // SKIPPED
    virtual void graph_interpreter_init() = 0; 
    virtual void get_next_op_info(op_info_t &op_info) = 0;
#endif
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
    // Tanto extensions: math
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
    virtual void tanto_abs_init() = 0;
    virtual void tanto_acos_init() = 0;
    virtual void tanto_asin_init() = 0;
    virtual void tanto_atan_init() = 0;
    virtual void tanto_cos_init() = 0;
    virtual void tanto_elu_init() = 0;
    virtual void tanto_eqz_init() = 0;
    virtual void tanto_erf_init() = 0;
    virtual void tanto_erfc_init() = 0;
    virtual void tanto_erfinv_init() = 0;
    virtual void tanto_exp_init() = 0;
    virtual void tanto_exp2_init() = 0;
    virtual void tanto_expm1_init() = 0;
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
    virtual void tanto_unpack_tilize_block_init(uint32_t icb, uint32_t block) = 0;
    virtual void tanto_unpack_transpose_init(uint32_t icb) = 0;
    virtual void tanto_unpack_untilize_block_init(uint32_t icb) = 0;
    // Tanto extensions: pack
    virtual void tanto_pack_init(uint32_t ocb) = 0;
    virtual void tanto_pack_reduce_rows_init(uint32_t ocb) = 0;
    virtual void tanto_pack_reduce_cols_init(uint32_t ocb) = 0;
    virtual void tanto_pack_reduce_scalar_init(uint32_t ocb) = 0;
};

} // namespace device
} // namespace metal
} // namespace tt

