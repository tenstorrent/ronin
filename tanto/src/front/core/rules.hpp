// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>

#include "clang/Tooling/Transformer/RewriteRule.h"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

//
//    Rule factory
//

class RuleFactory {
public:
    RuleFactory();
    ~RuleFactory();
public:
    void set_write_mode(bool write_mode);
    // style
    RewriteRule make_tidy_if_stmt_rule();
    RewriteRule make_cleanup_if_stmt_rule();
    // top level
    RewriteRule make_param_rule(std::function<uint32_t ()> get_value);
    // functions
    RewriteRule make_parm_global_rule();
    RewriteRule make_parm_local_rule();
    RewriteRule make_parm_semaphore_rule();
    RewriteRule make_parm_pipe_rule();
    RewriteRule make_parm_math_rule();
    // common
    RewriteRule make_pipe_set_frame_rule();
    RewriteRule make_pipe_wait_front_rule();
    RewriteRule make_pipe_pop_front_rule();
    RewriteRule make_pipe_reserve_back_rule();
    RewriteRule make_pipe_push_back_rule();
    // compute: math
    RewriteRule make_math_decl_rule();
    RewriteRule make_math_pack_rule();
    RewriteRule make_math_pack_row_rule();
    RewriteRule make_math_pack_col_rule();
    RewriteRule make_math_pack_scalar_rule();
    RewriteRule make_math_copy_rule();
    RewriteRule make_math_add_rule();
    RewriteRule make_math_sub_rule();
    RewriteRule make_math_mul_rule();
    RewriteRule make_math_add_bcast_rows_rule();
    RewriteRule make_math_sub_bcast_rows_rule();
    RewriteRule make_math_mul_bcast_rows_rule();
    RewriteRule make_math_add_bcast_cols_rule();
    RewriteRule make_math_sub_bcast_cols_rule();
    RewriteRule make_math_mul_bcast_cols_rule();
    RewriteRule make_math_add_bcast_scalar_rule();
    RewriteRule make_math_sub_bcast_scalar_rule();
    RewriteRule make_math_mul_bcast_scalar_rule();
    RewriteRule make_math_matmul_rule();
    RewriteRule make_math_reduce_max_rows_rule();
    RewriteRule make_math_reduce_max_cols_rule();
    RewriteRule make_math_reduce_max_scalar_rule();
    RewriteRule make_math_reduce_sum_rows_rule();
    RewriteRule make_math_reduce_sum_cols_rule();
    RewriteRule make_math_reduce_sum_scalar_rule();
    RewriteRule make_math_transpose_rule();
    RewriteRule make_math_abs_rule();
    RewriteRule make_math_acos_rule();
    RewriteRule make_math_add_scalar_rule();
    RewriteRule make_math_asin_rule();
    RewriteRule make_math_atan_rule();
    RewriteRule make_math_cos_rule();
    RewriteRule make_math_div_scalar_rule();
    RewriteRule make_math_elu_rule();
    RewriteRule make_math_eqz_rule();
    RewriteRule make_math_erf_rule();
    RewriteRule make_math_erfc_rule();
    RewriteRule make_math_erfinv_rule();
    RewriteRule make_math_exp_rule();
    RewriteRule make_math_exp2_rule();
    RewriteRule make_math_expm1_rule();
    RewriteRule make_math_gelu_rule();
    RewriteRule make_math_gez_rule();
    RewriteRule make_math_gtz_rule();
    RewriteRule make_math_heaviside_rule();
    RewriteRule make_math_i0_rule();
    RewriteRule make_math_isfinite_rule();
    RewriteRule make_math_isinf_rule();
    RewriteRule make_math_isnan_rule();
    RewriteRule make_math_isneginf_rule();
    RewriteRule make_math_isposinf_rule();
    RewriteRule make_math_leaky_relu_rule();
    RewriteRule make_math_lez_rule();
    RewriteRule make_math_log_rule();
    RewriteRule make_math_log_with_base_rule();
    RewriteRule make_math_logical_not_rule();
    RewriteRule make_math_ltz_rule();
    RewriteRule make_math_max_rule();
    RewriteRule make_math_mul_scalar_rule();
    RewriteRule make_math_nez_rule();
    RewriteRule make_math_power_rule();
    RewriteRule make_math_recip_rule();
    RewriteRule make_math_relu_rule();
    RewriteRule make_math_relu_max_rule();
    RewriteRule make_math_relu_min_rule();
    RewriteRule make_math_rsqrt_rule();
    RewriteRule make_math_rsub_scalar_rule();
    RewriteRule make_math_sigmoid_rule();
    RewriteRule make_math_sign_rule();
    RewriteRule make_math_signbit_rule();
    RewriteRule make_math_sin_rule();
    RewriteRule make_math_sqrt_rule();
    RewriteRule make_math_square_rule();
    RewriteRule make_math_sub_scalar_rule();
    RewriteRule make_math_tan_rule();
    RewriteRule make_math_tanh_rule();
    RewriteRule make_math_arg_rule();
    // compute: functions
    RewriteRule make_func_tilize_block_rule();
    RewriteRule make_func_untilize_block_rule();
    // compute: init
    RewriteRule make_copy_init_rule();
    RewriteRule make_add_init_rule();
    RewriteRule make_sub_init_rule();
    RewriteRule make_mul_init_rule();
    RewriteRule make_add_bcast_rows_init_rule();
    RewriteRule make_sub_bcast_rows_init_rule();
    RewriteRule make_mul_bcast_rows_init_rule();
    RewriteRule make_add_bcast_cols_init_rule();
    RewriteRule make_sub_bcast_cols_init_rule();
    RewriteRule make_mul_bcast_cols_init_rule();
    RewriteRule make_add_bcast_scalar_init_rule();
    RewriteRule make_sub_bcast_scalar_init_rule();
    RewriteRule make_mul_bcast_scalar_init_rule();
    RewriteRule make_matmul_init_rule();
    RewriteRule make_reduce_max_rows_init_rule();
    RewriteRule make_reduce_max_cols_init_rule();
    RewriteRule make_reduce_max_scalar_init_rule();
    RewriteRule make_reduce_sum_rows_init_rule();
    RewriteRule make_reduce_sum_cols_init_rule();
    RewriteRule make_reduce_sum_scalar_init_rule();
    RewriteRule make_transpose_init_rule();
    RewriteRule make_tilize_block_init_rule();
    RewriteRule make_untilize_block_init_rule();
    RewriteRule make_abs_init_rule();
    RewriteRule make_acos_init_rule();
    RewriteRule make_asin_init_rule();
    RewriteRule make_atan_init_rule();
    RewriteRule make_binary_scalar_init_rule();
    RewriteRule make_cos_init_rule();
    RewriteRule make_elu_init_rule();
    RewriteRule make_eqz_init_rule();
    RewriteRule make_erf_init_rule();
    RewriteRule make_erfc_init_rule();
    RewriteRule make_erfinv_init_rule();
    RewriteRule make_exp_init_rule();
    RewriteRule make_exp2_init_rule();
    RewriteRule make_expm1_init_rule();
    RewriteRule make_gelu_init_rule();
    RewriteRule make_gez_init_rule();
    RewriteRule make_gtz_init_rule();
    RewriteRule make_heaviside_init_rule();
    RewriteRule make_i0_init_rule();
    RewriteRule make_isfinite_init_rule();
    RewriteRule make_isinf_init_rule();
    RewriteRule make_isnan_init_rule();
    RewriteRule make_isneginf_init_rule();
    RewriteRule make_isposinf_init_rule();
    RewriteRule make_leaky_relu_init_rule();
    RewriteRule make_lez_init_rule();
    RewriteRule make_log_init_rule();
    RewriteRule make_log_with_base_init_rule();
    RewriteRule make_logical_not_init_rule();
    RewriteRule make_ltz_init_rule();
    RewriteRule make_max_init_rule();
    RewriteRule make_nez_init_rule();
    RewriteRule make_power_init_rule();
    RewriteRule make_recip_init_rule();
    RewriteRule make_relu_init_rule();
    RewriteRule make_relu_max_init_rule();
    RewriteRule make_relu_min_init_rule();
    RewriteRule make_rsqrt_init_rule();
    RewriteRule make_sigmoid_init_rule();
    RewriteRule make_sign_init_rule();
    RewriteRule make_signbit_init_rule();
    RewriteRule make_sin_init_rule();
    RewriteRule make_sqrt_init_rule();
    RewriteRule make_square_init_rule();
    RewriteRule make_tan_init_rule();
    RewriteRule make_tanh_init_rule();
    RewriteRule make_unpack_binary_init_rule();
    RewriteRule make_unpack_bcast_rows_init_rule();
    RewriteRule make_unpack_bcast_cols_init_rule();
    RewriteRule make_unpack_bcast_scalar_init_rule();
    RewriteRule make_unpack_matmul_init_rule();
    RewriteRule make_unpack_unary_init_rule();
    RewriteRule make_unpack_reduce_rows_init_rule();
    RewriteRule make_unpack_reduce_cols_init_rule();
    RewriteRule make_unpack_reduce_scalar_init_rule();
    RewriteRule make_unpack_tilize_block_init_rule();
    RewriteRule make_unpack_transpose_init_rule();
    RewriteRule make_unpack_untilize_block_init_rule();
    RewriteRule make_pack_init_rule();
    RewriteRule make_pack_row_init_rule();
    RewriteRule make_pack_col_init_rule();
    RewriteRule make_pack_scalar_init_rule();
    // dataflow: local
    RewriteRule make_local_get_rule();
    RewriteRule make_local_set_rule();
    RewriteRule make_local_read_global_rule();
    RewriteRule make_local_read_global_dist_rule();
    RewriteRule make_local_read_local_rule();
    RewriteRule make_local_read_local_xy_rule();
    RewriteRule make_local_read_pipe_rule();
    RewriteRule make_local_read_pipe_xy_rule();
    RewriteRule make_local_write_global_rule();
    RewriteRule make_local_write_global_dist_rule();
    RewriteRule make_local_write_local_rule();
    RewriteRule make_local_write_local_xy_rule();
    RewriteRule make_local_write_pipe_rule();
    RewriteRule make_local_write_pipe_xy_rule();
    RewriteRule make_local_write_mcast_local_rule();
    RewriteRule make_local_write_mcast_with_self_local_rule();
    RewriteRule make_local_write_mcast_pipe_rule();
    RewriteRule make_local_write_mcast_with_self_pipe_rule();
    RewriteRule make_local_move_init_rule();
    RewriteRule make_local_move_local_rule();
    RewriteRule make_local_move_pipe_rule();
    // dataflow: pipe
    RewriteRule make_pipe_read_global_rule();
    RewriteRule make_pipe_read_global_dist_rule();
    RewriteRule make_pipe_read_local_rule();
    RewriteRule make_pipe_read_local_xy_rule();
    RewriteRule make_pipe_read_pipe_rule();
    RewriteRule make_pipe_read_pipe_xy_rule();
    RewriteRule make_pipe_write_global_rule();
    RewriteRule make_pipe_write_global_dist_rule();
    RewriteRule make_pipe_write_local_rule();
    RewriteRule make_pipe_write_local_xy_rule();
    RewriteRule make_pipe_write_pipe_rule();
    RewriteRule make_pipe_write_pipe_xy_rule();
    RewriteRule make_pipe_write_mcast_local_rule();
    RewriteRule make_pipe_write_mcast_with_self_local_rule();
    RewriteRule make_pipe_write_mcast_pipe_rule();
    RewriteRule make_pipe_write_mcast_with_self_pipe_rule();
    RewriteRule make_pipe_move_init_rule();
    RewriteRule make_pipe_move_local_rule();
    RewriteRule make_pipe_move_pipe_rule();
    // dataflow: semaphore
    RewriteRule make_semaphore_set_rule();
    RewriteRule make_semaphore_set_remote_rule();
    RewriteRule make_semaphore_set_mcast_rule();
    RewriteRule make_semaphore_inc_rule();
    RewriteRule make_semaphore_wait_rule();
    // dataflow: functions
    RewriteRule make_func_read_barrier_rule();
    RewriteRule make_func_write_barrier_rule();
private:
    bool m_write_mode;
};

} // namespace front
} // namespace tanto
} // namespace ronin

