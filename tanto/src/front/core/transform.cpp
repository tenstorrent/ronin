// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>

#include "core/error.hpp"
#include "core/matchers.hpp"
#include "core/rules.hpp"
#include "core/tooling.hpp"
#include "core/transform.hpp"

namespace ronin {
namespace tanto {
namespace front {

using namespace clang;
using namespace transformer;

//
//    Transform
//

Transform::Transform():
        m_error_handler(nullptr),
        m_next_param_index(0),
        m_rewrite_ok(false) { 
    m_pass1_rule = make_pass1_rule();
    m_pass2_compute_rule = make_pass2_compute_rule();
    m_pass2_read_rule = make_pass2_dataflow_rule(false);
    m_pass2_write_rule = make_pass2_dataflow_rule(true);
}

Transform::~Transform() { }

void Transform::set_error_handler(ErrorHandler *error_handler) {
    m_error_handler = error_handler;
}

void Transform::reset() {
    m_param_map.clear();
}

bool Transform::add_param(uint32_t index, uint32_t value) {
    auto ret = m_param_map.emplace(index, value);
    if (!ret.second) {
        error("Duplicate value for parameter #" + std::to_string(index));
        return false;
    }
    return true;
}

bool Transform::pass1(const std::string &input_code, std::string &output_code) {
    m_next_param_index = 0;
    return rewrite(m_pass1_rule, input_code, output_code);
}

bool Transform::pass2_compute(const std::string &input_code, std::string &output_code) {
    return rewrite(m_pass2_compute_rule, input_code, output_code);
}

bool Transform::pass2_dataflow(
        const std::string &input_code, 
        std::string &output_code,
        bool write_mode) {
    if (!write_mode) {
        return rewrite(m_pass2_read_rule, input_code, output_code);
    } else {
        return rewrite(m_pass2_write_rule, input_code, output_code);
    }
}

bool Transform::rewrite(
        RewriteRule rule,
        const std::string &input_code, 
        std::string &output_code) {
    m_rewrite_ok = true;
    TransformerTool transformer_tool;
    transformer_tool.set_error_handler(m_error_handler);
    if (!transformer_tool.run(rule, input_code, output_code)) {
        return false;
    }
    return m_rewrite_ok;
}

RewriteRule Transform::make_pass1_rule() {
    RuleFactory &rf = m_rule_factory;
    return applyFirst({
        rf.make_param_rule([this]() -> uint32_t {
            return get_param_value();
        }),
        rf.make_tidy_if_stmt_rule()
    });
}

RewriteRule Transform::make_pass2_compute_rule() {
    RuleFactory &rf = m_rule_factory;
    return applyFirst({
        // style
        rf.make_cleanup_if_stmt_rule(),
        // functions
        rf.make_parm_global_rule(),
        rf.make_parm_local_rule(),
        rf.make_parm_semaphore_rule(),
        rf.make_parm_pipe_rule(),
        rf.make_parm_math_rule(),
        // pipe
        rf.make_pipe_set_frame_rule(),
        rf.make_pipe_wait_front_rule(),
        rf.make_pipe_pop_front_rule(),
        rf.make_pipe_reserve_back_rule(),
        rf.make_pipe_push_back_rule(),
        // math
        rf.make_math_decl_rule(),
        rf.make_math_pack_rule(),
        rf.make_math_pack_row_rule(),
        rf.make_math_pack_col_rule(),
        rf.make_math_pack_scalar_rule(),
        rf.make_math_copy_rule(),
        rf.make_math_add_rule(),
        rf.make_math_sub_rule(),
        rf.make_math_mul_rule(),
        rf.make_math_add_bcast_rows_rule(),
        rf.make_math_sub_bcast_rows_rule(),
        rf.make_math_mul_bcast_rows_rule(),
        rf.make_math_add_bcast_cols_rule(),
        rf.make_math_sub_bcast_cols_rule(),
        rf.make_math_mul_bcast_cols_rule(),
        rf.make_math_add_bcast_scalar_rule(),
        rf.make_math_sub_bcast_scalar_rule(),
        rf.make_math_mul_bcast_scalar_rule(),
        rf.make_math_matmul_rule(),
        rf.make_math_reduce_max_rows_rule(),
        rf.make_math_reduce_max_cols_rule(),
        rf.make_math_reduce_max_scalar_rule(),
        rf.make_math_reduce_sum_rows_rule(),
        rf.make_math_reduce_sum_cols_rule(),
        rf.make_math_reduce_sum_scalar_rule(),
        rf.make_math_transpose_rule(),
        rf.make_math_abs_rule(),
        rf.make_math_acos_rule(),
        rf.make_math_add_scalar_rule(),
        rf.make_math_asin_rule(),
        rf.make_math_atan_rule(),
        rf.make_math_cos_rule(),
        rf.make_math_div_scalar_rule(),
        rf.make_math_elu_rule(),
        rf.make_math_eqz_rule(),
        rf.make_math_erf_rule(),
        rf.make_math_erfc_rule(),
        rf.make_math_erfinv_rule(),
        rf.make_math_exp_rule(),
        rf.make_math_exp2_rule(),
        rf.make_math_expm1_rule(),
        rf.make_math_gelu_rule(),
        rf.make_math_gez_rule(),
        rf.make_math_gtz_rule(),
        rf.make_math_heaviside_rule(),
        rf.make_math_i0_rule(),
        rf.make_math_isfinite_rule(),
        rf.make_math_isinf_rule(),
        rf.make_math_isnan_rule(),
        rf.make_math_isneginf_rule(),
        rf.make_math_isposinf_rule(),
        rf.make_math_leaky_relu_rule(),
        rf.make_math_lez_rule(),
        rf.make_math_log_rule(),
        rf.make_math_log_with_base_rule(),
        rf.make_math_logical_not_rule(),
        rf.make_math_ltz_rule(),
        rf.make_math_max_rule(),
        rf.make_math_mul_scalar_rule(),
        rf.make_math_nez_rule(),
        rf.make_math_power_rule(),
        rf.make_math_recip_rule(),
        rf.make_math_relu_rule(),
        rf.make_math_relu_max_rule(),
        rf.make_math_relu_min_rule(),
        rf.make_math_rsqrt_rule(),
        rf.make_math_rsub_scalar_rule(),
        rf.make_math_sigmoid_rule(),
        rf.make_math_sign_rule(),
        rf.make_math_signbit_rule(),
        rf.make_math_sin_rule(),
        rf.make_math_sqrt_rule(),
        rf.make_math_square_rule(),
        rf.make_math_sub_scalar_rule(),
        rf.make_math_tan_rule(),
        rf.make_math_tanh_rule(),
        rf.make_math_arg_rule(),
        // global functions
        rf.make_func_tilize_block_rule(),
        rf.make_func_untilize_block_rule(),
        // init
        rf.make_copy_init_rule(),
        rf.make_add_init_rule(),
        rf.make_sub_init_rule(),
        rf.make_mul_init_rule(),
        rf.make_add_bcast_rows_init_rule(),
        rf.make_sub_bcast_rows_init_rule(),
        rf.make_mul_bcast_rows_init_rule(),
        rf.make_add_bcast_cols_init_rule(),
        rf.make_sub_bcast_cols_init_rule(),
        rf.make_mul_bcast_cols_init_rule(),
        rf.make_add_bcast_scalar_init_rule(),
        rf.make_sub_bcast_scalar_init_rule(),
        rf.make_mul_bcast_scalar_init_rule(),
        rf.make_matmul_init_rule(),
        rf.make_reduce_max_rows_init_rule(),
        rf.make_reduce_max_cols_init_rule(),
        rf.make_reduce_max_scalar_init_rule(),
        rf.make_reduce_sum_rows_init_rule(),
        rf.make_reduce_sum_cols_init_rule(),
        rf.make_reduce_sum_scalar_init_rule(),
        rf.make_transpose_init_rule(),
        rf.make_tilize_block_init_rule(),
        rf.make_untilize_block_init_rule(),
        rf.make_abs_init_rule(),
        rf.make_acos_init_rule(),
        rf.make_asin_init_rule(),
        rf.make_atan_init_rule(),
        rf.make_binary_scalar_init_rule(),
        rf.make_cos_init_rule(),
        rf.make_elu_init_rule(),
        rf.make_eqz_init_rule(),
        rf.make_erf_init_rule(),
        rf.make_erfc_init_rule(),
        rf.make_erfinv_init_rule(),
        rf.make_exp_init_rule(),
        rf.make_exp2_init_rule(),
        rf.make_expm1_init_rule(),
        rf.make_gelu_init_rule(),
        rf.make_gez_init_rule(),
        rf.make_gtz_init_rule(),
        rf.make_heaviside_init_rule(),
        rf.make_i0_init_rule(),
        rf.make_isfinite_init_rule(),
        rf.make_isinf_init_rule(),
        rf.make_isnan_init_rule(),
        rf.make_isneginf_init_rule(),
        rf.make_isposinf_init_rule(),
        rf.make_leaky_relu_init_rule(),
        rf.make_lez_init_rule(),
        rf.make_log_init_rule(),
        rf.make_log_with_base_init_rule(),
        rf.make_logical_not_init_rule(),
        rf.make_ltz_init_rule(),
        rf.make_max_init_rule(),
        rf.make_nez_init_rule(),
        rf.make_power_init_rule(),
        rf.make_recip_init_rule(),
        rf.make_relu_init_rule(),
        rf.make_relu_max_init_rule(),
        rf.make_relu_min_init_rule(),
        rf.make_rsqrt_init_rule(),
        rf.make_sigmoid_init_rule(),
        rf.make_sign_init_rule(),
        rf.make_signbit_init_rule(),
        rf.make_sin_init_rule(),
        rf.make_sqrt_init_rule(),
        rf.make_square_init_rule(),
        rf.make_tan_init_rule(),
        rf.make_tanh_init_rule(),
        rf.make_unpack_binary_init_rule(),
        rf.make_unpack_bcast_rows_init_rule(),
        rf.make_unpack_bcast_cols_init_rule(),
        rf.make_unpack_bcast_scalar_init_rule(),
        rf.make_unpack_matmul_init_rule(),
        rf.make_unpack_unary_init_rule(),
        rf.make_unpack_reduce_rows_init_rule(),
        rf.make_unpack_reduce_cols_init_rule(),
        rf.make_unpack_reduce_scalar_init_rule(),
        rf.make_unpack_tilize_block_init_rule(),
        rf.make_unpack_transpose_init_rule(),
        rf.make_unpack_untilize_block_init_rule(),
        rf.make_pack_init_rule(),
        rf.make_pack_row_init_rule(),
        rf.make_pack_col_init_rule(),
        rf.make_pack_scalar_init_rule()
    });
}

RewriteRule Transform::make_pass2_dataflow_rule(bool write_mode) {
    RuleFactory &rf = m_rule_factory;
    rf.set_write_mode(write_mode);
    return applyFirst({
        // style
        rf.make_cleanup_if_stmt_rule(),
        // top level
        rf.make_param_rule([this]() -> uint32_t {
            return get_param_value();
        }),
        // functions
        rf.make_parm_global_rule(),
        rf.make_parm_local_rule(),
        rf.make_parm_semaphore_rule(),
        rf.make_parm_pipe_rule(),
        rf.make_parm_math_rule(),
        // pipe (common)
        rf.make_pipe_set_frame_rule(),
        rf.make_pipe_wait_front_rule(),
        rf.make_pipe_pop_front_rule(),
        rf.make_pipe_reserve_back_rule(),
        rf.make_pipe_push_back_rule(),
        // local
        rf.make_local_get_rule(),
        rf.make_local_set_rule(),
        rf.make_local_read_global_rule(),
        rf.make_local_read_global_dist_rule(),
        rf.make_local_read_local_rule(),
        rf.make_local_read_local_xy_rule(),
        rf.make_local_read_pipe_rule(),
        rf.make_local_read_pipe_xy_rule(),
        rf.make_local_write_global_rule(),
        rf.make_local_write_global_dist_rule(),
        rf.make_local_write_local_rule(),
        rf.make_local_write_local_xy_rule(),
        rf.make_local_write_pipe_rule(),
        rf.make_local_write_pipe_xy_rule(),
        rf.make_local_write_mcast_local_rule(),
        rf.make_local_write_mcast_with_self_local_rule(),
        rf.make_local_write_mcast_pipe_rule(),
        rf.make_local_write_mcast_with_self_pipe_rule(),
        rf.make_local_move_init_rule(),
        rf.make_local_move_local_rule(),
        rf.make_local_move_pipe_rule(),
        // pipe (dataflow)
        rf.make_pipe_read_global_rule(),
        rf.make_pipe_read_global_dist_rule(),
        rf.make_pipe_read_local_rule(),
        rf.make_pipe_read_local_xy_rule(),
        rf.make_pipe_read_pipe_rule(),
        rf.make_pipe_read_pipe_xy_rule(),
        rf.make_pipe_write_global_rule(),
        rf.make_pipe_write_global_dist_rule(),
        rf.make_pipe_write_local_rule(),
        rf.make_pipe_write_local_xy_rule(),
        rf.make_pipe_write_pipe_rule(),
        rf.make_pipe_write_pipe_xy_rule(),
        rf.make_pipe_write_mcast_local_rule(),
        rf.make_pipe_write_mcast_with_self_local_rule(),
        rf.make_pipe_write_mcast_pipe_rule(),
        rf.make_pipe_write_mcast_with_self_pipe_rule(),
        rf.make_pipe_move_init_rule(),
        rf.make_pipe_move_local_rule(),
        rf.make_pipe_move_pipe_rule(),
        // semaphore
        rf.make_semaphore_set_rule(),
        rf.make_semaphore_set_remote_rule(),
        rf.make_semaphore_set_mcast_rule(),
        rf.make_semaphore_inc_rule(),
        rf.make_semaphore_wait_rule(),
        // global functions
        rf.make_func_read_barrier_rule(),
        rf.make_func_write_barrier_rule()
    });
}

uint32_t Transform::get_param_value() {
    uint32_t index = m_next_param_index;
    m_next_param_index++;
    auto it = m_param_map.find(index);
    if (it == m_param_map.end()) {
        error("Undefined parameter #" + std::to_string(index));
        m_rewrite_ok = false;
        return 1;
    }
    return it->second;
}

void Transform::error(const std::string &text) {
    if (m_error_handler != nullptr) {
        m_error_handler->error(text);
    }
}

} // namespace front
} // namespace tanto
} // namespace ronin

