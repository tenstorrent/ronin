// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "core/builtin_init.hpp"

namespace ronin {
namespace tanto {
namespace front {

namespace {

const char *g_builtin_init_header = R"cpp(

// math

void __copy_init();
void __add_init();
void __sub_init();
void __mul_init();
void __add_bcast_rows_init();
void __sub_bcast_rows_init();
void __mul_bcast_rows_init();
void __add_bcast_cols_init();
void __sub_bcast_cols_init();
void __mul_bcast_cols_init();
void __add_bcast_scalar_init();
void __sub_bcast_scalar_init();
void __mul_bcast_scalar_init();
void __matmul_init(bool transpose);
void __reduce_max_rows_init();
void __reduce_max_cols_init();
void __reduce_max_scalar_init();
void __reduce_sum_rows_init();
void __reduce_sum_cols_init();
void __reduce_sum_scalar_init();
void __transpose_init();
void __tilize_block_init();
void __untilize_block_init();

void __copy_dst_init();
void __add_dst_init();
void __sub_dst_init();
void __rsub_dst_init();
void __mul_dst_init();
void __div_dst_init();
void __power_dst_init();

void __abs_init();
void __acos_init();
void __asin_init();
void __atan_init();
void __binary_scalar_init();
void __cast_init();
void __ceil_init();
void __cos_init();
void __elu_init();
void __eqz_init();
void __erf_init();
void __erfc_init();
void __erfinv_init();
void __exp_init();
void __exp2_init();
void __expm1_init();
void __fill_init();
void __floor_init();
void __gelu_init();
void __gez_init();
void __gtz_init();
void __heaviside_init();
void __i0_init();
void __isfinite_init();
void __isinf_init();
void __isnan_init();
void __isneginf_init();
void __isposinf_init();
void __leaky_relu_init();
void __lez_init();
void __log_init();
void __log_with_base_init();
void __logical_not_init();
void __ltz_init();
void __max_init();
void __nez_init();
void __power_init();
void __recip_init();
void __relu_init();
void __relu_max_init();
void __relu_min_init();
void __rsqrt_init();
void __sigmoid_init();
void __sign_init();
void __signbit_init();
void __sin_init();
void __sqrt_init();
void __square_init();
void __tan_init();
void __tanh_init();

// unpack

template<typename U, typename V>
    void __unpack_binary_init(pipe<U> icb0, pipe<V> icb1);
template<typename U, typename V>
    void __unpack_bcast_rows_init(pipe<U> icb0, pipe<V> icb1);
template<typename U, typename V>
    void __unpack_bcast_cols_init(pipe<U> icb0, pipe<V> icb1);
template<typename U, typename V>
    void __unpack_bcast_scalar_init(pipe<U> icb0, pipe<V> icb1);
template<typename U, typename V>
    void __unpack_matmul_init(pipe<U> icb0, pipe<V> icb1, bool transpose);
template<typename U>
    void __unpack_unary_init(pipe<U> icb);
template<typename U, typename V>
    void __unpack_reduce_rows_init(pipe<U> icb, pipe<V> icb_scaler);
template<typename U, typename V>
    void __unpack_reduce_cols_init(pipe<U> icb, pipe<V> icb_scaler);
template<typename U, typename V>
    void __unpack_reduce_scalar_init(pipe<U> icb, pipe<V> icb_scaler);
template<typename U>
    void __unpack_tilize_block_init(pipe<U> icb, uint32 block);
template<typename U>
    void __unpack_transpose_init(pipe<U> icb);
template<typename U>
    void __unpack_untilize_block_init(pipe<U> icb);

// pack

template<typename U>
    void __pack_init(pipe<U> ocb);
template<typename U>
    void __pack_row_init(pipe<U> ocb);
template<typename U>
    void __pack_col_init(pipe<U> ocb);
template<typename U>
    void __pack_scalar_init(pipe<U> ocb);

)cpp";

} // namespace

const char *get_builtin_init_header() {
    return g_builtin_init_header;
}

} // namespace front
} // namespace tanto
} // namespace ronin

