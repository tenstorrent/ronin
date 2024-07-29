
#include <cstdint>
#include <cassert>

#include "whisper/riscv/riscv32.hpp"

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/compute_api.hpp"
#include "core/machine.hpp"

#include "riscv/builtin_compute_tanto.hpp"
#include "riscv/compute_tanto_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

namespace {

// Tanto extensions: math

void tanto_copy_init(Compute *api, Riscv32Core *core) {
    api->tanto_copy_init();
}

void tanto_add_init(Compute *api, Riscv32Core *core) {
    api->tanto_add_init();
}

void tanto_sub_init(Compute *api, Riscv32Core *core) {
    api->tanto_sub_init();
}

void tanto_mul_init(Compute *api, Riscv32Core *core) {
    api->tanto_mul_init();
}

void tanto_add_bcast_rows_init(Compute *api, Riscv32Core *core) {
    api->tanto_add_bcast_rows_init();
}

void tanto_sub_bcast_rows_init(Compute *api, Riscv32Core *core) {
    api->tanto_sub_bcast_rows_init();
}

void tanto_mul_bcast_rows_init(Compute *api, Riscv32Core *core) {
    api->tanto_mul_bcast_rows_init();
}

void tanto_add_bcast_cols_init(Compute *api, Riscv32Core *core) {
    api->tanto_add_bcast_cols_init();
}

void tanto_sub_bcast_cols_init(Compute *api, Riscv32Core *core) {
    api->tanto_sub_bcast_cols_init();
}

void tanto_mul_bcast_cols_init(Compute *api, Riscv32Core *core) {
    api->tanto_mul_bcast_cols_init();
}

void tanto_add_bcast_scalar_init(Compute *api, Riscv32Core *core) {
    api->tanto_add_bcast_scalar_init();
}

void tanto_sub_bcast_scalar_init(Compute *api, Riscv32Core *core) {
    api->tanto_sub_bcast_scalar_init();
}

void tanto_mul_bcast_scalar_init(Compute *api, Riscv32Core *core) {
    api->tanto_mul_bcast_scalar_init();
}

void tanto_matmul_init(Compute *api, Riscv32Core *core) {
    bool transpose = bool(core->get_arg(0));
    api->tanto_matmul_init(transpose);
}

void tanto_reduce_max_rows_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_max_rows_init();
}

void tanto_reduce_max_cols_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_max_cols_init();
}

void tanto_reduce_max_scalar_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_max_scalar_init();
}

void tanto_reduce_sum_rows_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_sum_rows_init();
}

void tanto_reduce_sum_cols_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_sum_cols_init();
}

void tanto_reduce_sum_scalar_init(Compute *api, Riscv32Core *core) {
    api->tanto_reduce_sum_scalar_init();
}

void tanto_transpose_init(Compute *api, Riscv32Core *core) {
    api->tanto_transpose_init();
}

void tanto_tilize_block_init(Compute *api, Riscv32Core *core) {
    api->tanto_tilize_block_init();
}

void tanto_untilize_block_init(Compute *api, Riscv32Core *core) {
    api->tanto_untilize_block_init();
}

void tanto_abs_init(Compute *api, Riscv32Core *core) {
    api->tanto_abs_init();
}

void tanto_acos_init(Compute *api, Riscv32Core *core) {
    api->tanto_acos_init();
}

void tanto_asin_init(Compute *api, Riscv32Core *core) {
    api->tanto_asin_init();
}

void tanto_atan_init(Compute *api, Riscv32Core *core) {
    api->tanto_atan_init();
}

void tanto_cos_init(Compute *api, Riscv32Core *core) {
    api->tanto_cos_init();
}

void tanto_elu_init(Compute *api, Riscv32Core *core) {
    api->tanto_elu_init();
}

void tanto_eqz_init(Compute *api, Riscv32Core *core) {
    api->tanto_eqz_init();
}

void tanto_erf_init(Compute *api, Riscv32Core *core) {
    api->tanto_erf_init();
}

void tanto_erfc_init(Compute *api, Riscv32Core *core) {
    api->tanto_erfc_init();
}

void tanto_erfinv_init(Compute *api, Riscv32Core *core) {
    api->tanto_erfinv_init();
}

void tanto_exp_init(Compute *api, Riscv32Core *core) {
    api->tanto_exp_init();
}

void tanto_exp2_init(Compute *api, Riscv32Core *core) {
    api->tanto_exp2_init();
}

void tanto_expm1_init(Compute *api, Riscv32Core *core) {
    api->tanto_expm1_init();
}

void tanto_gelu_init(Compute *api, Riscv32Core *core) {
    api->tanto_gelu_init();
}

void tanto_gez_init(Compute *api, Riscv32Core *core) {
    api->tanto_gez_init();
}

void tanto_gtz_init(Compute *api, Riscv32Core *core) {
    api->tanto_gtz_init();
}

void tanto_heaviside_init(Compute *api, Riscv32Core *core) {
    api->tanto_heaviside_init();
}

void tanto_i0_init(Compute *api, Riscv32Core *core) {
    api->tanto_i0_init();
}

void tanto_isfinite_init(Compute *api, Riscv32Core *core) {
    api->tanto_isfinite_init();
}

void tanto_isinf_init(Compute *api, Riscv32Core *core) {
    api->tanto_isinf_init();
}

void tanto_isnan_init(Compute *api, Riscv32Core *core) {
    api->tanto_isnan_init();
}

void tanto_isneginf_init(Compute *api, Riscv32Core *core) {
    api->tanto_isneginf_init();
}

void tanto_isposinf_init(Compute *api, Riscv32Core *core) {
    api->tanto_isposinf_init();
}

void tanto_leaky_relu_init(Compute *api, Riscv32Core *core) {
    api->tanto_leaky_relu_init();
}

void tanto_lez_init(Compute *api, Riscv32Core *core) {
    api->tanto_lez_init();
}

void tanto_log_init(Compute *api, Riscv32Core *core) {
    api->tanto_log_init();
}

void tanto_log_with_base_init(Compute *api, Riscv32Core *core) {
    api->tanto_log_with_base_init();
}

void tanto_logical_not_init(Compute *api, Riscv32Core *core) {
    api->tanto_logical_not_init();
}

void tanto_ltz_init(Compute *api, Riscv32Core *core) {
    api->tanto_ltz_init();
}

void tanto_nez_init(Compute *api, Riscv32Core *core) {
    api->tanto_nez_init();
}

void tanto_power_init(Compute *api, Riscv32Core *core) {
    api->tanto_power_init();
}

void tanto_recip_init(Compute *api, Riscv32Core *core) {
    api->tanto_recip_init();
}

void tanto_relu_init(Compute *api, Riscv32Core *core) {
    api->tanto_relu_init();
}

void tanto_relu_max_init(Compute *api, Riscv32Core *core) {
    api->tanto_relu_max_init();
}

void tanto_relu_min_init(Compute *api, Riscv32Core *core) {
    api->tanto_relu_min_init();
}

void tanto_rsqrt_init(Compute *api, Riscv32Core *core) {
    api->tanto_rsqrt_init();
}

void tanto_sigmoid_init(Compute *api, Riscv32Core *core) {
    api->tanto_sigmoid_init();
}

void tanto_sign_init(Compute *api, Riscv32Core *core) {
    api->tanto_sign_init();
}

void tanto_signbit_init(Compute *api, Riscv32Core *core) {
    api->tanto_signbit_init();
}

void tanto_sin_init(Compute *api, Riscv32Core *core) {
    api->tanto_sin_init();
}

void tanto_sqrt_init(Compute *api, Riscv32Core *core) {
    api->tanto_sqrt_init();
}

void tanto_square_init(Compute *api, Riscv32Core *core) {
    api->tanto_square_init();
}

void tanto_tan_init(Compute *api, Riscv32Core *core) {
    api->tanto_tan_init();
}

void tanto_tanh_init(Compute *api, Riscv32Core *core) {
    api->tanto_tanh_init();
}

// Tanto extensions: unpack

void tanto_unpack_binary_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    api->tanto_unpack_binary_init(icb0, icb1);
}

void tanto_unpack_bcast_rows_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    api->tanto_unpack_bcast_rows_init(icb0, icb1);
}

void tanto_unpack_bcast_cols_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    api->tanto_unpack_bcast_cols_init(icb0, icb1);
}

void tanto_unpack_bcast_scalar_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    api->tanto_unpack_bcast_scalar_init(icb0, icb1);
}

void tanto_unpack_matmul_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    bool transpose = bool(core->get_arg(2));
    api->tanto_unpack_matmul_init(icb0, icb1, transpose);
}

void tanto_unpack_unary_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->tanto_unpack_unary_init(icb);
}

void tanto_unpack_tilize_block_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t block = core->get_arg(1);
    api->tanto_unpack_tilize_block_init(icb, block);
}

void tanto_unpack_transpose_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->tanto_unpack_transpose_init(icb);
}

void tanto_unpack_untilize_block_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->tanto_unpack_untilize_block_init(icb);
}

// Tanto extensions: pack

void tanto_pack_init(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    api->tanto_pack_init(ocb);
}

void tanto_pack_reduce_rows_init(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    api->tanto_pack_reduce_rows_init(ocb);
}

void tanto_pack_reduce_cols_init(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    api->tanto_pack_reduce_cols_init(ocb);
}

void tanto_pack_reduce_scalar_init(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    api->tanto_pack_reduce_scalar_init(ocb);
}

} // namespace

//
//    ComputeTantoHandler
//

ComputeTantoHandler::ComputeTantoHandler(Machine *machine):
        m_machine(machine) { }

ComputeTantoHandler::~ComputeTantoHandler() { }

#define DECL_BUILTIN(name, count) \
    case ComputeTantoBuiltinId::name: \
        name(api, core); \
        break;

void ComputeTantoHandler::call(Riscv32Core *core, int id) {
    Compute *api = m_machine->get_compute_api();
    switch (ComputeTantoBuiltinId(id)) {
COMPUTE_TANTO_BUILTINS
    default:
        assert(false);
        break;
    }
}

#undef DECL_BUILTIN

} // namespace riscv
} // namespace device
} // namespace metal
} // namespace tt

