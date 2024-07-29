#include <cstdint>

#include "core/addr_map.hpp"
#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/compute_api.hpp"

#include "ref/llk.hpp"
#include "ref/compute_impl.hpp"

namespace tt {
namespace metal {
namespace device {
namespace ref {

//
//    ComputeImpl
//

ComputeImpl::ComputeImpl(Memory *l1, CB *cb):
        m_l1(l1),
        m_cb(cb),
        m_llk(l1, cb) { }

ComputeImpl::~ComputeImpl() { }

void ComputeImpl::reset() {
    m_llk.reset();
}

uint32_t ComputeImpl::get_arg_uint32(int arg_idx) {
    uint32_t *arg_base = reinterpret_cast<uint32_t *>(m_l1->map_addr(AddrMap::TRISC_L1_ARG_BASE));
    return arg_base[arg_idx];
}

// reg_api

void ComputeImpl::acquire_dst(DstMode mode) {
    m_llk.acquire_dst();
}

void ComputeImpl::tile_regs_acquire() {
    m_llk.acquire_dst();
}

void ComputeImpl::tile_regs_wait() {
    // nothing to do
}

void ComputeImpl::release_dst(DstMode mode) {
    // nothing to do
}

void ComputeImpl::tile_regs_commit() {
    // nothing to do
}

void ComputeImpl::tile_regs_release() {
    // nothing to do
}

// pack

void ComputeImpl::pack_relu_config(uint32_t config) {
    m_llk.pack_relu_config(config);
}

void ComputeImpl::pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t output_tile_index,
        bool out_of_order_output) {
    m_llk.pack(ifrom_dst, icb); 
}

void ComputeImpl::matmul_pack_tile(
        uint32_t ifrom_dst, 
        uint32_t icb, 
        uint32_t ntiles) {
    m_llk.matmul_pack(ifrom_dst, icb, ntiles);
}

void ComputeImpl::pack_reconfig_data_format(uint32_t new_operand) {
    // not supported
}

// unpack

void ComputeImpl::unpack_reconfig_data_format(
        uint32_t srca_new_operand, 
        uint32_t srcb_new_operand) {
    // not supported
}

// cb_api

void ComputeImpl::cb_wait_front(uint32_t cbid, uint32_t ntiles) {
    m_cb->cb_wait_front(cbid, ntiles);
}

void ComputeImpl::cb_pop_front(uint32_t cbid, uint32_t ntiles) {
    m_cb->cb_pop_front(cbid, ntiles);
}

void ComputeImpl::cb_reserve_back(uint32_t cbid, uint32_t ntiles) {
    m_cb->cb_reserve_back(cbid, ntiles);
}

void ComputeImpl::cb_push_back(uint32_t cbid, uint32_t ntiles) {
    m_cb->cb_push_back(cbid, ntiles);
}

// bcast

void ComputeImpl::sub_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::COL, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWSUB, BroadcastType::COL, idst);
}

void ComputeImpl::mul_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::COL, icb0, icb1, itile0, itile1); 
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWMUL, BroadcastType::COL, idst);
}

void ComputeImpl::mul_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::ROW, icb0, icb1, itile0, itile1); 
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWMUL, BroadcastType::ROW, idst);
}

void ComputeImpl::add_tiles_bcast_rows(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::ROW, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWADD, BroadcastType::ROW, idst);
}

void ComputeImpl::add_tiles_bcast_cols(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::COL, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWADD, BroadcastType::COL, idst);
}

void ComputeImpl::init_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::any_tiles_bcast(
        EltwiseBinaryType tBcastOp, 
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(tBcastDim, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(tBcastOp, tBcastDim, idst);
}

void ComputeImpl::add_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    any_tiles_bcast(EltwiseBinaryType::ELWADD, tBcastDim, icb0, icb1, itile0, itile1, idst);
}

void ComputeImpl::sub_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    any_tiles_bcast(EltwiseBinaryType::ELWSUB, tBcastDim, icb0, icb1, itile0, itile1, idst);
}

void ComputeImpl::mul_tiles_bcast(
        BroadcastType tBcastDim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    any_tiles_bcast(EltwiseBinaryType::ELWMUL, tBcastDim, icb0, icb1, itile0, itile1, idst);
}

void ComputeImpl::add_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::add_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::mul_tiles_bcast_scalar_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::mul_tiles_bcast_scalar(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::SCALAR, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(ELWMUL, BroadcastType::SCALAR, idst);
}

void ComputeImpl::mul_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::mul_bcast_rows_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::sub_bcast_cols_init_short(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

// eltwise_binary

void ComputeImpl::binary_op_init_common(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::mul_tiles_init_f() {
    // nothing to do
}

void ComputeImpl::mul_tiles_init(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::add_tiles_init_nof() {
    // nothing to do
}

void ComputeImpl::add_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) {
    // nothing to do
}

void ComputeImpl::sub_tiles_init_nof() {
    // nothing to do
}

void ComputeImpl::sub_tiles_init(uint32_t icb0, uint32_t icb1, bool acc_to_dest) {
    // nothing to do
}

void ComputeImpl::mul_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::NONE, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWMUL, BroadcastType::NONE, idst);
}

void ComputeImpl::add_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::NONE, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWADD, BroadcastType::NONE, idst); 
}

void ComputeImpl::sub_tiles(
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::NONE, icb0, icb1, itile0, itile1);
    m_llk.math_eltwise_binary(EltwiseBinaryType::ELWSUB, BroadcastType::NONE, idst); 
}

void ComputeImpl::binary_op_specific_init(
        bool full_init, 
        EltwiseBinaryType eltwise_binary_op_type) {
    // nothing to do
}

// eltwise_unary

void ComputeImpl::unary_op_init_common(uint32_t icb) {
    // nothing to do
}

void ComputeImpl::init_sfpu(uint32_t icb) {
    // nothing to do
}

// matmul

void ComputeImpl::mm_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose) {
    // nothing to do
}

#if 0 // TODO: Revise this
void ComputeImpl::mm_init_once() {
    // nothing to do
}
#endif

void ComputeImpl::matmul_tiles(
        uint32_t c_in0, 
        uint32_t c_in1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst, 
        uint32_t transpose) {
    m_llk.unpack_AB_matmul(c_in0, c_in1, itile0, itile1);
    m_llk.math_matmul(idst, bool(transpose)); 
}

void ComputeImpl::mm_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose) {
    // nothing to do
}

void ComputeImpl::mm_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t c_in_old_srca, 
        uint32_t transpose) {
    // nothing to do
}

void ComputeImpl::mm_block_init(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t out_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) {
    // nothing to do
}

void ComputeImpl::matmul_block(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t in0_tile_index, 
        uint32_t in1_tile_index, 
        uint32_t idst, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) {
    for (uint32_t rt = 0; rt < rt_dim; rt++) {
        for (uint32_t ct = 0; ct < ct_dim; ct++) {
            uint32_t cm_itile0 = in0_tile_index + rt * kt_dim;
            uint32_t cm_itile1 = in1_tile_index + ct;
            uint32_t cm_idst = idst + rt * ct_dim + ct;
            m_llk.unpack_AB_matmul(in0_cb_id, in1_cb_id, cm_itile0, cm_itile1);
            m_llk.math_matmul(cm_idst, transpose); 
        }
    }
}

void ComputeImpl::mm_block_init_short(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) {
    // nothing to do
}

void ComputeImpl::mm_block_init_short_with_dt(
        uint32_t in0_cb_id, 
        uint32_t in1_cb_id, 
        uint32_t old_in1_cb_id, 
        uint32_t transpose, 
        uint32_t ct_dim, 
        uint32_t rt_dim, 
        uint32_t kt_dim) {
    // nothing to do
}

// reduce

void ComputeImpl::reduce_init(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::reduce_init_short(
        PoolType reduce_op, 
        ReduceDim reduce_dim,
        uint32_t icb, 
        uint32_t icb_scaler, 
        uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::reduce_init_delta(
        PoolType reduce_type, 
        ReduceDim reduce_dim, 
        bool at_start,
        uint32_t ocb, 
        uint32_t icb0, 
        uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::reduce_revert_delta(ReduceDim reduce_dim, uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::reduce_tile(
        PoolType reduce_type, 
        ReduceDim reduce_dim,
        uint32_t icb0, 
        uint32_t icb1, 
        uint32_t itile0, 
        uint32_t itile1, 
        uint32_t idst) {
    m_llk.unpack_AB(BroadcastType::NONE, icb0, icb1, itile0, itile1);
    m_llk.math_reduce(reduce_type, reduce_dim, idst);

}

// tile_move_copy

void ComputeImpl::copy_tile_to_dst_init_short(uint32_t cbid, uint32_t transpose) {
    // nothing to do
}

void ComputeImpl::copy_tile_init() {
    // nothing to do
}

void ComputeImpl::copy_tile_to_dst_init_short_with_dt(
        uint32_t old_cbid, 
        uint32_t new_cbid, 
        uint32_t transpose) {
    // nothing to do
}

#if 0 // TODO: Revise this
void ComputeImpl::copy_tile_matmul_partials_init_short_with_dt(uint32_t cbid) {
    // nothing to do
}
#endif

void ComputeImpl::copy_tile(
        uint32_t in_cb_id, 
        uint32_t in_tile_index, 
        uint32_t dst_tile_index) {
    m_llk.unpack_A(in_cb_id, in_tile_index, false);
    m_llk.math_eltwise_unary_datacopy(dst_tile_index);
}

void ComputeImpl::copy_block_matmul_partials(
        uint32_t in_cb_id, 
        uint32_t start_in_tile_index, 
        uint32_t start_dst_tile_index, 
        uint32_t ntiles) {
    for (uint32_t i = 0; i < ntiles; i++) {
        m_llk.unpack_A(in_cb_id, start_in_tile_index + i, false);
        m_llk.math_eltwise_unary_datacopy(start_dst_tile_index + i);
    }
}

// tilize

void ComputeImpl::tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::tilize_init_short(uint32_t icb, uint32_t block) {
    // nothing to do
}

void ComputeImpl::tilize_block(uint32_t icb, uint32_t block, uint32_t ocb) {
    m_llk.unpack_tilize(icb, block);
    for (uint32_t idst = 0; idst < block; idst++) {
        m_llk.pack(idst, ocb);
    }
}

void ComputeImpl::tilize_uninit(uint32_t icb) {
    // nothing to do
}

// transpose_wh

void ComputeImpl::transpose_wh_init(uint32_t icb, uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst) {
    m_llk.unpack_A(icb, itile, true);
    m_llk.math_eltwise_unary_datacopy(idst); 
}

// untilize

void ComputeImpl::untilize_init(uint32_t icb, uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::untilize_init_short(uint32_t icb) {
    // nothing to do
}

void ComputeImpl::untilize_block(uint32_t N, uint32_t icb, uint32_t block, uint32_t ocb) {
    m_llk.unpack_untilize(icb, block);
    for (uint32_t idst = 0; idst < block; idst++) {
        m_llk.pack_raw(idst, ocb);
    }
}

void ComputeImpl::untilize_uninit(uint32_t icb) {
    // nothing to do
}

// eltwise_unary_sfpu

void ComputeImpl::rsqrt_tile_init() {
    // nothing to do
}

void ComputeImpl::rsqrt_tile(uint32_t idst, bool fast_and_approx) {
    m_llk.math_eltwise_unary_sfpu_rsqrt(idst, fast_and_approx);
}

void ComputeImpl::sigmoid_tile_init() {
    // nothing to do
}

void ComputeImpl::sigmoid_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_sigmoid(idst);
}

void ComputeImpl::log_tile_init() {
    // nothing to do
}

void ComputeImpl::log_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_log(idst);
}

void ComputeImpl::log_with_base_tile_init() {
    // nothing to do
}

void ComputeImpl::log_with_base_tile(uint32_t idst, uint32_t base_scale) {
    m_llk.math_eltwise_unary_sfpu_log_with_base(idst, base_scale);
}

void ComputeImpl::tanh_tile_init() {
    // nothing to do
}

void ComputeImpl::tanh_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_tanh(idst);
}

void ComputeImpl::signbit_tile_init() {
    // nothing to do
}

void ComputeImpl::signbit_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_signbit(idst);
}

void ComputeImpl::abs_tile_init() {
    // nothing to do
}

void ComputeImpl::abs_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_abs(idst);
}

void ComputeImpl::sign_tile_init() {
    // nothing to do
}

void ComputeImpl::sign_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_sign(idst);
}

void ComputeImpl::square_tile_init() {
    // nothing to do
}

void ComputeImpl::square_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_square(idst);
}

void ComputeImpl::ltz_tile_init() {
    // nothing to do
}

void ComputeImpl::ltz_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_ltz(idst);
}

void ComputeImpl::eqz_tile_init() {
    // nothing to do
}

void ComputeImpl::eqz_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_eqz(idst);
}

void ComputeImpl::lez_tile_init() {
    // nothing to do
}

void ComputeImpl::lez_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_lez(idst);
}

void ComputeImpl::gtz_tile_init() {
    // nothing to do
}

void ComputeImpl::gtz_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_gtz(idst);
}

void ComputeImpl::nez_tile_init() {
    // nothing to do
}

void ComputeImpl::nez_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_nez(idst);
}

void ComputeImpl::gez_tile_init() {
    // nothing to do
}

void ComputeImpl::gez_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_gez(idst);
}

void ComputeImpl::power_tile_init() {
    // nothing to do
}

void ComputeImpl::power_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_power(idst, param0);
}

#if 0 // SKIPPED
void ComputeImpl::graph_interpreter_init() { 
    // not supported
}

void ComputeImpl::get_next_op_info(op_info_t &op_info) {
    // not supported
}
#endif

void ComputeImpl::exp2_tile_init() {
    // nothing to do
}

void ComputeImpl::exp2_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_exp2(idst);
}

void ComputeImpl::heaviside_tile_init() {
    // nothing to do
}

void ComputeImpl::heaviside_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_heaviside(idst, param0);
}

void ComputeImpl::expm1_tile_init() {
    // nothing to do
}

void ComputeImpl::expm1_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_expm1(idst);
}

void ComputeImpl::asin_tile_init() {
    // nothing to do
}

void ComputeImpl::asin_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_asin(idst);
}

void ComputeImpl::atan_tile_init() {
    // nothing to do
}

void ComputeImpl::atan_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_atan(idst);
}

void ComputeImpl::acos_tile_init() {
    // nothing to do
}

void ComputeImpl::acos_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_acos(idst);
}

// eltwise_unary/elu

void ComputeImpl::elu_tile_init() {
    // nothing to do
}

void ComputeImpl::elu_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_elu(idst, param0);
}

// eltwise_unary/erf_erfc

void ComputeImpl::erf_tile_init() { 
    // nothing to do
}

void ComputeImpl::erf_tile(uint32_t idst, bool fast_and_approx) {
    m_llk.math_eltwise_unary_sfpu_erf(idst, fast_and_approx);
}

void ComputeImpl::erfc_tile_init() { 
    // nothing to do
}

void ComputeImpl::erfc_tile(uint32_t idst, bool fast_and_approx) {
    m_llk.math_eltwise_unary_sfpu_erfc(idst, fast_and_approx);
}

// eltwise_unary/erfinv

void ComputeImpl::erfinv_tile_init() {
    // nothing to do
}

void ComputeImpl::erfinv_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_erfinv(idst);
}

// eltwise_unary/exp

void ComputeImpl::exp_tile_init() {
    // nothing to do
}

void ComputeImpl::exp_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_exponential(idst);
}

// eltwise_unary/gelu

void ComputeImpl::gelu_tile_init() {
    // nothing to do
}

void ComputeImpl::gelu_tile(uint32_t idst, bool fast_and_approx) {
    m_llk.math_eltwise_unary_sfpu_gelu(idst, fast_and_approx);
}

// eltwise_unary/i0

void ComputeImpl::i0_tile_init() {
    // nothing to do
}

void ComputeImpl::i0_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_i0(idst);
}

// eltwise_unary/isinf_isnan

void ComputeImpl::isinf_tile_init() {
    // nothing to do
}

void ComputeImpl::isinf_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_isinf(idst);
}

void ComputeImpl::isposinf_tile_init() {
    // nothing to do
}

void ComputeImpl::isposinf_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_isposinf(idst);
}

void ComputeImpl::isneginf_tile_init() {
    // nothing to do
}

void ComputeImpl::isneginf_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_isneginf(idst);
}

void ComputeImpl::isnan_tile_init() {
    // nothing to do
}

void ComputeImpl::isnan_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_isnan(idst);
}

void ComputeImpl::isfinite_tile_init() {
    // nothing to do
}

void ComputeImpl::isfinite_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_isfinite(idst);
}

// eltwise_unary/logical_not_noti

void ComputeImpl::logical_not_unary_tile_init() {
    // nothing to do
}

void ComputeImpl::logical_not_unary_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_logical_not_unary(idst);
}

// eltwise_unary/recip

void ComputeImpl::recip_tile_init() {
    // nothing to do
}

void ComputeImpl::recip_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_reciprocal(idst);
}

// eltwise_unary/relu

void ComputeImpl::relu_max_tile_init() {
    // nothing to do
}

void ComputeImpl::relu_max_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_relu_max(idst, param0);
}

void ComputeImpl::relu_min_tile_init() {
    // nothing to do
}

void ComputeImpl::relu_min_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_relu_min(idst, param0);
}

void ComputeImpl::relu_tile_init() {
    // nothing to do
}

void ComputeImpl::relu_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_relu(idst);
}

void ComputeImpl::leaky_relu_tile_init() {
    // nothing to do
}

void ComputeImpl::leaky_relu_tile(uint32_t idst, uint32_t param0) {
    m_llk.math_eltwise_unary_sfpu_lrelu(idst, param0);
}

// eltwise_unary/sqrt

void ComputeImpl::sqrt_tile_init() {
    // nothing to do
}

void ComputeImpl::sqrt_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_sqrt(idst);
}

// eltwise_unary/trigonometry

void ComputeImpl::sin_tile_init() {
    // nothing to do
}

void ComputeImpl::sin_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_sine(idst);
}

void ComputeImpl::cos_tile_init() {
    // nothing to do
}

void ComputeImpl::cos_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_cosine(idst);
}

void ComputeImpl::tan_tile_init() {
    // nothing to do
}

void ComputeImpl::tan_tile(uint32_t idst) {
    m_llk.math_eltwise_unary_sfpu_tan(idst);
}

// Tanto extensions: math

void ComputeImpl::tanto_copy_init() {
    // nothing to do
}

void ComputeImpl::tanto_add_init() {
    // nothing to do
}

void ComputeImpl::tanto_sub_init() {
    // nothing to do
}

void ComputeImpl::tanto_mul_init() {
    // nothing to do
}

void ComputeImpl::tanto_add_bcast_rows_init() {
    // nothing to do
}

void ComputeImpl::tanto_sub_bcast_rows_init() {
    // nothing to do
}

void ComputeImpl::tanto_mul_bcast_rows_init() {
    // nothing to do
}

void ComputeImpl::tanto_add_bcast_cols_init() {
    // nothing to do
}

void ComputeImpl::tanto_sub_bcast_cols_init() {
    // nothing to do
}

void ComputeImpl::tanto_mul_bcast_cols_init() {
    // nothing to do
}

void ComputeImpl::tanto_add_bcast_scalar_init() {
    // nothing to do
}

void ComputeImpl::tanto_sub_bcast_scalar_init() {
    // nothing to do
}

void ComputeImpl::tanto_mul_bcast_scalar_init() {
    // nothing to do
}

void ComputeImpl::tanto_matmul_init(bool transpose) {
    // nothing to do
}

void ComputeImpl::tanto_reduce_max_rows_init() {
    // nothing to do
}

void ComputeImpl::tanto_reduce_max_cols_init() {
    // nothing to do
}

void ComputeImpl::tanto_reduce_max_scalar_init() {
    // nothing to do
}

void ComputeImpl::tanto_reduce_sum_rows_init() {
    // nothing to do
}

void ComputeImpl::tanto_reduce_sum_cols_init() {
    // nothing to do
}

void ComputeImpl::tanto_reduce_sum_scalar_init() {
    // nothing to do
}

void ComputeImpl::tanto_transpose_init() {
    // nothing to do
}

void ComputeImpl::tanto_tilize_block_init() {
    // nothing to do
}

void ComputeImpl::tanto_untilize_block_init() {
    // nothing to do
}

void ComputeImpl::tanto_abs_init() {
    // nothing to do
}

void ComputeImpl::tanto_acos_init() {
    // nothing to do
}

void ComputeImpl::tanto_asin_init() {
    // nothing to do
}

void ComputeImpl::tanto_atan_init() {
    // nothing to do
}

void ComputeImpl::tanto_cos_init() {
    // nothing to do
}

void ComputeImpl::tanto_elu_init() {
    // nothing to do
}

void ComputeImpl::tanto_eqz_init() {
    // nothing to do
}

void ComputeImpl::tanto_erf_init() {
    // nothing to do
}

void ComputeImpl::tanto_erfc_init() {
    // nothing to do
}

void ComputeImpl::tanto_erfinv_init() {
    // nothing to do
}

void ComputeImpl::tanto_exp_init() {
    // nothing to do
}

void ComputeImpl::tanto_exp2_init() {
    // nothing to do
}

void ComputeImpl::tanto_expm1_init() {
    // nothing to do
}

void ComputeImpl::tanto_gelu_init() {
    // nothing to do
}

void ComputeImpl::tanto_gez_init() {
    // nothing to do
}

void ComputeImpl::tanto_gtz_init() {
    // nothing to do
}

void ComputeImpl::tanto_heaviside_init() {
    // nothing to do
}

void ComputeImpl::tanto_i0_init() {
    // nothing to do
}

void ComputeImpl::tanto_isfinite_init() {
    // nothing to do
}

void ComputeImpl::tanto_isinf_init() {
    // nothing to do
}

void ComputeImpl::tanto_isnan_init() {
    // nothing to do
}

void ComputeImpl::tanto_isneginf_init() {
    // nothing to do
}

void ComputeImpl::tanto_isposinf_init() {
    // nothing to do
}

void ComputeImpl::tanto_leaky_relu_init() {
    // nothing to do
}

void ComputeImpl::tanto_lez_init() {
    // nothing to do
}

void ComputeImpl::tanto_log_init() {
    // nothing to do
}

void ComputeImpl::tanto_log_with_base_init() {
    // nothing to do
}

void ComputeImpl::tanto_logical_not_init() {
    // nothing to do
}

void ComputeImpl::tanto_ltz_init() {
    // nothing to do
}

void ComputeImpl::tanto_nez_init() {
    // nothing to do
}

void ComputeImpl::tanto_power_init() {
    // nothing to do
}

void ComputeImpl::tanto_recip_init() {
    // nothing to do
}

void ComputeImpl::tanto_relu_init() {
    // nothing to do
}

void ComputeImpl::tanto_relu_max_init() {
    // nothing to do
}

void ComputeImpl::tanto_relu_min_init() {
    // nothing to do
}

void ComputeImpl::tanto_rsqrt_init() {
    // nothing to do
}

void ComputeImpl::tanto_sigmoid_init() {
    // nothing to do
}

void ComputeImpl::tanto_sign_init() {
    // nothing to do
}

void ComputeImpl::tanto_signbit_init() {
    // nothing to do
}

void ComputeImpl::tanto_sin_init() {
    // nothing to do
}

void ComputeImpl::tanto_sqrt_init() {
    // nothing to do
}

void ComputeImpl::tanto_square_init() {
    // nothing to do
}

void ComputeImpl::tanto_tan_init() {
    // nothing to do
}

void ComputeImpl::tanto_tanh_init() {
    // nothing to do
}

// Tanto extensions: unpack

void ComputeImpl::tanto_unpack_binary_init(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_bcast_rows_init(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_bcast_cols_init(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_bcast_scalar_init(uint32_t icb0, uint32_t icb1) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_matmul_init(uint32_t icb0, uint32_t icb1, bool transpose) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_unary_init(uint32_t icb) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_tilize_block_init(uint32_t icb, uint32_t block) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_transpose_init(uint32_t icb) {
    // nothing to do
}

void ComputeImpl::tanto_unpack_untilize_block_init(uint32_t icb) {
    // nothing to do
}

// Tanto extensions: pack

void ComputeImpl::tanto_pack_init(uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::tanto_pack_reduce_rows_init(uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::tanto_pack_reduce_cols_init(uint32_t ocb) {
    // nothing to do
}

void ComputeImpl::tanto_pack_reduce_scalar_init(uint32_t ocb) {
    // nothing to do
}

} // namespace ref
} // namespace device
} // namespace metal
} // namespace tt

