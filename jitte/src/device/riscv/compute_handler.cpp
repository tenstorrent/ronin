// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <cassert>

#include "whisper/riscv/riscv32.hpp"

#include "core/kernel_structs.hpp"
#include "core/llk_defs.hpp"
#include "core/compute_api.hpp"
#include "core/machine.hpp"

#include "riscv/builtin_compute.hpp"
#include "riscv/compute_handler.hpp"

namespace tt {
namespace metal {
namespace device {
namespace riscv {

using ::riscv::core::Riscv32Core;

namespace {

void get_arg_uint32(Compute *api, Riscv32Core *core) {
    int arg_idx = int(core->get_arg(0));
    uint32_t ret = api->get_arg_uint32(arg_idx);
    core->set_ret(0, ret);
}

// reg_api

void acquire_dst(Compute *api, Riscv32Core *core) {
    DstMode mode = DstMode(core->get_arg(0));
    api->acquire_dst(mode);
}

void tile_regs_acquire(Compute *api, Riscv32Core *core) {
    api->tile_regs_acquire();
}

void tile_regs_wait(Compute *api, Riscv32Core *core) {
    api->tile_regs_wait();
}

void release_dst(Compute *api, Riscv32Core *core) {
    DstMode mode = DstMode(core->get_arg(0));
    api->release_dst(mode);
}

void tile_regs_commit(Compute *api, Riscv32Core *core) {
    api->tile_regs_commit();
}

void tile_regs_release(Compute *api, Riscv32Core *core) {
    api->tile_regs_release();
}

// pack

void pack_relu_config(Compute *api, Riscv32Core *core) {
    uint32_t config = core->get_arg(0);
    api->pack_relu_config(config);
}

void pack_tile(Compute *api, Riscv32Core *core) {
    uint32_t ifrom_dst = core->get_arg(0);
    uint32_t icb = core->get_arg(1);
    uint32_t output_tile_index = core->get_arg(2);
    bool out_of_order_output = bool(core->get_arg(3));
    api->pack_tile(ifrom_dst, icb, output_tile_index, out_of_order_output);
}

void matmul_pack_tile(Compute *api, Riscv32Core *core) {
    uint32_t ifrom_dst = core->get_arg(0);
    uint32_t icb = core->get_arg(1);
    uint32_t ntiles = core->get_arg(2);
    api->matmul_pack_tile(ifrom_dst, icb, ntiles);
}

void pack_reconfig_data_format(Compute *api, Riscv32Core *core) {
    uint32_t new_operand = core->get_arg(0);
    api->pack_reconfig_data_format(new_operand);
}

// unpack

void unpack_reconfig_data_format(Compute *api, Riscv32Core *core) {
    uint32_t srca_new_operand = core->get_arg(0); 
    uint32_t srcb_new_operand = core->get_arg(1);
    api->unpack_reconfig_data_format(
        srca_new_operand, 
        srcb_new_operand);
}

// cb_api

void cb_wait_front(Compute *api, Riscv32Core *core) {
    uint32_t cbid = core->get_arg(0);
    uint32_t ntiles = core->get_arg(1);
    api->cb_wait_front(cbid, ntiles);
}

void cb_pop_front(Compute *api, Riscv32Core *core) {
    uint32_t cbid = core->get_arg(0);
    uint32_t ntiles = core->get_arg(1);
    api->cb_pop_front(cbid, ntiles);
}

void cb_reserve_back(Compute *api, Riscv32Core *core) {
    uint32_t cbid = core->get_arg(0);
    uint32_t ntiles = core->get_arg(1);
    api->cb_reserve_back(cbid, ntiles);
}

void cb_push_back(Compute *api, Riscv32Core *core) {
    uint32_t cbid = core->get_arg(0);
    uint32_t ntiles = core->get_arg(1);
    api->cb_push_back(cbid, ntiles);
}

void get_write_ptr(Compute *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ret = api->get_write_ptr(operand);
    core->set_ret(0, ret);
}

void get_read_ptr(Compute *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ret = api->get_read_ptr(operand);
    core->set_ret(0, ret);
}

void set_write_ptr(Compute *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ptr = core->get_arg(1);
    api->set_write_ptr(operand, ptr);
}

void set_read_ptr(Compute *api, Riscv32Core *core) {
    uint32_t operand = core->get_arg(0);
    uint32_t ptr = core->get_arg(1);
    api->set_read_ptr(operand, ptr);
}

// bcast

void sub_tiles_bcast_cols(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1); 
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->sub_tiles_bcast_cols(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void mul_tiles_bcast_cols(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->mul_tiles_bcast_cols(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void mul_tiles_bcast_rows(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->mul_tiles_bcast_rows(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void add_tiles_bcast_rows(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->add_tiles_bcast_rows(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void add_tiles_bcast_cols(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->add_tiles_bcast_cols(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void init_bcast(Compute *api, Riscv32Core *core) {
    EltwiseBinaryType tBcastOp = EltwiseBinaryType(core->get_arg(0));
    BroadcastType tBcastDim = BroadcastType(core->get_arg(1));
    uint32_t icb0 = core->get_arg(2);
    uint32_t icb1 = core->get_arg(3);
    uint32_t ocb = core->get_arg(4);
    api->init_bcast(
        tBcastOp, 
        tBcastDim,
        icb0, 
        icb1, 
        ocb);
}

void any_tiles_bcast(Compute *api, Riscv32Core *core) {
    EltwiseBinaryType tBcastOp = EltwiseBinaryType(core->get_arg(0)); 
    BroadcastType tBcastDim = BroadcastType(core->get_arg(1));
    uint32_t icb0 = core->get_arg(2);
    uint32_t icb1 = core->get_arg(3);
    uint32_t itile0 = core->get_arg(4);
    uint32_t itile1 = core->get_arg(5);
    uint32_t idst = core->get_arg(6);
    api->any_tiles_bcast(
        tBcastOp, 
        tBcastDim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void add_tiles_bcast(Compute *api, Riscv32Core *core) {
    BroadcastType tBcastDim = BroadcastType(core->get_arg(0));
    uint32_t icb0 = core->get_arg(1); 
    uint32_t icb1 = core->get_arg(2);
    uint32_t itile0 = core->get_arg(3);
    uint32_t itile1 = core->get_arg(4);
    uint32_t idst = core->get_arg(5);
    api->add_tiles_bcast(
        tBcastDim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void sub_tiles_bcast(Compute *api, Riscv32Core *core) {
    BroadcastType tBcastDim = BroadcastType(core->get_arg(0));
    uint32_t icb0 = core->get_arg(1); 
    uint32_t icb1 = core->get_arg(2);
    uint32_t itile0 = core->get_arg(3);
    uint32_t itile1 = core->get_arg(4);
    uint32_t idst = core->get_arg(5);
    api->sub_tiles_bcast(
        tBcastDim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void mul_tiles_bcast(Compute *api, Riscv32Core *core) {
    BroadcastType tBcastDim = BroadcastType(core->get_arg(0));
    uint32_t icb0 = core->get_arg(1); 
    uint32_t icb1 = core->get_arg(2);
    uint32_t itile0 = core->get_arg(3);
    uint32_t itile1 = core->get_arg(4);
    uint32_t idst = core->get_arg(5);
    api->mul_tiles_bcast(
        tBcastDim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void add_bcast_rows_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->add_bcast_rows_init_short(icb0, icb1);
}

void add_bcast_cols_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->add_bcast_cols_init_short(icb0, icb1);
}

void mul_tiles_bcast_scalar_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->mul_tiles_bcast_scalar_init_short(icb0, icb1);
}

void mul_tiles_bcast_scalar(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->mul_tiles_bcast_scalar(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void mul_bcast_cols_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->mul_bcast_cols_init_short(icb0, icb1);
}

void mul_bcast_rows_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->mul_bcast_rows_init_short(icb0, icb1);
}

void sub_bcast_cols_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    api->sub_bcast_cols_init_short(icb0, icb1);
}

// eltwise_binary

void binary_op_init_common(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    uint32_t ocb = core->get_arg(2);
    api->binary_op_init_common(icb0, icb1, ocb);
}

void mul_tiles_init_f(Compute *api, Riscv32Core *core) {
    api->mul_tiles_init_f();
}

void mul_tiles_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    api->mul_tiles_init(icb0, icb1);
}

void add_tiles_init_nof(Compute *api, Riscv32Core *core) {
    api->add_tiles_init_nof();
}

void add_tiles_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    bool acc_to_dest = bool(core->get_arg(2));
    api->add_tiles_init(icb0, icb1, acc_to_dest);
}

void sub_tiles_init_nof(Compute *api, Riscv32Core *core) {
    api->sub_tiles_init_nof();
}

void sub_tiles_init(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    bool acc_to_dest = bool(core->get_arg(2));
    api->sub_tiles_init(icb0, icb1, acc_to_dest);
}

void mul_tiles(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0);
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->mul_tiles(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void add_tiles(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->add_tiles(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void sub_tiles(Compute *api, Riscv32Core *core) {
    uint32_t icb0 = core->get_arg(0); 
    uint32_t icb1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    api->sub_tiles(
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

void binary_op_specific_init(Compute *api, Riscv32Core *core) {
    bool full_init = bool(core->get_arg(0));
    EltwiseBinaryType eltwise_binary_op_type = EltwiseBinaryType(core->get_arg(1));
    api->binary_op_specific_init(full_init, eltwise_binary_op_type);
}

// eltwise_unary

void unary_op_init_common(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->unary_op_init_common(icb);
}

void init_sfpu(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->init_sfpu(icb);
}

// matmul

void mm_init(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0);
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t out_cb_id = core->get_arg(2);
    uint32_t transpose = core->get_arg(3);
    api->mm_init(
        in0_cb_id, 
        in1_cb_id, 
        out_cb_id, 
        transpose);
}

void matmul_tiles(Compute *api, Riscv32Core *core) {
    uint32_t c_in0 = core->get_arg(0);
    uint32_t c_in1 = core->get_arg(1);
    uint32_t itile0 = core->get_arg(2);
    uint32_t itile1 = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    uint32_t transpose = core->get_arg(5);
    api->matmul_tiles(
        c_in0, 
        c_in1, 
        itile0, 
        itile1, 
        idst, 
        transpose);
}

void mm_init_short(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0);
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t transpose = core->get_arg(2);
    api->mm_init_short(in0_cb_id, in1_cb_id, transpose);
}

void mm_init_short_with_dt(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0);
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t c_in_old_srca = core->get_arg(2);
    uint32_t transpose = core->get_arg(3);
    api->mm_init_short_with_dt(in0_cb_id, in1_cb_id, c_in_old_srca, transpose);
}

void mm_block_init(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0); 
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t out_cb_id = core->get_arg(2);
    uint32_t transpose = core->get_arg(3);
    uint32_t ct_dim = core->get_arg(4);
    uint32_t rt_dim = core->get_arg(5);
    uint32_t kt_dim = core->get_arg(6);
    api->mm_block_init(
        in0_cb_id, 
        in1_cb_id, 
        out_cb_id, 
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
}

void matmul_block(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0);
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t in0_tile_index = core->get_arg(2);
    uint32_t in1_tile_index = core->get_arg(3);
    uint32_t idst = core->get_arg(4);
    int transpose = core->get_arg(5);
    uint32_t ct_dim = core->get_arg(6);
    uint32_t rt_dim = core->get_arg(7);
    uint32_t kt_dim = core->get_arg(8);
    api->matmul_block(
        in0_cb_id, 
        in1_cb_id, 
        in0_tile_index, 
        in1_tile_index, 
        idst, 
        transpose, 
        ct_dim, 
        rt_dim, 
        kt_dim);
}

void mm_block_init_short(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0); 
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t transpose = core->get_arg(2);
    uint32_t ct_dim = core->get_arg(3);
    uint32_t rt_dim = core->get_arg(4);
    uint32_t kt_dim = core->get_arg(5);
    api->mm_block_init_short(
        in0_cb_id, 
        in1_cb_id, 
        transpose, 
        ct_dim, 
        rt_dim, 
        kt_dim);
}

void mm_block_init_short_with_dt(Compute *api, Riscv32Core *core) {
    uint32_t in0_cb_id = core->get_arg(0);
    uint32_t in1_cb_id = core->get_arg(1);
    uint32_t old_in1_cb_id = core->get_arg(2);
    uint32_t transpose = core->get_arg(3);
    uint32_t ct_dim = core->get_arg(4);
    uint32_t rt_dim = core->get_arg(5);
    uint32_t kt_dim = core->get_arg(6);
    api->mm_block_init_short_with_dt(
        in0_cb_id, 
        in1_cb_id, 
        old_in1_cb_id, 
        transpose, 
        ct_dim, 
        rt_dim, 
        kt_dim);
}

// reduce

void reduce_init(Compute *api, Riscv32Core *core) {
    PoolType reduce_type = PoolType(core->get_arg(0));
    ReduceDim reduce_dim = ReduceDim(core->get_arg(1));
    bool at_start = bool(core->get_arg(2));
    uint32_t icb = core->get_arg(3);
    uint32_t icb_scaler = core->get_arg(4);
    uint32_t ocb = core->get_arg(5);
    api->reduce_init(
        reduce_type, 
        reduce_dim, 
        at_start,
        icb, 
        icb_scaler, 
        ocb);
}

void reduce_init_short(Compute *api, Riscv32Core *core) {
    PoolType reduce_type = PoolType(core->get_arg(0));
    ReduceDim reduce_dim = ReduceDim(core->get_arg(1));
    uint32_t icb = core->get_arg(2); 
    uint32_t icb_scaler = core->get_arg(3);
    uint32_t ocb = core->get_arg(4);
    api->reduce_init_short(
        reduce_type, 
        reduce_dim,
        icb, 
        icb_scaler, 
        ocb);
}

void reduce_init_delta(Compute *api, Riscv32Core *core) {
    PoolType reduce_type = PoolType(core->get_arg(0)); 
    ReduceDim reduce_dim = ReduceDim(core->get_arg(1));
    bool at_start = bool(core->get_arg(2));
    uint32_t ocb = core->get_arg(3);
    uint32_t icb0 = core->get_arg(4);
    uint32_t icb1 = core->get_arg(5);
    api->reduce_init_delta(
        reduce_type, 
        reduce_dim, 
        at_start,
        ocb, 
        icb0,
        icb1);
}

void reduce_revert_delta(Compute *api, Riscv32Core *core) {
    ReduceDim reduce_dim = ReduceDim(core->get_arg(0));
    uint32_t ocb = core->get_arg(1);
    api->reduce_revert_delta(reduce_dim, ocb);
}

void reduce_tile(Compute *api, Riscv32Core *core) {
    PoolType reduce_type = PoolType(core->get_arg(0));
    ReduceDim reduce_dim = ReduceDim(core->get_arg(1));
    uint32_t icb0 = core->get_arg(2); 
    uint32_t icb1 = core->get_arg(3); 
    uint32_t itile0 = core->get_arg(4); 
    uint32_t itile1 = core->get_arg(5); 
    uint32_t idst = core->get_arg(6); 
    api->reduce_tile(
        reduce_type, 
        reduce_dim,
        icb0, 
        icb1, 
        itile0, 
        itile1, 
        idst);
}

// tile_move_copy

void copy_tile_to_dst_init_short(Compute *api, Riscv32Core *core) {
    uint32_t cbid = core->get_arg(0);
    uint32_t transpose = core->get_arg(1);
    api->copy_tile_to_dst_init_short(cbid, transpose);
}

void copy_tile_init(Compute *api, Riscv32Core *core) {
    api->copy_tile_init();
}

void copy_tile_to_dst_init_short_with_dt(Compute *api, Riscv32Core *core) {
    uint32_t old_cbid = core->get_arg(0);
    uint32_t new_cbid = core->get_arg(1);
    uint32_t transpose = core->get_arg(2);
    api->copy_tile_to_dst_init_short_with_dt(old_cbid, new_cbid, transpose);
}

void copy_tile(Compute *api, Riscv32Core *core) {
    uint32_t in_cb_id = core->get_arg(0);
    uint32_t in_tile_index = core->get_arg(1);
    uint32_t dst_tile_index = core->get_arg(2);
    api->copy_tile(in_cb_id, in_tile_index, dst_tile_index);
}

void copy_block_matmul_partials(Compute *api, Riscv32Core *core) {
    uint32_t in_cb_id = core->get_arg(0); 
    uint32_t start_in_tile_index = core->get_arg(1);
    uint32_t start_dst_tile_index = core->get_arg(2);
    uint32_t ntiles = core->get_arg(3);
    api->copy_block_matmul_partials(
        in_cb_id, 
        start_in_tile_index, 
        start_dst_tile_index, 
        ntiles);
}

// tilize

void tilize_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t block = core->get_arg(1);
    uint32_t ocb = core->get_arg(2);
    api->tilize_init(icb, block, ocb);
}

void tilize_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t block = core->get_arg(1);
    api->tilize_init_short(icb, block);
}

void tilize_block(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t block = core->get_arg(1);
    uint32_t ocb = core->get_arg(2);
    api->tilize_block(icb, block, ocb);
}

void tilize_uninit(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->tilize_uninit(icb);
}

// transpose_wh

void transpose_wh_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t ocb = core->get_arg(1);
    api->transpose_wh_init(icb, ocb);
}

void transpose_wh_tile(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t itile = core->get_arg(1);
    uint32_t idst = core->get_arg(2);
    api->transpose_wh_tile(icb, itile, idst);
}

// untilize

void untilize_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t ocb = core->get_arg(1);
    api->untilize_init(icb, ocb);
}

void untilize_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->untilize_init_short(icb);
}

void untilize_block(Compute *api, Riscv32Core *core) {
    uint32_t N = core->get_arg(0);
    uint32_t icb = core->get_arg(1);
    uint32_t block = core->get_arg(2);
    uint32_t ocb = core->get_arg(3);
    api->untilize_block(N, icb, block, ocb);
}

void untilize_uninit(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    api->untilize_uninit(icb);
}

// pack_untilize

void pack_untilize_init(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t ocb = core->get_arg(1);
    uint32_t block_ct_dim = core->get_arg(2);
    api->pack_untilize_init(icb, ocb, block_ct_dim);
}

void pack_untilize_block(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t block_rt_dim = core->get_arg(1);
    uint32_t ocb = core->get_arg(2);
    uint32_t block_ct_dim = core->get_arg(3);
    api->pack_untilize_block(
        icb, 
        block_rt_dim, 
        ocb,
        block_ct_dim);
}

void pack_untilize_uninit(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    api->pack_untilize_uninit(ocb);
}

void pack_untilize_dst_init_short(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    uint32_t face_r_dim = core->get_arg(1);
    uint32_t num_faces = core->get_arg(2);
    uint32_t block_ct_dim = core->get_arg(3);
    uint32_t full_ct_dim = core->get_arg(4);
    bool diagonal = bool(core->get_arg(5));
    api->pack_untilize_dst_init_short(
        ocb, 
        face_r_dim, 
        num_faces,
        block_ct_dim, 
        full_ct_dim, 
        diagonal);
}

void pack_untilize_dst(Compute *api, Riscv32Core *core) {
    uint32_t ocb = core->get_arg(0);
    uint32_t block_rt_dim = core->get_arg(1);
    uint32_t block_c_index = core->get_arg(2);
    uint32_t face_r_dim = core->get_arg(3);
    uint32_t num_faces = core->get_arg(4);
    uint32_t block_ct_dim = core->get_arg(5);
    uint32_t full_ct_dim = core->get_arg(6);
    bool diagonal = bool(core->get_arg(7));
    api->pack_untilize_dst(
        ocb, 
        block_rt_dim, 
        block_c_index, 
        face_r_dim, 
        num_faces,
        block_ct_dim, 
        full_ct_dim, 
        diagonal);
}

void pack_untilize_init_short(Compute *api, Riscv32Core *core) {
    uint32_t icb = core->get_arg(0);
    uint32_t ocb = core->get_arg(1);
    uint32_t block_ct_dim = core->get_arg(2);
    api->pack_untilize_init_short(icb, ocb, block_ct_dim);
}

// eltwise_unary_sfpu

void rsqrt_tile_init(Compute *api, Riscv32Core *core) {
    api->rsqrt_tile_init();
}

void rsqrt_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    bool fast_and_approx = bool(core->get_arg(1));
    api->rsqrt_tile(idst, fast_and_approx);
}

void sigmoid_tile_init(Compute *api, Riscv32Core *core) {
    api->sigmoid_tile_init();
}

void sigmoid_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->sigmoid_tile(idst);
}

void log_tile_init(Compute *api, Riscv32Core *core) {
    api->log_tile_init();
}

void log_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->log_tile(idst);
}

void log_with_base_tile_init(Compute *api, Riscv32Core *core) {
    api->log_with_base_tile_init();
}

void log_with_base_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t base_scale = core->get_arg(1);
    api->log_with_base_tile(idst, base_scale);
}

void tanh_tile_init(Compute *api, Riscv32Core *core) {
    api->tanh_tile_init();
}

void tanh_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->tanh_tile(idst);
}

void signbit_tile_init(Compute *api, Riscv32Core *core) {
    api->signbit_tile_init();
}

void signbit_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->signbit_tile(idst);
}

void abs_tile_init(Compute *api, Riscv32Core *core) {
    api->abs_tile_init();
}

void abs_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->abs_tile(idst);
}

void sign_tile_init(Compute *api, Riscv32Core *core) {
    api->sign_tile_init();
}

void sign_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->sign_tile(idst);
}

void square_tile_init(Compute *api, Riscv32Core *core) {
    api->square_tile_init();
}

void square_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->square_tile(idst);
}

void ltz_tile_init(Compute *api, Riscv32Core *core) {
    api->ltz_tile_init();
}

void ltz_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->ltz_tile(idst);
}

void eqz_tile_init(Compute *api, Riscv32Core *core) {
    api->eqz_tile_init();
}

void eqz_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->eqz_tile(idst);
}

void lez_tile_init(Compute *api, Riscv32Core *core) {
    api->lez_tile_init();
}

void lez_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->lez_tile(idst);
}

void gtz_tile_init(Compute *api, Riscv32Core *core) {
    api->gtz_tile_init();
}

void gtz_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->gtz_tile(idst);
}

void nez_tile_init(Compute *api, Riscv32Core *core) {
    api->nez_tile_init();
}

void nez_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->nez_tile(idst);
}

void gez_tile_init(Compute *api, Riscv32Core *core) {
    api->gez_tile_init();
}

void gez_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->gez_tile(idst);
}

void power_tile_init(Compute *api, Riscv32Core *core) {
    api->power_tile_init();
}

void power_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->power_tile(idst, param0);
}

void max_tile_init(Compute *api, Riscv32Core *core) {
    api->max_tile_init();
}

void max_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst0 = core->get_arg(0);
    uint32_t idst1 = core->get_arg(1);
    api->max_tile(idst0, idst1);
}

void exp2_tile_init(Compute *api, Riscv32Core *core) {
    api->exp2_tile_init();
}

void exp2_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->exp2_tile(idst);
}

void heaviside_tile_init(Compute *api, Riscv32Core *core) {
    api->heaviside_tile_init();
}

void heaviside_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->heaviside_tile(idst, param0);
}

void expm1_tile_init(Compute *api, Riscv32Core *core) {
    api->expm1_tile_init();
}

void expm1_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->expm1_tile(idst);
}

void asin_tile_init(Compute *api, Riscv32Core *core) {
    api->asin_tile_init();
}

void asin_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->asin_tile(idst);
}

void atan_tile_init(Compute *api, Riscv32Core *core) {
    api->atan_tile_init();
}

void atan_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->atan_tile(idst);
}

void acos_tile_init(Compute *api, Riscv32Core *core) {
    api->acos_tile_init();
}

void acos_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->acos_tile(idst);
}

// eltwise_unary/elu

void elu_tile_init(Compute *api, Riscv32Core *core) {
    api->elu_tile_init();
}

void elu_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->elu_tile(idst, param0);
}

// eltwise_unary/binop_with_scalar

void binop_with_scalar_tile_init(Compute *api, Riscv32Core *core) {
    api->binop_with_scalar_tile_init();
}

void add_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->add_unary_tile(idst, param0);
}

void sub_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->sub_unary_tile(idst, param0);
}

void mul_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->mul_unary_tile(idst, param0);
}

void div_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->div_unary_tile(idst, param0);
}

void rsub_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->rsub_unary_tile(idst, param0);
}

// eltwise_unary/erf_erfc

void erf_tile_init(Compute *api, Riscv32Core *core) {
    api->erf_tile_init();
}

void erf_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    bool fast_and_approx = bool(core->get_arg(1));
    api->erf_tile(idst, fast_and_approx);
}

void erfc_tile_init(Compute *api, Riscv32Core *core) {
    api->erfc_tile_init();
}

void erfc_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    bool fast_and_approx = bool(core->get_arg(1));
    api->erfc_tile(idst, fast_and_approx);
}

// eltwise_unary/erfinv

void erfinv_tile_init(Compute *api, Riscv32Core *core) {
    api->erfinv_tile_init();
}

void erfinv_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->erfinv_tile(idst);
}

// eltwise_unary/exp

void exp_tile_init(Compute *api, Riscv32Core *core) {
    api->exp_tile_init();
}

void exp_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->exp_tile(idst);
}

// eltwise_unary/gelu

void gelu_tile_init(Compute *api, Riscv32Core *core) {
    api->gelu_tile_init();
}

void gelu_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    bool fast_and_approx = bool(core->get_arg(1));
    api->gelu_tile(idst, fast_and_approx);
}

// eltwise_unary/i0

void i0_tile_init(Compute *api, Riscv32Core *core) {
    api->i0_tile_init();
}

void i0_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->i0_tile(idst);
}

// eltwise_unary/isinf_isnan

void isinf_tile_init(Compute *api, Riscv32Core *core) {
    api->isinf_tile_init();
}

void isinf_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->isinf_tile(idst);
}

void isposinf_tile_init(Compute *api, Riscv32Core *core) {
    api->isposinf_tile_init();
}

void isposinf_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->isposinf_tile(idst);
}

void isneginf_tile_init(Compute *api, Riscv32Core *core) {
    api->isneginf_tile_init();
}

void isneginf_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->isneginf_tile(idst);
}

void isnan_tile_init(Compute *api, Riscv32Core *core) {
    api->isnan_tile_init();
}

void isnan_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->isnan_tile(idst);
}

void isfinite_tile_init(Compute *api, Riscv32Core *core) {
    api->isfinite_tile_init();
}

void isfinite_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->isfinite_tile(idst);
}

// eltwise_unary/logical_not_noti

void logical_not_unary_tile_init(Compute *api, Riscv32Core *core) {
    api->logical_not_unary_tile_init();
}

void logical_not_unary_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->logical_not_unary_tile(idst);
}

// eltwise_unary/recip

void recip_tile_init(Compute *api, Riscv32Core *core) {
    api->recip_tile_init();
}

void recip_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->recip_tile(idst);
}

// eltwise_unary/relu

void relu_max_tile_init(Compute *api, Riscv32Core *core) {
    api->relu_max_tile_init();
}

void relu_max_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->relu_max_tile(idst, param0);
}

void relu_min_tile_init(Compute *api, Riscv32Core *core) {
    api->relu_min_tile_init();
}

void relu_min_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->relu_min_tile(idst, param0);
}

void relu_tile_init(Compute *api, Riscv32Core *core) {
    api->relu_tile_init();
}

void relu_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->relu_tile(idst);
}

void leaky_relu_tile_init(Compute *api, Riscv32Core *core) {
    api->leaky_relu_tile_init();
}

void leaky_relu_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    uint32_t param0 = core->get_arg(1);
    api->leaky_relu_tile(idst, param0);
}

// eltwise_unary/sqrt

void sqrt_tile_init(Compute *api, Riscv32Core *core) {
    api->sqrt_tile_init();
}

void sqrt_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->sqrt_tile(idst);
}

// eltwise_unary/trigonometry

void sin_tile_init(Compute *api, Riscv32Core *core) {
    api->sin_tile_init();
}

void sin_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->sin_tile(idst);
}

void cos_tile_init(Compute *api, Riscv32Core *core) {
    api->cos_tile_init();
}

void cos_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->cos_tile(idst);
}

void tan_tile_init(Compute *api, Riscv32Core *core) {
    api->tan_tile_init();
}

void tan_tile(Compute *api, Riscv32Core *core) {
    uint32_t idst = core->get_arg(0);
    api->tan_tile(idst);
}

} // namespace

//
//    ComputeHandler
//

ComputeHandler::ComputeHandler(Machine *machine):
        m_machine(machine) { }

ComputeHandler::~ComputeHandler() { }

#define DECL_BUILTIN(name, count) \
    case ComputeBuiltinId::name: \
        name(api, core); \
        break;

void ComputeHandler::call(Riscv32Core *core, int id) {
    Compute *api = m_machine->get_compute_api();
    switch (ComputeBuiltinId(id)) {
COMPUTE_BUILTINS
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

