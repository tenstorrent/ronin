// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// Originally "reduce_w.cpp"

static constexpr uint32 REDUCE_OP_MAX = 0, REDUCE_OP_SUM = 1;

static constexpr uint32 reduce_op_code = uint32(1);

void reduce_op(Pipe px, Pipe ps) {

  reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(px.cb_id, ps.cb_id, 0, 0,
                                                    0);
}

void kernel(Pipe px, Pipe ps, Pipe py, uint32 Ht, uint32 Wt, uint32 NC) {
  tanto_unpack_reduce_rows_init(px.cb_id, ps.cb_id);
  tanto_reduce_sum_rows_init();
  tanto_pack_row_init(py.cb_id);
  cb_wait_front(ps.cb_id, ps.frame_size);
  for (uint32 nc = 0; nc < NC; nc++) {
    for (uint32 ht = 0; ht < Ht; ht++) {
      // tiles are expected to be coming in in NCHW order (W-contiguous)
      tile_regs_acquire();
      tile_regs_wait();
      for (uint32 wt = 0; wt < Wt; wt++) {
        cb_wait_front(px.cb_id, px.frame_size);
        reduce_op(px, ps);
        cb_pop_front(px.cb_id, px.frame_size);
      }
      cb_reserve_back(py.cb_id, py.frame_size);
      pack_tile(0, py.cb_id);
      cb_push_back(py.cb_id, py.frame_size);
      tile_regs_commit();
      tile_regs_release();
    }
  }
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe ps;
  ps.cb_id = get_arg_val<uint32>(2);
  ps.frame_size = get_arg_val<uint32>(3);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(4);
  py.frame_size = get_arg_val<uint32>(5);
  uint32 Ht = get_arg_val<uint32>(6);
  uint32 Wt = get_arg_val<uint32>(7);
  uint32 NC = get_arg_val<uint32>(8);
  tanto_compute_init();
  kernel(px, ps, py, Ht, Wt, NC);
}
} // namespace NAMESPACE

