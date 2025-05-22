// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// Originally "bcast_w.cpp"

static constexpr uint32 OP_ADD = 0, OP_SUB = 1, OP_MUL = 2;

static constexpr uint32 bcast_op_code = uint32(2);

void bcast_op(

    Pipe pa, Pipe pb) {

  any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
      pa.cb_id, pb.cb_id, 0, 0, 0);
}

void kernel(Pipe pa, Pipe pb, Pipe pc, uint32 B, uint32 Ht, uint32 Wt) {
  tanto_unpack_bcast_cols_init(pa.cb_id, pb.cb_id);
  tanto_mul_bcast_cols_init();
  tanto_pack_init(pc.cb_id);
  for (uint32 b = 0; b < B; b++) {
    for (uint32 h = 0; h < Ht; h++) {
      cb_wait_front(pb.cb_id, pb.frame_size);
      for (uint32 w = 0; w < Wt; w++) {
        cb_reserve_back(pc.cb_id, pc.frame_size);
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(pa.cb_id, pa.frame_size);
        bcast_op(pa, pb);
        pack_tile(0, pc.cb_id);
        cb_pop_front(pa.cb_id, pa.frame_size);
        cb_push_back(pc.cb_id, pc.frame_size);
        tile_regs_commit();
        tile_regs_release();
      }
      cb_pop_front(pb.cb_id, pb.frame_size);
    }
  }
}

void MAIN {
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(0);
  pa.frame_size = get_arg_val<uint32>(1);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(2);
  pb.frame_size = get_arg_val<uint32>(3);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(4);
  pc.frame_size = get_arg_val<uint32>(5);
  uint32 B = get_arg_val<uint32>(6);
  uint32 Ht = get_arg_val<uint32>(7);
  uint32 Wt = get_arg_val<uint32>(8);
  tanto_compute_init();
  kernel(pa, pb, pc, B, Ht, Wt);
}
} // namespace NAMESPACE

