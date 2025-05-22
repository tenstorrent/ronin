// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// Originally
// "tt_metal/programming_examples/matmul_common/kernels/compute/bmm.cpp"

void kernel(Pipe pa, Pipe pb, Pipe pc, uint32 batch, uint32 Mt, uint32 Kt,
            uint32 Nt) {
  tanto_unpack_matmul_init(pa.cb_id, pb.cb_id, false);
  tanto_matmul_init(false);
  tanto_pack_init(pc.cb_id);
  for (uint32 nb = 0; nb < batch; nb++) {
    for (uint32 mt = 0; mt < Mt; mt++) {
      for (uint32 nt = 0; nt < Nt; nt++) {
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32 kt = 0; kt < Kt; kt++) {
          cb_wait_front(pa.cb_id, pa.frame_size);
          cb_wait_front(pb.cb_id, pb.frame_size);
          matmul_tiles(pa.cb_id, pb.cb_id, 0, 0, 0, false);
          cb_pop_front(pb.cb_id, pb.frame_size);
          cb_pop_front(pa.cb_id, pa.frame_size);
        }
        cb_reserve_back(pc.cb_id, pc.frame_size);
        pack_tile(0, pc.cb_id);
        cb_push_back(pc.cb_id, pc.frame_size);
        tile_regs_commit();
        tile_regs_release();
      }
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
  uint32 batch = get_arg_val<uint32>(6);
  uint32 Mt = get_arg_val<uint32>(7);
  uint32 Kt = get_arg_val<uint32>(8);
  uint32 Nt = get_arg_val<uint32>(9);
  tanto_compute_init();
  kernel(pa, pb, pc, batch, Mt, Kt, Nt);
}
} // namespace NAMESPACE

