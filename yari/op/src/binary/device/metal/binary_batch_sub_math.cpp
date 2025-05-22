// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

static constexpr uint32 OP_ADD = 0, OP_SUB = 1, OP_MUL = 2;

static constexpr uint32 op_code = uint32(1);

void binary_op(Pipe pa, Pipe pb, uint32 index) {

  sub_tiles(pa.cb_id, pb.cb_id, index, index, index);
}

void kernel(Pipe pa, Pipe pb, Pipe pc, uint32 num_frames, uint32 frame_tiles) {
  tanto_unpack_binary_init(pa.cb_id, pb.cb_id);
  tanto_sub_init();
  tanto_pack_init(pc.cb_id);
  pa.frame_size = frame_tiles;
  pb.frame_size = frame_tiles;
  pc.frame_size = frame_tiles;
  for (uint32 frame = 0; frame < num_frames; frame++) {
    cb_reserve_back(pc.cb_id, pc.frame_size);
    cb_wait_front(pa.cb_id, pa.frame_size);
    cb_wait_front(pb.cb_id, pb.frame_size);
    tile_regs_acquire();
    tile_regs_wait();
    for (uint32 i = 0; i < frame_tiles; i++) {
      binary_op(pa, pb, i);
    }
    for (uint32 i = 0; i < frame_tiles; i++) {
      pack_tile(i, pc.cb_id);
    }
    cb_pop_front(pa.cb_id, pa.frame_size);
    cb_pop_front(pb.cb_id, pb.frame_size);
    cb_push_back(pc.cb_id, pc.frame_size);
    tile_regs_commit();
    tile_regs_release();
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
  uint32 num_frames = get_arg_val<uint32>(6);
  uint32 frame_tiles = get_arg_val<uint32>(7);
  tanto_compute_init();
  kernel(pa, pb, pc, num_frames, frame_tiles);
}
} // namespace NAMESPACE

