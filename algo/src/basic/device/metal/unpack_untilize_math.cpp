// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// Originally "untilize.cpp"

void kernel(Pipe px, Pipe py, uint32 num_blocks, uint32 block_tiles) {
  tanto_unpack_untilize_block_init(px.cb_id);
  tanto_copy_init();
  tanto_pack_init(py.cb_id);
  px.frame_size = block_tiles;
  py.frame_size = block_tiles;
  for (uint32 b = 0; b < num_blocks; b++) {
    cb_wait_front(px.cb_id, px.frame_size);
    cb_reserve_back(py.cb_id, py.frame_size);
    untilize_block<1>(px.cb_id, block_tiles, py.cb_id);
    cb_push_back(py.cb_id, py.frame_size);
    cb_pop_front(px.cb_id, px.frame_size);
  }
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  uint32 num_blocks = get_arg_val<uint32>(4);
  uint32 block_tiles = get_arg_val<uint32>(5);
  tanto_compute_init();
  kernel(px, py, num_blocks, block_tiles);
}
} // namespace NAMESPACE

