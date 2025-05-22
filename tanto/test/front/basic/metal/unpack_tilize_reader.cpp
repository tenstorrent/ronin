// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally "reader_unary_push_4.cpp"

void kernel(Global gx, Pipe px, uint32 gx_pos, uint32 num_blocks,
            uint32 block_tiles) {
  uint32 block_items = block_tiles * 1024;
  px.frame_size = block_tiles;
  for (uint32 b = 0; b < num_blocks; b++) {
    cb_reserve_back(px.cb_id, px.frame_size);
    noc_async_read_global_dram(get_write_ptr(px.cb_id) + (0 << 1), gx.addr,
                               gx.log2_page_size, gx_pos << 1,
                               block_items << 1);
    noc_async_read_barrier();
    cb_push_back(px.cb_id, px.frame_size);
    gx_pos += block_items;
  }
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(2);
  px.frame_size = get_arg_val<uint32>(3);
  uint32 gx_pos = get_arg_val<uint32>(4);
  uint32 num_blocks = get_arg_val<uint32>(5);
  uint32 block_tiles = get_arg_val<uint32>(6);
  kernel(gx, px, gx_pos, num_blocks, block_tiles);
}

