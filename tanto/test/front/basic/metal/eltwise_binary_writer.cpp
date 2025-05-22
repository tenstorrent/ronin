// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gc, Pipe pc, uint32 gc_pos, uint32 num_blocks,
            uint32 block_tiles) {
  uint32 block_items = block_tiles * 1024;
  for (uint32 i = 0; i < num_blocks; i++) {
    cb_wait_front(pc.cb_id, pc.frame_size);
    noc_async_write_global_dram(get_read_ptr(pc.cb_id) + (0 << 1), gc.addr,
                                gc.log2_page_size, gc_pos << 1,
                                block_items << 1);
    noc_async_write_barrier();
    cb_pop_front(pc.cb_id, pc.frame_size);
    gc_pos += block_items;
  }
}

void kernel_main() {
  Global gc;
  gc.addr = get_arg_val<uint32>(0);
  gc.log2_page_size = get_arg_val<uint32>(1);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(2);
  pc.frame_size = get_arg_val<uint32>(3);
  uint32 gc_pos = get_arg_val<uint32>(4);
  uint32 num_blocks = get_arg_val<uint32>(5);
  uint32 block_tiles = get_arg_val<uint32>(6);
  kernel(gc, pc, gc_pos, num_blocks, block_tiles);
}

