// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally "writer_unary.cpp"

void kernel(Global gy, Pipe py, uint32 gy_pos, uint32 num_blocks,
            uint32 block_tiles) {
  uint32 block_items = block_tiles * 1024;
  py.frame_size = block_tiles;
  for (uint32 b = 0; b < num_blocks; b++) {
    cb_wait_front(py.cb_id, py.frame_size);
    noc_async_write_global_dram(get_read_ptr(py.cb_id) + (0 << 1), gy.addr,
                                gy.log2_page_size, gy_pos << 1,
                                block_items << 1);
    noc_async_write_barrier();
    cb_pop_front(py.cb_id, py.frame_size);
    gy_pos += block_items;
  }
}

void kernel_main() {
  Global gy;
  gy.addr = get_arg_val<uint32>(0);
  gy.log2_page_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  uint32 gy_pos = get_arg_val<uint32>(4);
  uint32 num_blocks = get_arg_val<uint32>(5);
  uint32 block_tiles = get_arg_val<uint32>(6);
  kernel(gy, py, gy_pos, num_blocks, block_tiles);
}

