// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally "writer_unary_8bank.cpp"

void kernel(Global gy, Pipe py, uint32 gy_pos, uint32 num_tiles) {
  for (uint32 i = 0; i < num_tiles; i++) {
    cb_wait_front(py.cb_id, py.frame_size);
    noc_async_write_global_dram(get_read_ptr(py.cb_id) + (0 << 1), gy.addr,
                                gy.log2_page_size, gy_pos << 1, 1024 << 1);
    noc_async_write_barrier();
    cb_pop_front(py.cb_id, py.frame_size);
    gy_pos += 1024;
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
  uint32 num_tiles = get_arg_val<uint32>(5);
  kernel(gy, py, gy_pos, num_tiles);
}

