// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally
// "tt_metal/programming_examples/matmul_common/kernels/dataflow/writer_unary_interleaved_start_id.cpp"

void kernel(Global gc, Pipe pc, uint32 tile_pos, uint32 num_tiles) {
  constexpr uint32 onetile = 1024;
  uint32 gc_pos = tile_pos * onetile;
  for (uint32 i = 0; i < num_tiles; i++) {
    cb_wait_front(pc.cb_id, pc.frame_size);
    noc_async_write_global_dram(get_read_ptr(pc.cb_id) + (0 << 1), gc.addr,
                                gc.log2_page_size, gc_pos << 1, onetile << 1);
    noc_async_write_barrier();
    cb_pop_front(pc.cb_id, pc.frame_size);
    gc_pos += onetile;
  }
}

void kernel_main() {
  Global gc;
  gc.addr = get_arg_val<uint32>(0);
  gc.log2_page_size = get_arg_val<uint32>(1);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(2);
  pc.frame_size = get_arg_val<uint32>(3);
  uint32 tile_pos = get_arg_val<uint32>(4);
  uint32 num_tiles = get_arg_val<uint32>(5);
  kernel(gc, pc, tile_pos, num_tiles);
}

