// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global ga, Global gb, Pipe pa, Pipe pb, uint32 ga_pos,
            uint32 gb_pos, uint32 num_blocks, uint32 block_tiles) {
  uint32 block_items = block_tiles * 1024;
  for (uint32 i = 0; i < num_blocks; i++) {
    cb_reserve_back(pa.cb_id, pa.frame_size);
    cb_reserve_back(pb.cb_id, pb.frame_size);
    noc_async_read_global_dram(get_write_ptr(pa.cb_id) + (0 << 1), ga.addr,
                               ga.log2_page_size, ga_pos << 1,
                               block_items << 1);
    noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                               gb.log2_page_size, gb_pos << 1,
                               block_items << 1);
    noc_async_read_barrier();
    cb_push_back(pa.cb_id, pa.frame_size);
    cb_push_back(pb.cb_id, pb.frame_size);
    ga_pos += block_items;
    gb_pos += block_items;
  }
}

void kernel_main() {
  Global ga;
  ga.addr = get_arg_val<uint32>(0);
  ga.log2_page_size = get_arg_val<uint32>(1);
  Global gb;
  gb.addr = get_arg_val<uint32>(2);
  gb.log2_page_size = get_arg_val<uint32>(3);
  Pipe pa;
  pa.cb_id = get_arg_val<uint32>(4);
  pa.frame_size = get_arg_val<uint32>(5);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(6);
  pb.frame_size = get_arg_val<uint32>(7);
  uint32 ga_pos = get_arg_val<uint32>(8);
  uint32 gb_pos = get_arg_val<uint32>(9);
  uint32 num_blocks = get_arg_val<uint32>(10);
  uint32 block_tiles = get_arg_val<uint32>(11);
  kernel(ga, gb, pa, pb, ga_pos, gb_pos, num_blocks, block_tiles);
}

