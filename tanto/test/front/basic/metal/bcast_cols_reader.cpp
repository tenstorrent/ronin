// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// originally "reader_bcast_w_8bank.cpp"

void kernel(Global ga, Global gb, Pipe pa, Pipe pb, uint32 ga_pos,
            uint32 gb_pos, uint32 NC, uint32 Ht, uint32 Wt, uint32 gb_no_nc) {
  constexpr uint32 onetile = 1024;
  for (uint32 nc = 0; nc < NC; nc++) {
    for (uint32 ht = 0; ht < Ht; ht++) {
      cb_reserve_back(pb.cb_id, pb.frame_size);
      noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                                 gb.log2_page_size, gb_pos << 1, onetile << 1);
      noc_async_read_barrier();
      cb_push_back(pb.cb_id, pb.frame_size);
      gb_pos += onetile;
      for (uint32 wt = 0; wt < Wt; wt++) {
        cb_reserve_back(pa.cb_id, pa.frame_size);
        noc_async_read_global_dram(get_write_ptr(pa.cb_id) + (0 << 1), ga.addr,
                                   ga.log2_page_size, ga_pos << 1,
                                   onetile << 1);
        noc_async_read_barrier();
        cb_push_back(pa.cb_id, pa.frame_size);
        ga_pos += onetile;
      }
    }
    if (gb_no_nc) {
      gb_pos -= Ht * onetile;
    }
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
  uint32 NC = get_arg_val<uint32>(10);
  uint32 Ht = get_arg_val<uint32>(11);
  uint32 Wt = get_arg_val<uint32>(12);
  uint32 gb_no_nc = get_arg_val<uint32>(13);
  kernel(ga, gb, pa, pb, ga_pos, gb_pos, NC, Ht, Wt, gb_no_nc);
}

