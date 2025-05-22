// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally
// "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp"

void kernel(Global ga, Global gb, Pipe pa, Pipe pb, uint32 ga_pos,
            uint32 gb_pos, uint32 batch, uint32 Mt, uint32 Kt, uint32 Nt,
            uint32 MtKt, uint32 KtNt, uint32 bcast_b) {
  constexpr uint32 onetile = 1024;
  for (uint32 nb = 0; nb < batch; nb++) {
    uint32 ga_idx = ga_pos;
    for (uint32 mt = 0; mt < Mt; mt++) {
      uint32 gb_idx = gb_pos;
      for (uint32 nt = 0; nt < Nt; nt++) {
        for (uint32 kt = 0; kt < Kt; kt++) {
          cb_reserve_back(pa.cb_id, pa.frame_size);
          noc_async_read_global_dram(get_write_ptr(pa.cb_id) + (0 << 1),
                                     ga.addr, ga.log2_page_size, ga_idx << 1,
                                     onetile << 1);
          noc_async_read_barrier();
          cb_push_back(pa.cb_id, pa.frame_size);
          cb_reserve_back(pb.cb_id, pb.frame_size);
          noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1),
                                     gb.addr, gb.log2_page_size, gb_idx << 1,
                                     onetile << 1);
          noc_async_read_barrier();
          cb_push_back(pb.cb_id, pb.frame_size);
          ga_idx += onetile;
          gb_idx += Nt * onetile;
        } // kt
        gb_idx -= KtNt * onetile;
        gb_idx += onetile;
        ga_idx -= Kt * onetile;
      } // nt
      ga_idx += Kt * onetile;
    } // mt
    ga_pos += MtKt * onetile;
    if (bcast_b == 0) {
      gb_pos += KtNt * onetile;
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
  uint32 batch = get_arg_val<uint32>(10);
  uint32 Mt = get_arg_val<uint32>(11);
  uint32 Kt = get_arg_val<uint32>(12);
  uint32 Nt = get_arg_val<uint32>(13);
  uint32 MtKt = get_arg_val<uint32>(14);
  uint32 KtNt = get_arg_val<uint32>(15);
  uint32 bcast_b = get_arg_val<uint32>(16);
  kernel(ga, gb, pa, pb, ga_pos, gb_pos, batch, Mt, Kt, Nt, MtKt, KtNt,
         bcast_b);
}

