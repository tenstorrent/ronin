// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally
// "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank_output_tiles_partitioned.cpp"

void kernel(Global ga, Global gb, Pipe pa, Pipe pb, uint32 Mt, uint32 Kt,
            uint32 Nt, uint32 bcast_b, uint32 out_tile_pos,
            uint32 out_num_tiles) {
  constexpr uint32 onetile = 1024;

  // ACHTUNG: These values can be also computed on host
  uint32 MtNt = Mt * Nt;
  uint32 KtNt = Kt * Nt;
  uint32 ga_pos = (out_tile_pos / Nt) * Kt;
  uint32 out_mtnt = out_tile_pos % MtNt;
  uint32 out_nt = out_tile_pos % Nt;
  uint32 gb_pos = out_nt;
  if (bcast_b == 0) {
    uint32 out_b = out_tile_pos / MtNt;
    gb_pos += out_b * KtNt;
  }
  ga_pos *= onetile;
  gb_pos *= onetile;

  for (uint32 n = 0; n < out_num_tiles; n++) {
    for (uint32 kt = 0; kt < Kt; kt++) {
      cb_reserve_back(pa.cb_id, pa.frame_size);
      noc_async_read_global_dram(get_write_ptr(pa.cb_id) + (0 << 1), ga.addr,
                                 ga.log2_page_size, ga_pos << 1, onetile << 1);
      noc_async_read_barrier();
      cb_push_back(pa.cb_id, pa.frame_size);
      cb_reserve_back(pb.cb_id, pb.frame_size);
      noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                                 gb.log2_page_size, gb_pos << 1, onetile << 1);
      noc_async_read_barrier();
      cb_push_back(pb.cb_id, pb.frame_size);
      ga_pos += onetile;
      gb_pos += Nt * onetile;
    }
    out_mtnt++;
    out_nt++;
    gb_pos -= KtNt * onetile;
    gb_pos += onetile;
    if (out_nt == Nt) {
      out_nt = 0;
      gb_pos -= Nt * onetile;
      if (out_mtnt == MtNt) {
        out_mtnt = 0;
        if (bcast_b == 0) {
          gb_pos += KtNt * onetile;
        }
      }
    } else {
      ga_pos -= Kt * onetile;
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
  uint32 Mt = get_arg_val<uint32>(8);
  uint32 Kt = get_arg_val<uint32>(9);
  uint32 Nt = get_arg_val<uint32>(10);
  uint32 bcast_b = get_arg_val<uint32>(11);
  uint32 out_tile_pos = get_arg_val<uint32>(12);
  uint32 out_num_tiles = get_arg_val<uint32>(13);
  kernel(ga, gb, pa, pb, Mt, Kt, Nt, bcast_b, out_tile_pos, out_num_tiles);
}

