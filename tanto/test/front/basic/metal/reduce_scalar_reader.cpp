// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally "reader_unary_8bank_reduce.cpp / reader_unary_8bank.cpp"

void kernel(Global gx, Global gs, Pipe px, Pipe ps, uint32 gx_pos,
            uint32 num_tiles) {
  cb_reserve_back(ps.cb_id, ps.frame_size);
  noc_async_read_global_dram(get_write_ptr(ps.cb_id) + (0 << 1), gs.addr,
                             gs.log2_page_size, 0 << 1, 1024 << 1);
  noc_async_read_barrier();
  cb_push_back(ps.cb_id, ps.frame_size);

  for (uint32 i = 0; i < num_tiles; i++) {
    cb_reserve_back(px.cb_id, px.frame_size);
    noc_async_read_global_dram(get_write_ptr(px.cb_id) + (0 << 1), gx.addr,
                               gx.log2_page_size, gx_pos << 1, 1024 << 1);
    noc_async_read_barrier();
    cb_push_back(px.cb_id, px.frame_size);
    gx_pos += 1024;
  }
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Global gs;
  gs.addr = get_arg_val<uint32>(2);
  gs.log2_page_size = get_arg_val<uint32>(3);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(4);
  px.frame_size = get_arg_val<uint32>(5);
  Pipe ps;
  ps.cb_id = get_arg_val<uint32>(6);
  ps.frame_size = get_arg_val<uint32>(7);
  uint32 gx_pos = get_arg_val<uint32>(8);
  uint32 num_tiles = get_arg_val<uint32>(9);
  kernel(gx, gs, px, ps, gx_pos, num_tiles);
}

