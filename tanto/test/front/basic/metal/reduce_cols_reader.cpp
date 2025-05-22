// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// originally "reader_unary_transpose_wh_interleaved.cpp"

void kernel(Global gx, Global gs, Pipe px, Pipe ps, uint32 gx_pos, uint32 N,
            uint32 Ht, uint32 Wt, uint32 HtWt) {
  uint32 gx_dh = Wt * 1024;
  uint32 gx_dw = HtWt * 1024;

  cb_reserve_back(ps.cb_id, ps.frame_size);
  noc_async_read_global_dram(get_write_ptr(ps.cb_id) + (0 << 1), gs.addr,
                             gs.log2_page_size, 0 << 1, 1024 << 1);
  noc_async_read_barrier();
  cb_push_back(ps.cb_id, ps.frame_size);

  // read NHW tensor in NWH order
  for (uint32 n = 0; n < N; n++) {
    uint32 gx_idx = gx_pos;
    for (uint32 w = 0; w < Wt; w++) {
      for (uint32 h = 0; h < Ht; h++) {
        cb_reserve_back(px.cb_id, px.frame_size);
        noc_async_read_global_dram(get_write_ptr(px.cb_id) + (0 << 1), gx.addr,
                                   gx.log2_page_size, gx_idx << 1, 1024 << 1);
        noc_async_read_barrier();
        cb_push_back(px.cb_id, px.frame_size);
        gx_idx += gx_dh;
      }
      gx_idx -= gx_dw;
      gx_idx += 1024;
    }
    gx_pos += gx_dw;
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
  uint32 N = get_arg_val<uint32>(9);
  uint32 Ht = get_arg_val<uint32>(10);
  uint32 Wt = get_arg_val<uint32>(11);
  uint32 HtWt = get_arg_val<uint32>(12);
  kernel(gx, gs, px, ps, gx_pos, N, Ht, Wt, HtWt);
}

