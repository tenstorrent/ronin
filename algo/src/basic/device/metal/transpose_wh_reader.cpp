// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally "reader_unary_transpose_wh.cpp"

void kernel(Global gx, Pipe px, uint32 gx_pos, uint32 N, uint32 Ht, uint32 Wt,
            uint32 HtWt) {
  constexpr uint32 onetile = 1024;
  for (uint32 n = 0; n < N; n++) {
    uint32 gx_idx = gx_pos;
    for (uint32 w = 0; w < Wt; w++) {
      for (uint32 h = 0; h < Ht; h++) {
        cb_reserve_back(px.cb_id, px.frame_size);
        noc_async_read_global_dram(get_write_ptr(px.cb_id) + (0 << 1), gx.addr,
                                   gx.log2_page_size, gx_idx << 1,
                                   onetile << 1);
        noc_async_read_barrier();
        cb_push_back(px.cb_id, px.frame_size);
        gx_idx += Wt * onetile;
      }
      gx_idx -= HtWt * onetile;
      gx_idx += onetile;
    }
    gx_pos += HtWt * onetile;
  }
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(2);
  px.frame_size = get_arg_val<uint32>(3);
  uint32 gx_pos = get_arg_val<uint32>(4);
  uint32 N = get_arg_val<uint32>(5);
  uint32 Ht = get_arg_val<uint32>(6);
  uint32 Wt = get_arg_val<uint32>(7);
  uint32 HtWt = get_arg_val<uint32>(8);
  kernel(gx, px, gx_pos, N, Ht, Wt, HtWt);
}

