// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void read_px(Global gx, Local lzero, Pipe px, uint32 H, uint32 W,
             uint32 x_start, uint32 h_start) {
  uint32 src_pos = x_start;
  uint32 dst_pos = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  for (uint32 i = 0; i < 32; i++) {
    if (h_start + i >= H) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)),
                     get_write_ptr(px.cb_id) + (dst_pos << 1), W << 1);
    } else {
      noc_async_read_global_dram(get_write_ptr(px.cb_id) + (dst_pos << 1),
                                 gx.addr, gx.log2_page_size, src_pos << 1,
                                 W << 1);
    }
    src_pos += W;
    dst_pos += W;
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gs, Global gzero, Local lzero, Pipe px, Pipe ps,
            uint32 N, uint32 H, uint32 W, uint32 zero_size, uint32 x_pos,
            uint32 x_stride) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  // read_barrier is below
  px.frame_size = (W / 32);
  ps.frame_size = 1;
  cb_reserve_back(ps.cb_id, ps.frame_size);
  noc_async_read_global_dram(get_write_ptr(ps.cb_id) + (0 << 1), gs.addr,
                             gs.log2_page_size, 0 << 1, 1024 << 1);
  noc_async_read_barrier();
  cb_push_back(ps.cb_id, ps.frame_size);
  uint32 x_start = x_pos;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 h_start = 0; h_start < H; h_start += 32) {
      read_px(gx, lzero, px, H, W, x_start, h_start);
      x_start += W * 32;
    }
    x_start += x_stride;
  }
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Global gs;
  gs.addr = get_arg_val<uint32>(2);
  gs.log2_page_size = get_arg_val<uint32>(3);
  Global gzero;
  gzero.addr = get_arg_val<uint32>(4);
  gzero.log2_page_size = get_arg_val<uint32>(5);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(6);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(7);
  px.frame_size = get_arg_val<uint32>(8);
  Pipe ps;
  ps.cb_id = get_arg_val<uint32>(9);
  ps.frame_size = get_arg_val<uint32>(10);
  uint32 N = get_arg_val<uint32>(11);
  uint32 H = get_arg_val<uint32>(12);
  uint32 W = get_arg_val<uint32>(13);
  uint32 zero_size = get_arg_val<uint32>(14);
  uint32 x_pos = get_arg_val<uint32>(15);
  uint32 x_stride = get_arg_val<uint32>(16);
  kernel(gx, gs, gzero, lzero, px, ps, N, H, W, zero_size, x_pos, x_stride);
}

