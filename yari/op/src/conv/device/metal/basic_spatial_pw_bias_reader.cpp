// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void read_px(Global gx, Local lzero, Pipe px, uint32 HW_upper, uint32 C,
             uint32 x_start, uint32 hw_start) {
  uint32 src_pos = x_start;
  uint32 dst_pos = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  for (uint32 i = 0; i < 32; i++) {
    if (hw_start + i >= HW_upper) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)),
                     get_write_ptr(px.cb_id) + (dst_pos << 1), C << 1);
    } else {
      noc_async_read_global_dram(get_write_ptr(px.cb_id) + (dst_pos << 1),
                                 gx.addr, gx.log2_page_size, src_pos << 1,
                                 C << 1);
    }
    src_pos += C;
    dst_pos += C;
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gb, Global gzero, Local lzero, Pipe px, Pipe pb,
            uint32 N, uint32 HW, uint32 C, uint32 K, uint32 HW_upper,
            uint32 zero_size, uint32 x_pos, uint32 x_stride) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  // read_barrier is below
  px.frame_size = (C / 32);
  pb.frame_size = (K / 32);
  cb_reserve_back(pb.cb_id, pb.frame_size);
  noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                             gb.log2_page_size, 0 << 1, (K * 32) << 1);
  noc_async_read_barrier();
  cb_push_back(pb.cb_id, pb.frame_size);
  uint32 x_batch = x_pos;
  for (uint32 n = 0; n < N; n++) {
    uint32 x_start = x_batch;
    for (uint32 hw_start = 0; hw_start < HW; hw_start += 32) {
      read_px(gx, lzero, px, HW_upper, C, x_start, hw_start);
      x_start += C * 32;
    } // hw_start
    x_batch += x_stride;
  } // n
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Global gb;
  gb.addr = get_arg_val<uint32>(2);
  gb.log2_page_size = get_arg_val<uint32>(3);
  Global gzero;
  gzero.addr = get_arg_val<uint32>(4);
  gzero.log2_page_size = get_arg_val<uint32>(5);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(6);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(7);
  px.frame_size = get_arg_val<uint32>(8);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(9);
  pb.frame_size = get_arg_val<uint32>(10);
  uint32 N = get_arg_val<uint32>(11);
  uint32 HW = get_arg_val<uint32>(12);
  uint32 C = get_arg_val<uint32>(13);
  uint32 K = get_arg_val<uint32>(14);
  uint32 HW_upper = get_arg_val<uint32>(15);
  uint32 zero_size = get_arg_val<uint32>(16);
  uint32 x_pos = get_arg_val<uint32>(17);
  uint32 x_stride = get_arg_val<uint32>(18);
  kernel(gx, gb, gzero, lzero, px, pb, N, HW, C, K, HW_upper, zero_size, x_pos,
         x_stride);
}

