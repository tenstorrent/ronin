// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void read_px(Local lx, Local lp, Local lzero, Pipe px, uint32 C,
             uint32 lp_pos) {
  cb_reserve_back(px.cb_id, px.frame_size);
  noc_async_read_one_packet_set_state(get_noc_addr(0), C << 1);
  uint32 px_pos = 0;
  for (uint32 i = 0; i < 32; i++) {
    uint32 lx_pos =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(lp.addr)[(lp_pos + i)];
    if ((lx_pos >> 31) != 0) {
      noc_async_read_one_packet_with_state(
          lzero.addr + (0 << 1), get_write_ptr(px.cb_id) + (px_pos << 1));
    } else {
      noc_async_read_one_packet_with_state(
          lx.addr + (lx_pos << 1), get_write_ptr(px.cb_id) + (px_pos << 1));
    }
    px_pos += C;
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gw, Global gp, Global gzero, Local lx, Local lp,
            Local lzero, Pipe px, Pipe pw, uint32 N, uint32 C, uint32 HWC,
            uint32 PQ, uint32 zero_size, uint32 x_pos, uint32 x_stride) {
  px.frame_size = (C / 32);
  pw.frame_size = 4;
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  noc_async_read_barrier();
  uint32 x_start = x_pos;
  for (uint32 n = 0; n < N; n++) {
    noc_async_read_global_dram(lx.addr + (0 << 1), gx.addr, gx.log2_page_size,
                               x_start << 1, HWC << 1);
    noc_async_read_barrier();
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      cb_reserve_back(pw.cb_id, pw.frame_size);
      noc_async_read_global_dram(get_write_ptr(pw.cb_id) + (0 << 1), gw.addr,
                                 gw.log2_page_size, (pq_start * 4) << 1,
                                 128 << 1);
      noc_async_read_global_dram(lp.addr + (0 << 2), gp.addr, gp.log2_page_size,
                                 (pq_start * 4) << 2, 128 << 2);
      noc_async_read_barrier();
      cb_push_back(pw.cb_id, pw.frame_size);
      for (int i = 0; i < 128; i += 32) {
        read_px(lx, lp, lzero, px, C, i);
      }
    } // pq_start
    x_start += x_stride;
  }
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Global gw;
  gw.addr = get_arg_val<uint32>(2);
  gw.log2_page_size = get_arg_val<uint32>(3);
  Global gp;
  gp.addr = get_arg_val<uint32>(4);
  gp.log2_page_size = get_arg_val<uint32>(5);
  Global gzero;
  gzero.addr = get_arg_val<uint32>(6);
  gzero.log2_page_size = get_arg_val<uint32>(7);
  Local lx;
  lx.addr = get_arg_val<uint32>(8);
  Local lp;
  lp.addr = get_arg_val<uint32>(9);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(10);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(11);
  px.frame_size = get_arg_val<uint32>(12);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(13);
  pw.frame_size = get_arg_val<uint32>(14);
  uint32 N = get_arg_val<uint32>(15);
  uint32 C = get_arg_val<uint32>(16);
  uint32 HWC = get_arg_val<uint32>(17);
  uint32 PQ = get_arg_val<uint32>(18);
  uint32 zero_size = get_arg_val<uint32>(19);
  uint32 x_pos = get_arg_val<uint32>(20);
  uint32 x_stride = get_arg_val<uint32>(21);
  kernel(gx, gw, gp, gzero, lx, lp, lzero, px, pw, N, C, HWC, PQ, zero_size,
         x_pos, x_stride);
}

