// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void clear_lx(Local lx, Local lzero, uint32 H, uint32 WC, uint32 before_h,
              uint32 after_h, uint32 before_wc, uint32 after_wc) {
  uint32 WC_full = before_wc + WC + after_wc;
  uint32 pos = 0;
  for (uint32 i = 0; i < before_h; i++) {
    noc_async_read(get_noc_addr(lzero.addr + (0 << 1)), lx.addr + (pos << 1),
                   WC_full << 1);
    pos += WC_full;
  }
  for (uint32 i = 0; i < H; i++) {
    if (before_wc != 0) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)), lx.addr + (pos << 1),
                     before_wc << 1);
    }
    pos += before_wc + WC;
    if (after_wc != 0) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)), lx.addr + (pos << 1),
                     after_wc << 1);
    }
    pos += after_wc;
  }
  for (uint32 i = 0; i < after_h; i++) {
    noc_async_read(get_noc_addr(lzero.addr + (0 << 1)), lx.addr + (pos << 1),
                   WC_full << 1);
    pos += WC_full;
  }
  noc_async_read_barrier();
}

void load_lx(Global gx, Local lx, uint32 H, uint32 WC, uint32 before_wc,
             uint32 after_wc, uint32 before_hwc, uint32 x_start) {
  uint32 pos = before_hwc + before_wc;
  uint32 stride = WC + after_wc + before_wc;
  for (uint32 i = 0; i < H; i++) {
    noc_async_read_global_dram(lx.addr + (pos << 1), gx.addr, gx.log2_page_size,
                               x_start << 1, WC << 1);
    x_start += WC;
    pos += stride;
  }
  noc_async_read_barrier();
}

void read_px(Local lx, Pipe px, uint32 R, uint32 SC, uint32 RSC_rnd,
             uint32 offset_wc, uint32 delta_p, uint32 delta_q, uint32 delta_r,
             uint32 end_q, uint32 &p_term, uint32 &q_term) {
  uint32 dst_start = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  noc_async_read_one_packet_set_state(get_noc_addr(0), SC << 1);
  for (uint32 i = 0; i < 32; i++) {
    uint32 dst_pos = dst_start;
    uint32 r_term = 0;
    for (uint32 r = 0; r < R; r++) {
      uint32 src_pos = p_term + q_term + r_term + offset_wc;
      noc_async_read_one_packet_with_state(
          lx.addr + (src_pos << 1), get_write_ptr(px.cb_id) + (dst_pos << 1));
      dst_pos += SC;
      r_term += delta_r;
    }
    dst_start += RSC_rnd;
    q_term += delta_q;
    if (q_term >= end_q) {
      q_term = 0;
      p_term += delta_p;
    }
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gb, Global gzero, Local lx, Local lzero, Pipe px,
            Pipe pb, uint32 N, uint32 H, uint32 K, uint32 R, uint32 WC,
            uint32 PQ, uint32 SC, uint32 RSC_rnd, uint32 before_h,
            uint32 after_h, uint32 before_wc, uint32 after_wc, uint32 offset_wc,
            uint32 before_hwc, uint32 delta_p, uint32 delta_q, uint32 delta_r,
            uint32 end_q, uint32 x_pos, uint32 x_stride, uint32 zero_size) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  // read_barrier is below
  px.frame_size = (RSC_rnd / 32);
  pb.frame_size = (K / 32);
  cb_reserve_back(pb.cb_id, pb.frame_size);
  noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                             gb.log2_page_size, 0 << 1, (K * 32) << 1);
  noc_async_read_barrier();
  cb_push_back(pb.cb_id, pb.frame_size);
  clear_lx(lx, lzero, H, WC, before_h, after_h, before_wc, after_wc);
  uint32 x_start = x_pos;
  uint32 p_term = 0;
  uint32 q_term = 0;
  for (uint32 n = 0; n < N; n++) {
    load_lx(gx, lx, H, WC, before_wc, after_wc, before_hwc, x_start);
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      read_px(lx, px, R, SC, RSC_rnd, offset_wc, delta_p, delta_q, delta_r,
              end_q, p_term, q_term);
    } // pq_start
    x_start += x_stride;
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
  Local lx;
  lx.addr = get_arg_val<uint32>(6);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(7);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(8);
  px.frame_size = get_arg_val<uint32>(9);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(10);
  pb.frame_size = get_arg_val<uint32>(11);
  uint32 N = get_arg_val<uint32>(12);
  uint32 H = get_arg_val<uint32>(13);
  uint32 K = get_arg_val<uint32>(14);
  uint32 R = get_arg_val<uint32>(15);
  uint32 WC = get_arg_val<uint32>(16);
  uint32 PQ = get_arg_val<uint32>(17);
  uint32 SC = get_arg_val<uint32>(18);
  uint32 RSC_rnd = get_arg_val<uint32>(19);
  uint32 before_h = get_arg_val<uint32>(20);
  uint32 after_h = get_arg_val<uint32>(21);
  uint32 before_wc = get_arg_val<uint32>(22);
  uint32 after_wc = get_arg_val<uint32>(23);
  uint32 offset_wc = get_arg_val<uint32>(24);
  uint32 before_hwc = get_arg_val<uint32>(25);
  uint32 delta_p = get_arg_val<uint32>(26);
  uint32 delta_q = get_arg_val<uint32>(27);
  uint32 delta_r = get_arg_val<uint32>(28);
  uint32 end_q = get_arg_val<uint32>(29);
  uint32 x_pos = get_arg_val<uint32>(30);
  uint32 x_stride = get_arg_val<uint32>(31);
  uint32 zero_size = get_arg_val<uint32>(32);
  kernel(gx, gb, gzero, lx, lzero, px, pb, N, H, K, R, WC, PQ, SC, RSC_rnd,
         before_h, after_h, before_wc, after_wc, offset_wc, before_hwc, delta_p,
         delta_q, delta_r, end_q, x_pos, x_stride, zero_size);
}

