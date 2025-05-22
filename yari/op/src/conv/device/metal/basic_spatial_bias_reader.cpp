// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void read_px(Global gx, Local lzero, Pipe px, uint32 C, uint32 start_q,
             uint32 delta_p, uint32 delta_q, uint32 end_q, uint32 start,
             uint32 &p_term, uint32 &q_term, uint32 r_term, uint32 s_term,
             uint32 mask) {
  uint32 dst_pos = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  for (uint32 i = 0; i < 32; i++) {
    if ((mask & 1) == 0) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)),
                     get_write_ptr(px.cb_id) + (dst_pos << 1), C << 1);
    } else {
      uint32 src_pos = start + p_term + q_term + r_term + s_term;
      noc_async_read_global_dram(get_write_ptr(px.cb_id) + (dst_pos << 1),
                                 gx.addr, gx.log2_page_size, src_pos << 1,
                                 C << 1);
    }
    q_term += delta_q;
    if (q_term >= end_q) {
      q_term = start_q;
      p_term += delta_p;
    }
    mask >>= 1;
    dst_pos += C;
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gb, Global gzero, Global gmask, Local lzero,
            Local lmask, Pipe px, Pipe pb, uint32 N, uint32 C, uint32 K,
            uint32 R, uint32 S, uint32 PQ, uint32 start_p, uint32 start_q,
            uint32 delta_p, uint32 delta_q, uint32 delta_r, uint32 delta_s,
            uint32 end_q, uint32 zero_size, uint32 mask_size, uint32 x_pos,
            uint32 x_stride, uint32 mask_pos) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  noc_async_read_global_dram(lmask.addr + (0 << 2), gmask.addr,
                             gmask.log2_page_size, 0 << 2, mask_size << 2);
  // read_barrier is below
  px.frame_size = (C / 32);
  pb.frame_size = (K / 32);
  cb_reserve_back(pb.cb_id, pb.frame_size);
  noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                             gb.log2_page_size, 0 << 1, (K * 32) << 1);
  noc_async_read_barrier();
  cb_push_back(pb.cb_id, pb.frame_size);
  uint32 x_start = x_pos;
  for (uint32 n = 0; n < N; n++) {
    uint32 p_start = start_p;
    uint32 q_start = start_q;
    uint32 p_term = 0;
    uint32 q_term = 0;
    uint32 mask_start = mask_pos;
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      uint32 r_term = 0;
      for (uint32 r = 0; r < R; r++) {
        uint32 s_term = 0;
        for (uint32 s = 0; s < S; s++) {
          p_term = p_start;
          q_term = q_start;
          uint32 mask = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              lmask.addr)[mask_start];
          read_px(gx, lzero, px, C, start_q, delta_p, delta_q, end_q, x_start,
                  p_term, q_term, r_term, s_term, mask);
          mask_start++;
          s_term += delta_s;
        } // s
        r_term += delta_r;
      } // r
      p_start = p_term;
      q_start = q_term;
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
  Global gmask;
  gmask.addr = get_arg_val<uint32>(6);
  gmask.log2_page_size = get_arg_val<uint32>(7);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(8);
  Local lmask;
  lmask.addr = get_arg_val<uint32>(9);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(10);
  px.frame_size = get_arg_val<uint32>(11);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(12);
  pb.frame_size = get_arg_val<uint32>(13);
  uint32 N = get_arg_val<uint32>(14);
  uint32 C = get_arg_val<uint32>(15);
  uint32 K = get_arg_val<uint32>(16);
  uint32 R = get_arg_val<uint32>(17);
  uint32 S = get_arg_val<uint32>(18);
  uint32 PQ = get_arg_val<uint32>(19);
  uint32 start_p = get_arg_val<uint32>(20);
  uint32 start_q = get_arg_val<uint32>(21);
  uint32 delta_p = get_arg_val<uint32>(22);
  uint32 delta_q = get_arg_val<uint32>(23);
  uint32 delta_r = get_arg_val<uint32>(24);
  uint32 delta_s = get_arg_val<uint32>(25);
  uint32 end_q = get_arg_val<uint32>(26);
  uint32 zero_size = get_arg_val<uint32>(27);
  uint32 mask_size = get_arg_val<uint32>(28);
  uint32 x_pos = get_arg_val<uint32>(29);
  uint32 x_stride = get_arg_val<uint32>(30);
  uint32 mask_pos = get_arg_val<uint32>(31);
  kernel(gx, gb, gzero, gmask, lzero, lmask, px, pb, N, C, K, R, S, PQ, start_p,
         start_q, delta_p, delta_q, delta_r, delta_s, end_q, zero_size,
         mask_size, x_pos, x_stride, mask_pos);
}

