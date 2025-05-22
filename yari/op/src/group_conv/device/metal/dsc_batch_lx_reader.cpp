// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void read_px(Local lx, Local lzero, Pipe px, uint32 C, uint32 start_q,
             uint32 delta_p, uint32 delta_q, uint32 end_q, uint32 &p_term,
             uint32 &q_term, uint32 r_term, uint32 s_term, uint32 mask) {
  uint32 dst_pos = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  noc_async_read_one_packet_set_state(get_noc_addr(0), C << 1);
  for (uint32 i = 0; i < 32; i++) {
    if ((mask & 1) == 0) {
      noc_async_read_one_packet_with_state(
          lzero.addr + (0 << 1), get_write_ptr(px.cb_id) + (dst_pos << 1));
    } else {
      uint32 src_pos = p_term + q_term + r_term + s_term;
      noc_async_read_one_packet_with_state(
          lx.addr + (src_pos << 1), get_write_ptr(px.cb_id) + (dst_pos << 1));
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

void kernel(Global gx, Global gb, Global gb2, Global gzero, Global gmask,
            Local lx, Local lzero, Local lmask, Pipe px, Pipe pb, Pipe pb2,
            uint32 N, uint32 C, uint32 K, uint32 R, uint32 S, uint32 HWC,
            uint32 PQ, uint32 start_p, uint32 start_q, uint32 delta_p,
            uint32 delta_q, uint32 delta_r, uint32 delta_s, uint32 end_q,
            uint32 zero_size, uint32 mask_size, uint32 x_pos, uint32 x_stride) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  noc_async_read_global_dram(lmask.addr + (0 << 2), gmask.addr,
                             gmask.log2_page_size, 0 << 2, mask_size << 2);
  // read_barrier is below
  px.frame_size = (C / 32);
  pb.frame_size = (C / 32);
  pb2.frame_size = (K / 32);
  cb_reserve_back(pb.cb_id, pb.frame_size);
  cb_reserve_back(pb2.cb_id, pb2.frame_size);
  noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                             gb.log2_page_size, 0 << 1, (C * 32) << 1);
  noc_async_read_global_dram(get_write_ptr(pb2.cb_id) + (0 << 1), gb2.addr,
                             gb2.log2_page_size, 0 << 1, (K * 32) << 1);
  noc_async_read_barrier();
  cb_push_back(pb.cb_id, pb.frame_size);
  cb_push_back(pb2.cb_id, pb2.frame_size);
  uint32 x_start = x_pos;
  for (uint32 n = 0; n < N; n++) {
    noc_async_read_global_dram(lx.addr + (0 << 1), gx.addr, gx.log2_page_size,
                               x_start << 1, HWC << 1);
    noc_async_read_barrier();
    uint32 p_start = start_p;
    uint32 q_start = start_q;
    uint32 p_term = 0;
    uint32 q_term = 0;
    uint32 mask_pos = 0;
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      uint32 r_term = 0;
      for (uint32 r = 0; r < R; r++) {
        uint32 s_term = 0;
        for (uint32 s = 0; s < S; s++) {
          p_term = p_start;
          q_term = q_start;
          uint32 mask = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              lmask.addr)[mask_pos];
          read_px(lx, lzero, px, C, start_q, delta_p, delta_q, end_q, p_term,
                  q_term, r_term, s_term, mask);
          mask_pos++;
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
  Global gb2;
  gb2.addr = get_arg_val<uint32>(4);
  gb2.log2_page_size = get_arg_val<uint32>(5);
  Global gzero;
  gzero.addr = get_arg_val<uint32>(6);
  gzero.log2_page_size = get_arg_val<uint32>(7);
  Global gmask;
  gmask.addr = get_arg_val<uint32>(8);
  gmask.log2_page_size = get_arg_val<uint32>(9);
  Local lx;
  lx.addr = get_arg_val<uint32>(10);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(11);
  Local lmask;
  lmask.addr = get_arg_val<uint32>(12);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(13);
  px.frame_size = get_arg_val<uint32>(14);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(15);
  pb.frame_size = get_arg_val<uint32>(16);
  Pipe pb2;
  pb2.cb_id = get_arg_val<uint32>(17);
  pb2.frame_size = get_arg_val<uint32>(18);
  uint32 N = get_arg_val<uint32>(19);
  uint32 C = get_arg_val<uint32>(20);
  uint32 K = get_arg_val<uint32>(21);
  uint32 R = get_arg_val<uint32>(22);
  uint32 S = get_arg_val<uint32>(23);
  uint32 HWC = get_arg_val<uint32>(24);
  uint32 PQ = get_arg_val<uint32>(25);
  uint32 start_p = get_arg_val<uint32>(26);
  uint32 start_q = get_arg_val<uint32>(27);
  uint32 delta_p = get_arg_val<uint32>(28);
  uint32 delta_q = get_arg_val<uint32>(29);
  uint32 delta_r = get_arg_val<uint32>(30);
  uint32 delta_s = get_arg_val<uint32>(31);
  uint32 end_q = get_arg_val<uint32>(32);
  uint32 zero_size = get_arg_val<uint32>(33);
  uint32 mask_size = get_arg_val<uint32>(34);
  uint32 x_pos = get_arg_val<uint32>(35);
  uint32 x_stride = get_arg_val<uint32>(36);
  kernel(gx, gb, gb2, gzero, gmask, lx, lzero, lmask, px, pb, pb2, N, C, K, R,
         S, HWC, PQ, start_p, start_q, delta_p, delta_q, delta_r, delta_s,
         end_q, zero_size, mask_size, x_pos, x_stride);
}

