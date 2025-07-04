// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void make_pc1_im(Pipe ppc_im, Pipe pc1_im, uint32 rs2) {
  cb_reserve_back(pc1_im.cb_id, pc1_im.frame_size);
  // lh and lw occupy consecutive rows
  // hh and hw occupy consecutive rows
  // (lh, lw) and (hh, hw) occupy consecutive tiles
  uint32 p = rs2 / 32;
  uint32 q = rs2 % 32;
  uint32 lh = (p * 64 + q) * 32;
  uint32 lw = lh + 32;
  uint32 hh = lh + 1024;
  uint32 hw = hh + 32;
  // assemble first row of untilized [32, 32 * 4] frame in pc1_im
  // other rows are not significant
  // order is [hh, lh, hw, lw]
  noc_async_read_one_packet_set_state(get_noc_addr(0), 32 << 1);
  noc_async_read_one_packet_with_state(get_read_ptr(ppc_im.cb_id) + (hh << 1),
                                       get_write_ptr(pc1_im.cb_id) + (0 << 1));
  noc_async_read_one_packet_with_state(get_read_ptr(ppc_im.cb_id) + (lh << 1),
                                       get_write_ptr(pc1_im.cb_id) + (32 << 1));
  noc_async_read_one_packet_with_state(get_read_ptr(ppc_im.cb_id) + (hw << 1),
                                       get_write_ptr(pc1_im.cb_id) +
                                           ((32 * 2) << 1));
  noc_async_read_one_packet_with_state(get_read_ptr(ppc_im.cb_id) + (lw << 1),
                                       get_write_ptr(pc1_im.cb_id) +
                                           ((32 * 3) << 1));
  noc_async_read_barrier();
  cb_push_back(pc1_im.cb_id, pc1_im.frame_size);
}

void read_px(Global gx, Local lzero, Pipe px, Pipe ppi_im, uint32 C, uint32 H,
             uint32 W, uint32 D, uint32 rs2, uint32 th, uint32 tw,
             uint32 start_q, uint32 delta_p, uint32 delta_q, uint32 end_q,
             uint32 x_start, uint32 &p_term, uint32 &q_term, uint32 r_term,
             uint32 s_term) {
  uint32 dst_pos = 0;
  cb_reserve_back(px.cb_id, px.frame_size);
  for (uint32 i = 0; i < 32; i++) {
    uint32 ih = uint32(reinterpret_cast<volatile tt_l1_ptr uint16_t *>(
        get_read_ptr(ppi_im.cb_id))[rs2]);
    uint32 iw = uint32(reinterpret_cast<volatile tt_l1_ptr uint16_t *>(
        get_read_ptr(ppi_im.cb_id))[(rs2 + 1)]);
#if 1
    // enable for Metal / disable for Jitte (temporary patch)
    ih = ((0x80 | (ih & 0x7f)) << ((ih >> 7) - 127)) >> 7;
    iw = ((0x80 | (iw & 0x7f)) << ((iw >> 7) - 127)) >> 7;
#endif
    ih -= 63;
    iw -= 63;
    uint32 h = p_term + r_term + ih + th;
    uint32 w = q_term + s_term + iw + tw;
    if (h >= H || w >= W) {
      noc_async_read(get_noc_addr(lzero.addr + (0 << 1)),
                     get_write_ptr(px.cb_id) + (dst_pos << 1), C << 1);
    } else {
      // ACHTUNG: int multiplications here
      uint32 src_pos = x_start + (h * W + w) * C;
      noc_async_read_global_dram(get_write_ptr(px.cb_id) + (dst_pos << 1),
                                 gx.addr, gx.log2_page_size, src_pos << 1,
                                 C << 1);
    }
    q_term += delta_q;
    if (q_term >= end_q) {
      q_term = start_q;
      p_term += delta_p;
    }
    dst_pos += C;
    rs2 += D;
  }
  noc_async_read_barrier();
  cb_push_back(px.cb_id, px.frame_size);
}

void kernel(Global gx, Global gd, Global gb, Global gzero, Local lzero, Pipe px,
            Pipe pd, Pipe pb, Pipe ppi_im, Pipe ppc_im, Pipe pc1_im, uint32 N,
            uint32 H, uint32 W, uint32 C, uint32 K, uint32 R, uint32 S,
            uint32 D, uint32 PQ, uint32 start_p, uint32 start_q, uint32 delta_p,
            uint32 delta_q, uint32 delta_r, uint32 delta_s, uint32 end_q,
            uint32 zero_size, uint32 x_pos, uint32 x_stride, uint32 d_pos,
            uint32 d_stride) {
  noc_async_read_global_dram(lzero.addr + (0 << 1), gzero.addr,
                             gzero.log2_page_size, 0 << 1, zero_size << 1);
  // read_barrier is below
  px.frame_size = (C / 32);
  pd.frame_size = (D / 32);
  pb.frame_size = (K / 32);
  ppi_im.frame_size = (D / 32);
  ppc_im.frame_size = ((D / 32) * 2);
  pc1_im.frame_size = 4;
  cb_reserve_back(pb.cb_id, pb.frame_size);
  noc_async_read_global_dram(get_write_ptr(pb.cb_id) + (0 << 1), gb.addr,
                             gb.log2_page_size, 0 << 1, (K * 32) << 1);
  noc_async_read_barrier();
  cb_push_back(pb.cb_id, pb.frame_size);
  uint32 x_start = x_pos;
  uint32 d_start = d_pos;
  for (uint32 n = 0; n < N; n++) {
    uint32 d_curr = d_start;
    uint32 p_start = start_p;
    uint32 q_start = start_q;
    uint32 p_term = 0;
    uint32 q_term = 0;
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      cb_reserve_back(pd.cb_id, pd.frame_size);
      noc_async_read_global_dram(get_write_ptr(pd.cb_id) + (0 << 1), gd.addr,
                                 gd.log2_page_size, d_curr << 1, (D * 32) << 1);
      noc_async_read_barrier();
      cb_push_back(pd.cb_id, pd.frame_size);
      // math kernel transforms pd -> (ppi_im, ppc_im)
      cb_wait_front(ppi_im.cb_id, ppi_im.frame_size);
      cb_wait_front(ppc_im.cb_id, ppc_im.frame_size);
      uint32 rs2 = 0;
      uint32 r_term = 0;
      for (uint32 r = 0; r < R; r++) {
        uint32 s_term = 0;
        for (uint32 s = 0; s < S; s++) {
          make_pc1_im(ppc_im, pc1_im, rs2);
          for (uint32 i = 0; i < 4; i++) {
            p_term = p_start;
            q_term = q_start;
            uint32 th = i >> 1;
            uint32 tw = i & 1;
            read_px(gx, lzero, px, ppi_im, C, H, W, D, rs2, th, tw, start_q,
                    delta_p, delta_q, end_q, x_start, p_term, q_term, r_term,
                    s_term);
          } // i
          s_term += delta_s;
          rs2 += 2;
        } // s
        r_term += delta_r;
      } // r
      cb_pop_front(ppi_im.cb_id, ppi_im.frame_size);
      cb_pop_front(ppc_im.cb_id, ppc_im.frame_size);
      p_start = p_term;
      q_start = q_term;
      d_curr += D * 32;
    } // pq_start
    x_start += x_stride;
    d_start += d_stride;
  } // n
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Global gd;
  gd.addr = get_arg_val<uint32>(2);
  gd.log2_page_size = get_arg_val<uint32>(3);
  Global gb;
  gb.addr = get_arg_val<uint32>(4);
  gb.log2_page_size = get_arg_val<uint32>(5);
  Global gzero;
  gzero.addr = get_arg_val<uint32>(6);
  gzero.log2_page_size = get_arg_val<uint32>(7);
  Local lzero;
  lzero.addr = get_arg_val<uint32>(8);
  Pipe px;
  px.cb_id = get_arg_val<uint32>(9);
  px.frame_size = get_arg_val<uint32>(10);
  Pipe pd;
  pd.cb_id = get_arg_val<uint32>(11);
  pd.frame_size = get_arg_val<uint32>(12);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(13);
  pb.frame_size = get_arg_val<uint32>(14);
  Pipe ppi_im;
  ppi_im.cb_id = get_arg_val<uint32>(15);
  ppi_im.frame_size = get_arg_val<uint32>(16);
  Pipe ppc_im;
  ppc_im.cb_id = get_arg_val<uint32>(17);
  ppc_im.frame_size = get_arg_val<uint32>(18);
  Pipe pc1_im;
  pc1_im.cb_id = get_arg_val<uint32>(19);
  pc1_im.frame_size = get_arg_val<uint32>(20);
  uint32 N = get_arg_val<uint32>(21);
  uint32 H = get_arg_val<uint32>(22);
  uint32 W = get_arg_val<uint32>(23);
  uint32 C = get_arg_val<uint32>(24);
  uint32 K = get_arg_val<uint32>(25);
  uint32 R = get_arg_val<uint32>(26);
  uint32 S = get_arg_val<uint32>(27);
  uint32 D = get_arg_val<uint32>(28);
  uint32 PQ = get_arg_val<uint32>(29);
  uint32 start_p = get_arg_val<uint32>(30);
  uint32 start_q = get_arg_val<uint32>(31);
  uint32 delta_p = get_arg_val<uint32>(32);
  uint32 delta_q = get_arg_val<uint32>(33);
  uint32 delta_r = get_arg_val<uint32>(34);
  uint32 delta_s = get_arg_val<uint32>(35);
  uint32 end_q = get_arg_val<uint32>(36);
  uint32 zero_size = get_arg_val<uint32>(37);
  uint32 x_pos = get_arg_val<uint32>(38);
  uint32 x_stride = get_arg_val<uint32>(39);
  uint32 d_pos = get_arg_val<uint32>(40);
  uint32 d_stride = get_arg_val<uint32>(41);
  kernel(gx, gd, gb, gzero, lzero, px, pd, pb, ppi_im, ppc_im, pc1_im, N, H, W,
         C, K, R, S, D, PQ, start_p, start_q, delta_p, delta_q, delta_r,
         delta_s, end_q, zero_size, x_pos, x_stride, d_pos, d_stride);
}

