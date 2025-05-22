// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gw, Global gw2, Global gy, Pipe pw, Pipe pw2, Pipe py,
            uint32 N, uint32 C, uint32 K, uint32 KC, uint32 PQ, uint32 RSC,
            uint32 y_pos, uint32 y_stride) {
  pw.frame_size = (RSC / 32);
  pw2.frame_size = (KC / 1024);
  py.frame_size = (K / 32);
  cb_reserve_back(pw.cb_id, pw.frame_size);
  cb_reserve_back(pw2.cb_id, pw2.frame_size);
  noc_async_read_global_dram(get_write_ptr(pw.cb_id) + (0 << 1), gw.addr,
                             gw.log2_page_size, 0 << 1, (RSC * 32) << 1);
  noc_async_read_global_dram(get_write_ptr(pw2.cb_id) + (0 << 1), gw2.addr,
                             gw2.log2_page_size, 0 << 1, KC << 1);
  noc_async_read_barrier();
  cb_push_back(pw.cb_id, pw.frame_size);
  cb_push_back(pw2.cb_id, pw2.frame_size);
  uint32 y_start = y_pos;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      cb_wait_front(py.cb_id, py.frame_size);
      noc_async_write_global_dram(get_read_ptr(py.cb_id) + (0 << 1), gy.addr,
                                  gy.log2_page_size, y_start << 1,
                                  (K * 32) << 1);
      noc_async_write_barrier();
      cb_pop_front(py.cb_id, py.frame_size);
      y_start += K * 32;
    } // pq_start
    y_start += y_stride;
  } // n
}

void kernel_main() {
  Global gw;
  gw.addr = get_arg_val<uint32>(0);
  gw.log2_page_size = get_arg_val<uint32>(1);
  Global gw2;
  gw2.addr = get_arg_val<uint32>(2);
  gw2.log2_page_size = get_arg_val<uint32>(3);
  Global gy;
  gy.addr = get_arg_val<uint32>(4);
  gy.log2_page_size = get_arg_val<uint32>(5);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(6);
  pw.frame_size = get_arg_val<uint32>(7);
  Pipe pw2;
  pw2.cb_id = get_arg_val<uint32>(8);
  pw2.frame_size = get_arg_val<uint32>(9);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(10);
  py.frame_size = get_arg_val<uint32>(11);
  uint32 N = get_arg_val<uint32>(12);
  uint32 C = get_arg_val<uint32>(13);
  uint32 K = get_arg_val<uint32>(14);
  uint32 KC = get_arg_val<uint32>(15);
  uint32 PQ = get_arg_val<uint32>(16);
  uint32 RSC = get_arg_val<uint32>(17);
  uint32 y_pos = get_arg_val<uint32>(18);
  uint32 y_stride = get_arg_val<uint32>(19);
  kernel(gw, gw2, gy, pw, pw2, py, N, C, K, KC, PQ, RSC, y_pos, y_stride);
}

