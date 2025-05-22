// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gw, Global gz, Global gy, Pipe pw, Pipe pz, Pipe py,
            uint32 N, uint32 C, uint32 K, uint32 PQ, uint32 RS, uint32 y_pos,
            uint32 y_stride) {
  pw.frame_size = (C / 32);
  pz.frame_size = (K / 32);
  py.frame_size = (K / 32);
  uint32 y_start = y_pos;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      uint32 w_start = 0;
      for (uint32 rs = 0; rs < RS; rs++) {
        for (uint32 k_start = 0; k_start < K; k_start += 32) {
          cb_reserve_back(pw.cb_id, pw.frame_size);
          noc_async_read_global_dram(get_write_ptr(pw.cb_id) + (0 << 1),
                                     gw.addr, gw.log2_page_size, w_start << 1,
                                     (C * 32) << 1);
          noc_async_read_barrier();
          cb_push_back(pw.cb_id, pw.frame_size);
          w_start += C * 32;
        } // k_start
      }   // rs
      cb_reserve_back(pz.cb_id, pz.frame_size);
      noc_async_read_global_dram(get_write_ptr(pz.cb_id) + (0 << 1), gz.addr,
                                 gz.log2_page_size, y_start << 1,
                                 (K * 32) << 1);
      noc_async_read_barrier();
      cb_push_back(pz.cb_id, pz.frame_size);
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
  Global gz;
  gz.addr = get_arg_val<uint32>(2);
  gz.log2_page_size = get_arg_val<uint32>(3);
  Global gy;
  gy.addr = get_arg_val<uint32>(4);
  gy.log2_page_size = get_arg_val<uint32>(5);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(6);
  pw.frame_size = get_arg_val<uint32>(7);
  Pipe pz;
  pz.cb_id = get_arg_val<uint32>(8);
  pz.frame_size = get_arg_val<uint32>(9);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(10);
  py.frame_size = get_arg_val<uint32>(11);
  uint32 N = get_arg_val<uint32>(12);
  uint32 C = get_arg_val<uint32>(13);
  uint32 K = get_arg_val<uint32>(14);
  uint32 PQ = get_arg_val<uint32>(15);
  uint32 RS = get_arg_val<uint32>(16);
  uint32 y_pos = get_arg_val<uint32>(17);
  uint32 y_stride = get_arg_val<uint32>(18);
  kernel(gw, gz, gy, pw, pz, py, N, C, K, PQ, RS, y_pos, y_stride);
}

