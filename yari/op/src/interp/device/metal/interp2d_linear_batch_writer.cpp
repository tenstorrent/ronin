// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gy, Pipe py, uint32 N, uint32 C, uint32 PQ, uint32 y_pos,
            uint32 y_stride) {
  py.frame_size = (C / 32);
  uint32 y_start = y_pos;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      cb_wait_front(py.cb_id, py.frame_size);
      noc_async_write_global_dram(get_read_ptr(py.cb_id) + (0 << 1), gy.addr,
                                  gy.log2_page_size, y_start << 1,
                                  (C * 32) << 1);
      noc_async_write_barrier();
      cb_pop_front(py.cb_id, py.frame_size);
      y_start += C * 32;
    } // pq_start
  }   // n
}

void kernel_main() {
  Global gy;
  gy.addr = get_arg_val<uint32>(0);
  gy.log2_page_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  uint32 N = get_arg_val<uint32>(4);
  uint32 C = get_arg_val<uint32>(5);
  uint32 PQ = get_arg_val<uint32>(6);
  uint32 y_pos = get_arg_val<uint32>(7);
  uint32 y_stride = get_arg_val<uint32>(8);
  kernel(gy, py, N, C, PQ, y_pos, y_stride);
}

