// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gw, Global gy, Pipe pw, Pipe py, Semaphore sem_send,
            Semaphore sem_recv, uint32 send_mode, uint32 x0, uint32 y0,
            uint32 x1, uint32 y1, uint32 num_dests, uint32 N, uint32 K,
            uint32 PQ, uint32 KRSC_rnd, uint32 y_pos, uint32 y_stride) {
  pw.frame_size = (KRSC_rnd / 1024);
  py.frame_size = (K / 32);
  if (send_mode != 0) {
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_recv.addr), 1);
    cb_reserve_back(pw.cb_id, pw.frame_size);
    noc_async_read_global_dram(get_write_ptr(pw.cb_id) + (0 << 1), gw.addr,
                               gw.log2_page_size, 0 << 1, KRSC_rnd << 1);
    noc_async_read_barrier();
    noc_semaphore_wait(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_send.addr),
        num_dests);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_send.addr), 0);
    noc_async_write_multicast(
        get_write_ptr(pw.cb_id) + (0 << 1),
        get_noc_multicast_addr(x0, y0, x1, y1,
                               get_write_ptr(pw.cb_id) + (0 << 1)),
        KRSC_rnd << 1, num_dests);
    noc_semaphore_set_multicast(
        sem_recv.addr, get_noc_multicast_addr(x0, y0, x1, y1, sem_recv.addr),
        num_dests);
    cb_push_back(pw.cb_id, pw.frame_size);
  } else {
    cb_reserve_back(pw.cb_id, pw.frame_size);
    noc_semaphore_set(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_recv.addr), 0);
    noc_semaphore_inc(get_noc_addr(x0, y0, sem_send.addr), 1);
    noc_semaphore_wait(
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sem_recv.addr), 1);
    cb_push_back(pw.cb_id, pw.frame_size);
  }
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
  Global gy;
  gy.addr = get_arg_val<uint32>(2);
  gy.log2_page_size = get_arg_val<uint32>(3);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(4);
  pw.frame_size = get_arg_val<uint32>(5);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(6);
  py.frame_size = get_arg_val<uint32>(7);
  Semaphore sem_send;
  sem_send.addr = tanto_get_semaphore(get_arg_val<uint32>(8));
  Semaphore sem_recv;
  sem_recv.addr = tanto_get_semaphore(get_arg_val<uint32>(9));
  uint32 send_mode = get_arg_val<uint32>(10);
  uint32 x0 = get_arg_val<uint32>(11);
  uint32 y0 = get_arg_val<uint32>(12);
  uint32 x1 = get_arg_val<uint32>(13);
  uint32 y1 = get_arg_val<uint32>(14);
  uint32 num_dests = get_arg_val<uint32>(15);
  uint32 N = get_arg_val<uint32>(16);
  uint32 K = get_arg_val<uint32>(17);
  uint32 PQ = get_arg_val<uint32>(18);
  uint32 KRSC_rnd = get_arg_val<uint32>(19);
  uint32 y_pos = get_arg_val<uint32>(20);
  uint32 y_stride = get_arg_val<uint32>(21);
  kernel(gw, gy, pw, py, sem_send, sem_recv, send_mode, x0, y0, x1, y1,
         num_dests, N, K, PQ, KRSC_rnd, y_pos, y_stride);
}

