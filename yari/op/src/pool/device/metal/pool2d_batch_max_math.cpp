// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

void kernel(Pipe px, Pipe py, Pipe py_im, uint32 N, uint32 C, uint32 PQ,
            uint32 RS) {
  tanto_copy_init();
  tanto_max_init();
  uint32 Ct = C / 32;
  px.frame_size = Ct;
  py.frame_size = Ct;
  py_im.frame_size = Ct;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      cb_reserve_back(py_im.cb_id, py_im.frame_size);
      cb_wait_front(px.cb_id, px.frame_size);
      tanto_unpack_unary_init(px.cb_id);
      tanto_pack_init(py_im.cb_id);
      for (uint32 c = 0; c < Ct; c++) {
        tile_regs_acquire();
        tile_regs_wait();
        copy_tile(px.cb_id, c, 0);
        pack_tile(0, py_im.cb_id);
        tile_regs_commit();
        tile_regs_release();
      } // c
      cb_pop_front(px.cb_id, px.frame_size);
      cb_push_back(py_im.cb_id, py_im.frame_size);
      tanto_pack_init(py_im.cb_id);
      for (uint32 rs = 1; rs < RS - 1; rs++) {
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        cb_wait_front(px.cb_id, px.frame_size);
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 c = 0; c < Ct; c++) {
          tile_regs_acquire();
          tile_regs_wait();
          tanto_unpack_unary_init(px.cb_id);
          copy_tile(px.cb_id, c, 0);
          tanto_unpack_unary_init(py_im.cb_id);
          copy_tile(py_im.cb_id, c, 1);
          tanto_max_tile(0);
          pack_tile(0, py_im.cb_id);
          tile_regs_commit();
          tile_regs_release();
        }
        cb_pop_front(px.cb_id, px.frame_size);
        cb_pop_front(py_im.cb_id, py_im.frame_size);
        cb_push_back(py_im.cb_id, py_im.frame_size);
      } // rs
      cb_reserve_back(py.cb_id, py.frame_size);
      cb_wait_front(px.cb_id, px.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_pack_init(py.cb_id);
      for (uint32 c = 0; c < Ct; c++) {
        tile_regs_acquire();
        tile_regs_wait();
        tanto_unpack_unary_init(px.cb_id);
        copy_tile(px.cb_id, c, 0);
        tanto_unpack_unary_init(py_im.cb_id);
        copy_tile(py_im.cb_id, c, 1);
        tanto_max_tile(0);
        pack_tile(0, py.cb_id);
        tile_regs_commit();
        tile_regs_release();
      }
      cb_pop_front(px.cb_id, px.frame_size);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(py.cb_id, py.frame_size);
    } // pq_start
  }   // n
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(4);
  py_im.frame_size = get_arg_val<uint32>(5);
  uint32 N = get_arg_val<uint32>(6);
  uint32 C = get_arg_val<uint32>(7);
  uint32 PQ = get_arg_val<uint32>(8);
  uint32 RS = get_arg_val<uint32>(9);
  tanto_compute_init();
  kernel(px, py, py_im, N, C, PQ, RS);
}
} // namespace NAMESPACE

