// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

void kernel(Pipe px, Pipe pw, Pipe py, Pipe px_im, Pipe pw_im, Pipe pt_im,
            Pipe py_im, uint32 N, uint32 C, uint32 Co, uint32 Ci, uint32 PQ) {
  uint32 Ct = C / 32;
  px.frame_size = Ct;
  pw.frame_size = 4;
  py.frame_size = Ct;
  px_im.frame_size = Ct;
  pw_im.frame_size = 4;
  pt_im.frame_size = Ci;
  py_im.frame_size = Ci;
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      // pw_im = tilize(pw)
      cb_reserve_back(pw_im.cb_id, pw_im.frame_size);
      cb_wait_front(pw.cb_id, pw.frame_size);
      tanto_unpack_tilize_block_init(pw.cb_id, 4);
      tanto_copy_init();
      tanto_pack_init(pw_im.cb_id);
      tilize_block(pw.cb_id, 4, pw_im.cb_id);
      cb_pop_front(pw.cb_id, pw.frame_size);
      cb_push_back(pw_im.cb_id, pw_im.frame_size);
      // pw_im = transpose(pw_im)
      {
        tanto_unpack_transpose_init(pw_im.cb_id);
        tanto_transpose_init();
        tanto_pack_init(pw_im.cb_id);
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(pw_im.cb_id, pw_im.frame_size);
        for (uint32 i = 0; i < 4; i++) {
          transpose_wh_tile(pw_im.cb_id, i, i);
        }
        cb_pop_front(pw_im.cb_id, pw_im.frame_size);
        cb_reserve_back(pw_im.cb_id, pw_im.frame_size);
        for (uint32 i = 0; i < 4; i++) {
          pack_tile(i, pw_im.cb_id);
        }
        cb_push_back(pw_im.cb_id, pw_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // acc
      cb_wait_front(pw_im.cb_id, pw_im.frame_size);
      for (uint32 i = 0; i < 4; i++) {
        // px_im = tilize(px)
        cb_reserve_back(px_im.cb_id, px_im.frame_size);
        cb_wait_front(px.cb_id, px.frame_size);
        tanto_unpack_tilize_block_init(px.cb_id, Ct);
        tanto_copy_init();
        tanto_pack_init(px_im.cb_id);
        tilize_block(px.cb_id, Ct, px_im.cb_id);
        cb_pop_front(px.cb_id, px.frame_size);
        cb_push_back(px_im.cb_id, px_im.frame_size);
        cb_wait_front(px_im.cb_id, px_im.frame_size);
        uint32 cx = 0;
        for (uint32 co = 0; co < Co; co++) {
          if (i == 0) {
            tanto_unpack_bcast_cols_init(px_im.cb_id, pw_im.cb_id);
            tanto_mul_bcast_cols_init();
            tanto_pack_init(py_im.cb_id);
            // py_im = px_im * pw_im
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32 ci = 0; ci < Ci; ci++) {
              any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
                  px_im.cb_id, pw_im.cb_id, cx, i, ci);
              cx++;
            }
            cb_reserve_back(py_im.cb_id, py_im.frame_size);
            for (uint32 ci = 0; ci < Ci; ci++) {
              pack_tile(ci, py_im.cb_id);
            }
            cb_push_back(py_im.cb_id, py_im.frame_size);
            tile_regs_commit();
            tile_regs_release();
          } else {
            {
              tanto_unpack_bcast_cols_init(px_im.cb_id, pw_im.cb_id);
              tanto_mul_bcast_cols_init();
              tanto_pack_init(pt_im.cb_id);
              // pt_im = px_im * pw_im
              tile_regs_acquire();
              tile_regs_wait();
              for (uint32 ci = 0; ci < Ci; ci++) {
                any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
                    px_im.cb_id, pw_im.cb_id, cx, i, ci);
                cx++;
              }
              cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                pack_tile(ci, pt_im.cb_id);
              }
              cb_push_back(pt_im.cb_id, pt_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
            {
              tanto_unpack_binary_init(py_im.cb_id, pt_im.cb_id);
              tanto_add_init();
              tanto_pack_init(py_im.cb_id);
              // py_im = py_im + pt_im
              tile_regs_acquire();
              tile_regs_wait();
              cb_wait_front(pt_im.cb_id, pt_im.frame_size);
              cb_wait_front(py_im.cb_id, py_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                add_tiles(py_im.cb_id, pt_im.cb_id, ci, ci, ci);
              }
              cb_pop_front(py_im.cb_id, py_im.frame_size);
              cb_pop_front(pt_im.cb_id, pt_im.frame_size);
              cb_reserve_back(py_im.cb_id, py_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                pack_tile(ci, py_im.cb_id);
              }
              cb_push_back(py_im.cb_id, py_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
          }
        } // co
        cb_pop_front(px_im.cb_id, px_im.frame_size);
      } // i
      cb_pop_front(pw_im.cb_id, pw_im.frame_size);
      // py = untilize(py_im)
      py_im.frame_size = Ct;
      cb_reserve_back(py.cb_id, py.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_unpack_untilize_block_init(py_im.cb_id);
      tanto_copy_init();
      tanto_pack_init(py.cb_id);
      untilize_block<1>(py_im.cb_id, Ct, py.cb_id);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(py.cb_id, py.frame_size);
      py_im.frame_size = Ci;
    } // pq_start
  }   // n
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(2);
  pw.frame_size = get_arg_val<uint32>(3);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(4);
  py.frame_size = get_arg_val<uint32>(5);
  Pipe px_im;
  px_im.cb_id = get_arg_val<uint32>(6);
  px_im.frame_size = get_arg_val<uint32>(7);
  Pipe pw_im;
  pw_im.cb_id = get_arg_val<uint32>(8);
  pw_im.frame_size = get_arg_val<uint32>(9);
  Pipe pt_im;
  pt_im.cb_id = get_arg_val<uint32>(10);
  pt_im.frame_size = get_arg_val<uint32>(11);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(12);
  py_im.frame_size = get_arg_val<uint32>(13);
  uint32 N = get_arg_val<uint32>(14);
  uint32 C = get_arg_val<uint32>(15);
  uint32 Co = get_arg_val<uint32>(16);
  uint32 Ci = get_arg_val<uint32>(17);
  uint32 PQ = get_arg_val<uint32>(18);
  tanto_compute_init();
  kernel(px, pw, py, px_im, pw_im, pt_im, py_im, N, C, Co, Ci, PQ);
}
} // namespace NAMESPACE

