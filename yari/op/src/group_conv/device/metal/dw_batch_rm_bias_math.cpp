// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

void kernel(Pipe px, Pipe pw, Pipe pb, Pipe py, Pipe pt_im, Pipe py_im,
            uint32 N, uint32 K, uint32 Ko, uint32 Ki, uint32 PQ, uint32 RS,
            uint32 RSK) {
  uint32 Kt = K / 32;
  px.frame_size = Kt;
  pw.frame_size = (RSK / 32);
  pb.frame_size = Kt;
  py.frame_size = Ki;
  pt_im.frame_size = Ki;
  py_im.frame_size = Ki;
  cb_wait_front(pw.cb_id, pw.frame_size);
  cb_wait_front(pb.cb_id, pb.frame_size);
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      uint32 iw = 0;
      for (uint32 rs = 0; rs < RS; rs++) {
        cb_wait_front(px.cb_id, px.frame_size);
        uint32 kx = 0;
        for (uint32 ko = 0; ko < Ko; ko++) {
          if (rs == 0) {
            tanto_unpack_binary_init(px.cb_id, pw.cb_id);
            tanto_mul_init();
            tanto_pack_init(py_im.cb_id);
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32 ki = 0; ki < Ki; ki++) {
              mul_tiles(px.cb_id, pw.cb_id, kx, iw, ki);
              kx++;
              iw++;
            }
            cb_reserve_back(py_im.cb_id, py_im.frame_size);
            for (uint32 ki = 0; ki < Ki; ki++) {
              pack_tile(ki, py_im.cb_id);
            }
            cb_push_back(py_im.cb_id, py_im.frame_size);
            tile_regs_commit();
            tile_regs_release();
          } else {
            {
              tanto_unpack_binary_init(px.cb_id, pw.cb_id);
              tanto_mul_init();
              tanto_pack_init(pt_im.cb_id);
              tile_regs_acquire();
              tile_regs_wait();
              for (uint32 ki = 0; ki < Ki; ki++) {
                mul_tiles(px.cb_id, pw.cb_id, kx, iw, ki);
                kx++;
                iw++;
              }
              cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
              for (uint32 ki = 0; ki < Ki; ki++) {
                pack_tile(ki, pt_im.cb_id);
              }
              cb_push_back(pt_im.cb_id, pt_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
            {
              tanto_unpack_binary_init(py_im.cb_id, pt_im.cb_id);
              tanto_add_init();
              tanto_pack_init(py_im.cb_id);
              tile_regs_acquire();
              tile_regs_wait();
              cb_wait_front(pt_im.cb_id, pt_im.frame_size);
              cb_wait_front(py_im.cb_id, py_im.frame_size);
              for (uint32 ki = 0; ki < Ki; ki++) {
                add_tiles(py_im.cb_id, pt_im.cb_id, ki, ki, ki);
              }
              cb_pop_front(py_im.cb_id, py_im.frame_size);
              cb_pop_front(pt_im.cb_id, pt_im.frame_size);
              cb_reserve_back(py_im.cb_id, py_im.frame_size);
              for (uint32 ki = 0; ki < Ki; ki++) {
                pack_tile(ki, py_im.cb_id);
              }
              cb_push_back(py_im.cb_id, py_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
          }
        } // ko
        cb_pop_front(px.cb_id, px.frame_size);
      } // rs
      uint32 kb = 0;
      tanto_unpack_binary_init(py_im.cb_id, pb.cb_id);
      tanto_add_init();
      tanto_pack_init(py.cb_id);
      for (uint32 ko = 0; ko < Ko; ko++) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          add_tiles(py_im.cb_id, pb.cb_id, ki, kb, ki);
          kb++;
        }
        cb_pop_front(py_im.cb_id, py_im.frame_size);
        cb_reserve_back(py.cb_id, py.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          pack_tile(ki, py.cb_id);
        }
        cb_push_back(py.cb_id, py.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // ko
    }   // pq_start
  }     // n
  cb_pop_front(pb.cb_id, pb.frame_size);
  cb_pop_front(pw.cb_id, pw.frame_size);
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(2);
  pw.frame_size = get_arg_val<uint32>(3);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(4);
  pb.frame_size = get_arg_val<uint32>(5);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(6);
  py.frame_size = get_arg_val<uint32>(7);
  Pipe pt_im;
  pt_im.cb_id = get_arg_val<uint32>(8);
  pt_im.frame_size = get_arg_val<uint32>(9);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(10);
  py_im.frame_size = get_arg_val<uint32>(11);
  uint32 N = get_arg_val<uint32>(12);
  uint32 K = get_arg_val<uint32>(13);
  uint32 Ko = get_arg_val<uint32>(14);
  uint32 Ki = get_arg_val<uint32>(15);
  uint32 PQ = get_arg_val<uint32>(16);
  uint32 RS = get_arg_val<uint32>(17);
  uint32 RSK = get_arg_val<uint32>(18);
  tanto_compute_init();
  kernel(px, pw, pb, py, pt_im, py_im, N, K, Ko, Ki, PQ, RS, RSK);
}
} // namespace NAMESPACE

