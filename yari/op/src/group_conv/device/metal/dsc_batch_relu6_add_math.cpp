// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

static constexpr uint32 UNARY_OP_RELU = 0, UNARY_OP_RELU6 = 1;

static constexpr uint32 unary_op_code = uint32(1);

void unary_op(uint32 index, uint32 param0) { relu_max_tile(index, 0x40c00000); }

void matmul_slice(

    Pipe pa, Pipe pb, uint32 pboff, uint32 idst, uint32 tiles) {
  for (uint32 i = 0; i < tiles; i++) {
    matmul_tiles(pa.cb_id, pb.cb_id, i, i + pboff, idst, true);
  }
}

void kernel(Pipe px, Pipe pw, Pipe pb, Pipe pw2, Pipe pb2, Pipe pz, Pipe py,
            Pipe pu_im, Pipe pt_im, Pipe pz_im, Pipe py_im, uint32 N, uint32 C,
            uint32 K, uint32 Co, uint32 Ci, uint32 Ko, uint32 Ki, uint32 KC,
            uint32 PQ, uint32 RS, uint32 RSC, uint32 unary_param0) {
  tanto_relu_max_init();
  uint32 Ct = C / 32;
  uint32 Kt = K / 32;
  px.frame_size = Ct;
  pw.frame_size = (RSC / 32);
  pb.frame_size = Ct;
  pw2.frame_size = (KC / 1024);
  pb2.frame_size = Kt;
  pz.frame_size = Kt;
  py.frame_size = Ki;
  pu_im.frame_size = Ci;
  pt_im.frame_size = Ci;
  pz_im.frame_size = Kt;
  py_im.frame_size = Ki;
  cb_wait_front(pw.cb_id, pw.frame_size);
  cb_wait_front(pb.cb_id, pb.frame_size);
  cb_wait_front(pw2.cb_id, pw2.frame_size);
  cb_wait_front(pb2.cb_id, pb2.frame_size);
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      // Layer 1: depthwise
      uint32 iw = 0;
      for (uint32 rs = 0; rs < RS; rs++) {
        cb_wait_front(px.cb_id, px.frame_size);
        uint32 cx = 0;
        for (uint32 co = 0; co < Co; co++) {
          if (rs == 0) {
            tanto_unpack_binary_init(px.cb_id, pw.cb_id);
            tanto_mul_init();
            tanto_pack_init(pu_im.cb_id);
            // pu_im = px * pw
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32 ci = 0; ci < Ci; ci++) {
              mul_tiles(px.cb_id, pw.cb_id, cx, iw, ci);
              cx++;
              iw++;
            }
            cb_reserve_back(pu_im.cb_id, pu_im.frame_size);
            for (uint32 ci = 0; ci < Ci; ci++) {
              pack_tile(ci, pu_im.cb_id);
            }
            cb_push_back(pu_im.cb_id, pu_im.frame_size);
            tile_regs_commit();
            tile_regs_release();
          } else {
            // pt_im = px * pw
            {
              tanto_unpack_binary_init(px.cb_id, pw.cb_id);
              tanto_mul_init();
              tanto_pack_init(pt_im.cb_id);
              tile_regs_acquire();
              tile_regs_wait();
              for (uint32 ci = 0; ci < Ci; ci++) {
                mul_tiles(px.cb_id, pw.cb_id, cx, iw, ci);
                cx++;
                iw++;
              }
              cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                pack_tile(ci, pt_im.cb_id);
              }
              cb_push_back(pt_im.cb_id, pt_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
            // pu_im = pu_im + pt_im
            {
              tanto_unpack_binary_init(pu_im.cb_id, pt_im.cb_id);
              tanto_add_init();
              tanto_pack_init(pu_im.cb_id);
              tile_regs_acquire();
              tile_regs_wait();
              cb_wait_front(pt_im.cb_id, pt_im.frame_size);
              cb_wait_front(pu_im.cb_id, pu_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                add_tiles(pu_im.cb_id, pt_im.cb_id, ci, ci, ci);
              }
              cb_pop_front(pu_im.cb_id, pu_im.frame_size);
              cb_pop_front(pt_im.cb_id, pt_im.frame_size);
              cb_reserve_back(pu_im.cb_id, pu_im.frame_size);
              for (uint32 ci = 0; ci < Ci; ci++) {
                pack_tile(ci, pu_im.cb_id);
              }
              cb_push_back(pu_im.cb_id, pu_im.frame_size);
              tile_regs_commit();
              tile_regs_release();
            } // acc
          }
        } // co
        cb_pop_front(px.cb_id, px.frame_size);
      } // rs
      // pu_im = unary(pu_im + pb)
      uint32 ib = 0;
      tanto_unpack_binary_init(pu_im.cb_id, pb.cb_id);
      tanto_add_init();
      tanto_pack_init(pu_im.cb_id);
      for (uint32 co = 0; co < Co; co++) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(pu_im.cb_id, pu_im.frame_size);
        for (uint32 ci = 0; ci < Ci; ci++) {
          add_tiles(pu_im.cb_id, pb.cb_id, ci, ib, ci);
          unary_op(ci, unary_param0);
          ib++;
        }
        cb_pop_front(pu_im.cb_id, pu_im.frame_size);
        cb_reserve_back(pu_im.cb_id, pu_im.frame_size);
        for (uint32 ci = 0; ci < Ci; ci++) {
          pack_tile(ci, pu_im.cb_id);
        }
        cb_push_back(pu_im.cb_id, pu_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // ko
      // Layer 2: pointwise
      // pt_im = tilize(pu_im)
      pu_im.frame_size = Ct;
      pt_im.frame_size = Ct;
      cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
      cb_wait_front(pu_im.cb_id, pu_im.frame_size);
      tanto_unpack_tilize_block_init(pu_im.cb_id, Ct);
      tanto_copy_init();
      tanto_pack_init(pt_im.cb_id);
      tilize_block(pu_im.cb_id, Ct, pt_im.cb_id);
      cb_pop_front(pu_im.cb_id, pu_im.frame_size);
      cb_push_back(pt_im.cb_id, pt_im.frame_size);
      // py_im = matmul(pt_im, pw2)
      cb_wait_front(pt_im.cb_id, pt_im.frame_size);
      uint32 iw2 = 0;
      tanto_unpack_matmul_init(pt_im.cb_id, pw2.cb_id, true);
      tanto_matmul_init(true);
      tanto_pack_init(py_im.cb_id);
      for (uint32 ko = 0; ko < Ko; ko++) {
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32 ki = 0; ki < Ki; ki++) {
          matmul_slice(pt_im, pw2, iw2, ki, Ct);
          iw2 += Ct;
        }
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          pack_tile(ki, py_im.cb_id);
        }
        cb_push_back(py_im.cb_id, py_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // ko
      cb_pop_front(pt_im.cb_id, pt_im.frame_size);
      // restore pu_im, pt_im to small frames
      pu_im.frame_size = Ci;
      pt_im.frame_size = Ci;
      // py_im = py_im + pb2
      uint32 ib2 = 0;
      tanto_unpack_bcast_rows_init(py_im.cb_id, pb2.cb_id);
      tanto_add_bcast_rows_init();
      tanto_pack_init(py_im.cb_id);
      for (uint32 ko = 0; ko < Ko; ko++) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(
              py_im.cb_id, pb2.cb_id, ki, ib2, ki);
          ib2++;
        }
        cb_pop_front(py_im.cb_id, py_im.frame_size);
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          pack_tile(ki, py_im.cb_id);
        }
        cb_push_back(py_im.cb_id, py_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // ko
      // pz_im = untilize(py_im)
      py_im.frame_size = Kt;
      cb_reserve_back(pz_im.cb_id, pz_im.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_unpack_untilize_block_init(py_im.cb_id);
      tanto_copy_init();
      tanto_pack_init(pz_im.cb_id);
      untilize_block<1>(py_im.cb_id, Kt, pz_im.cb_id);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(pz_im.cb_id, pz_im.frame_size);
      py_im.frame_size = Ki;
      // py = pz_im + pz
      cb_wait_front(pz_im.cb_id, pz_im.frame_size);
      cb_wait_front(pz.cb_id, pz.frame_size);
      uint32 kz = 0;
      tanto_unpack_binary_init(pz_im.cb_id, pz.cb_id);
      tanto_add_init();
      tanto_pack_init(py.cb_id);
      for (uint32 ko = 0; ko < Ko; ko++) {
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32 ki = 0; ki < Ki; ki++) {
          add_tiles(pz_im.cb_id, pz.cb_id, kz, kz, ki);
          kz++;
        }
        cb_reserve_back(py.cb_id, py.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          pack_tile(ki, py.cb_id);
        }
        cb_push_back(py.cb_id, py.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // ko
      cb_pop_front(pz.cb_id, pz.frame_size);
      cb_pop_front(pz_im.cb_id, pz_im.frame_size);
    } // pq_start
  }   // n
  cb_pop_front(pw.cb_id, pw.frame_size);
  cb_pop_front(pb.cb_id, pb.frame_size);
  cb_pop_front(pw2.cb_id, pw2.frame_size);
  cb_pop_front(pb2.cb_id, pb2.frame_size);
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
  Pipe pw2;
  pw2.cb_id = get_arg_val<uint32>(6);
  pw2.frame_size = get_arg_val<uint32>(7);
  Pipe pb2;
  pb2.cb_id = get_arg_val<uint32>(8);
  pb2.frame_size = get_arg_val<uint32>(9);
  Pipe pz;
  pz.cb_id = get_arg_val<uint32>(10);
  pz.frame_size = get_arg_val<uint32>(11);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(12);
  py.frame_size = get_arg_val<uint32>(13);
  Pipe pu_im;
  pu_im.cb_id = get_arg_val<uint32>(14);
  pu_im.frame_size = get_arg_val<uint32>(15);
  Pipe pt_im;
  pt_im.cb_id = get_arg_val<uint32>(16);
  pt_im.frame_size = get_arg_val<uint32>(17);
  Pipe pz_im;
  pz_im.cb_id = get_arg_val<uint32>(18);
  pz_im.frame_size = get_arg_val<uint32>(19);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(20);
  py_im.frame_size = get_arg_val<uint32>(21);
  uint32 N = get_arg_val<uint32>(22);
  uint32 C = get_arg_val<uint32>(23);
  uint32 K = get_arg_val<uint32>(24);
  uint32 Co = get_arg_val<uint32>(25);
  uint32 Ci = get_arg_val<uint32>(26);
  uint32 Ko = get_arg_val<uint32>(27);
  uint32 Ki = get_arg_val<uint32>(28);
  uint32 KC = get_arg_val<uint32>(29);
  uint32 PQ = get_arg_val<uint32>(30);
  uint32 RS = get_arg_val<uint32>(31);
  uint32 RSC = get_arg_val<uint32>(32);
  uint32 unary_param0 = get_arg_val<uint32>(33);
  tanto_compute_init();
  kernel(px, pw, pb, pw2, pb2, pz, py, pu_im, pt_im, pz_im, py_im, N, C, K, Co,
         Ci, Ko, Ki, KC, PQ, RS, RSC, unary_param0);
}
} // namespace NAMESPACE

