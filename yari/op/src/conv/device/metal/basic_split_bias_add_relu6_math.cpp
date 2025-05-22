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

    Pipe px, Pipe pw, uint32 pwoff, uint32 idst, uint32 tiles) {
  for (uint32 i = 0; i < tiles; i++) {
    matmul_tiles(px.cb_id, pw.cb_id, i, i + pwoff, idst, true);
  }
}

void kernel(Pipe px, Pipe pw, Pipe pb, Pipe pz, Pipe py, Pipe px_im, Pipe pz_im,
            Pipe py_im, uint32 N, uint32 C, uint32 PQ, uint32 RS, uint32 Kb,
            uint32 RSKbC, uint32 unary_param0) {
  tanto_relu_max_init();
  uint32 Ct = C / 32;
  uint32 Kbt = Kb / 32;
  px.frame_size = Ct;
  pw.frame_size = (RSKbC / 1024);
  pb.frame_size = Kbt;
  pz.frame_size = Kbt;
  py.frame_size = Kbt;
  px_im.frame_size = Ct;
  pz_im.frame_size = Kbt;
  py_im.frame_size = Kbt;
  cb_wait_front(pw.cb_id, pw.frame_size);
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      uint32 w_start = 0;
      for (uint32 rs = 0; rs < RS; rs++) {
        // px_im = tilize(px)
        cb_reserve_back(px_im.cb_id, px_im.frame_size);
        cb_wait_front(px.cb_id, px.frame_size);
        tanto_unpack_tilize_block_init(px.cb_id, Ct);
        tanto_copy_init();
        tanto_pack_init(px_im.cb_id);
        tilize_block(px.cb_id, Ct, px_im.cb_id);
        cb_pop_front(px.cb_id, px.frame_size);
        cb_push_back(px_im.cb_id, px_im.frame_size);
        // py_im = matmul(px_im, pw)
        cb_wait_front(px_im.cb_id, px_im.frame_size);
        tile_regs_acquire();
        tile_regs_wait();
        if (rs != 0) {
          tanto_unpack_unary_init(py_im.cb_id);
          tanto_copy_init();
          cb_wait_front(py_im.cb_id, py_im.frame_size);
          for (uint32 k = 0; k < Kbt; k++) {
            copy_tile(py_im.cb_id, k, k);
          }
          cb_pop_front(py_im.cb_id, py_im.frame_size);
        }
        tanto_unpack_matmul_init(px_im.cb_id, pw.cb_id, true);
        tanto_matmul_init(true);
        for (uint32 k = 0; k < Kbt; k++) {
          matmul_slice(px_im, pw, w_start, k, Ct);
          w_start += Ct;
        }
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        tanto_pack_init(py_im.cb_id);
        for (uint32 k = 0; k < Kbt; k++) {
          pack_tile(k, py_im.cb_id);
        }
        cb_push_back(py_im.cb_id, py_im.frame_size);
        cb_pop_front(px_im.cb_id, px_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // rs
      // py_im = py_im + pb
      {
        tanto_unpack_bcast_rows_init(py_im.cb_id, pb.cb_id);
        tanto_add_bcast_rows_init();
        tanto_pack_init(py_im.cb_id);
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 k = 0; k < Kbt; k++) {
          any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(
              py_im.cb_id, pb.cb_id, k, k, k);
        }
        cb_pop_front(py_im.cb_id, py_im.frame_size);
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        for (uint32 k = 0; k < Kbt; k++) {
          pack_tile(k, py_im.cb_id);
        }
        cb_push_back(py_im.cb_id, py_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // acc
      // pz_im = tilize(pz)
      cb_reserve_back(pz_im.cb_id, pz_im.frame_size);
      cb_wait_front(pz.cb_id, pz.frame_size);
      tanto_unpack_tilize_block_init(pz.cb_id, Kbt);
      tanto_copy_init();
      tanto_pack_init(pz_im.cb_id);
      tilize_block(pz.cb_id, Kbt, pz_im.cb_id);
      cb_pop_front(pz.cb_id, pz.frame_size);
      cb_push_back(pz_im.cb_id, pz_im.frame_size);
      // py_im = unary(py_im + pz_im)
      {
        tanto_unpack_binary_init(py_im.cb_id, pz_im.cb_id);
        tanto_add_init();
        tanto_pack_init(py_im.cb_id);
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(pz_im.cb_id, pz_im.frame_size);
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 k = 0; k < Kbt; k++) {
          add_tiles(py_im.cb_id, pz_im.cb_id, k, k, k);
          unary_op(k, unary_param0);
        }
        cb_pop_front(py_im.cb_id, py_im.frame_size);
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        for (uint32 k = 0; k < Kbt; k++) {
          pack_tile(k, py_im.cb_id);
        }
        cb_push_back(py_im.cb_id, py_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      } // acc
      cb_pop_front(pz_im.cb_id, pz_im.frame_size);
      // py = untilize(py_im)
      cb_reserve_back(py.cb_id, py.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_unpack_untilize_block_init(py_im.cb_id);
      tanto_copy_init();
      tanto_pack_init(py.cb_id);
      untilize_block<1>(py_im.cb_id, Kbt, py.cb_id);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(py.cb_id, py.frame_size);
    } // pq_start
  }   // n
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
  Pipe pz;
  pz.cb_id = get_arg_val<uint32>(6);
  pz.frame_size = get_arg_val<uint32>(7);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(8);
  py.frame_size = get_arg_val<uint32>(9);
  Pipe px_im;
  px_im.cb_id = get_arg_val<uint32>(10);
  px_im.frame_size = get_arg_val<uint32>(11);
  Pipe pz_im;
  pz_im.cb_id = get_arg_val<uint32>(12);
  pz_im.frame_size = get_arg_val<uint32>(13);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(14);
  py_im.frame_size = get_arg_val<uint32>(15);
  uint32 N = get_arg_val<uint32>(16);
  uint32 C = get_arg_val<uint32>(17);
  uint32 PQ = get_arg_val<uint32>(18);
  uint32 RS = get_arg_val<uint32>(19);
  uint32 Kb = get_arg_val<uint32>(20);
  uint32 RSKbC = get_arg_val<uint32>(21);
  uint32 unary_param0 = get_arg_val<uint32>(22);
  tanto_compute_init();
  kernel(px, pw, pb, pz, py, px_im, pz_im, py_im, N, C, PQ, RS, Kb, RSKbC,
         unary_param0);
}
} // namespace NAMESPACE

