// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

void matmul_slice(

    Pipe px, Pipe pw, uint32 idst, uint32 tiles) {
  for (uint32 i = 0; i < tiles; i++) {
    matmul_tiles(px.cb_id, pw.cb_id, i, i, idst, true);
  }
}

void prepare_interp(Pipe pd, Pipe pd1_im, Pipe pd2_im, Pipe ppi_im, Pipe ppc_im,
                    uint32 Dt) {
  // pd [PQ32, D]
  cb_wait_front(pd.cb_id, pd.frame_size);
  // ppi_im = uint16(floor(pd) + 63)
  // add 63 to encode negative values as uint16
  cb_reserve_back(ppi_im.cb_id, ppi_im.frame_size);
  tanto_unpack_unary_init(pd.cb_id);
  tanto_copy_init();
  tanto_pack_init(ppi_im.cb_id);
  for (uint32 i = 0; i < Dt; i++) {
    tile_regs_acquire();
    tile_regs_wait();
    constexpr uint32 C63 = 0x427c0000;
    // unpack $0
    copy_tile(pd.cb_id, i, 0);
    // $0 = floor($0)
    tanto_floor_init();
    floor_tile(0);
    // $0 = $0 + 63
    tanto_fill_init();
    fill_tile_bitcast(1, C63);
    tanto_add_dst_init();
    add_binary_tile(0, 1);
    // $0 = uint16($0)
    tanto_cast_init();
    tanto_cast_bf16_u16(0);
    pack_tile(0, ppi_im.cb_id);
    tile_regs_commit();
    tile_regs_release();
  }
  cb_push_back(ppi_im.cb_id, ppi_im.frame_size);
  // pd1_im = tilize(pd)
  cb_reserve_back(pd1_im.cb_id, pd1_im.frame_size);
  tanto_unpack_tilize_block_init(pd.cb_id, Dt);
  tanto_copy_init();
  tanto_pack_init(pd1_im.cb_id);
  tilize_block(pd.cb_id, Dt, pd1_im.cb_id);
  cb_push_back(pd1_im.cb_id, pd1_im.frame_size);
  cb_pop_front(pd.cb_id, pd.frame_size);
  // pd2_im = transpose(pd1_im)
  cb_reserve_back(pd2_im.cb_id, pd2_im.frame_size);
  cb_wait_front(pd1_im.cb_id, pd1_im.frame_size);
  tanto_unpack_transpose_init(pd1_im.cb_id);
  tanto_transpose_init();
  tanto_pack_init(pd2_im.cb_id);
  for (uint32 i = 0; i < Dt; i++) {
    tile_regs_acquire();
    tile_regs_wait();
    transpose_wh_tile(pd1_im.cb_id, i, 0);
    pack_tile(0, pd2_im.cb_id);
    tile_regs_commit();
    tile_regs_release();
  }
  cb_pop_front(pd1_im.cb_id, pd1_im.frame_size);
  cb_push_back(pd2_im.cb_id, pd2_im.frame_size);
  // pd1_im = tile-wise untilize (pd2_im)
  pd1_im.frame_size = 1;
  pd2_im.frame_size = 1;
  tanto_unpack_untilize_block_init(pd2_im.cb_id);
  tanto_copy_init();
  tanto_pack_init(pd1_im.cb_id);
  for (uint32 i = 0; i < Dt; i++) {
    cb_reserve_back(pd1_im.cb_id, pd1_im.frame_size);
    cb_wait_front(pd2_im.cb_id, pd2_im.frame_size);
    untilize_block<1>(pd2_im.cb_id, 1, pd1_im.cb_id);
    cb_pop_front(pd2_im.cb_id, pd2_im.frame_size);
    cb_push_back(pd1_im.cb_id, pd1_im.frame_size);
  }
  pd1_im.frame_size = Dt;
  pd2_im.frame_size = Dt;
  // ppc_im = prepare(pd1_im)
  cb_reserve_back(ppc_im.cb_id, ppc_im.frame_size);
  cb_wait_front(pd1_im.cb_id, pd1_im.frame_size);
  tanto_unpack_unary_init(pd1_im.cb_id);
  tanto_copy_init();
  tanto_pack_init(ppc_im.cb_id);
  for (uint32 i = 0; i < Dt; i++) {
    tile_regs_acquire();
    tile_regs_wait();
    // dst[2] offset
    // dst[1] lh/lw
    // dst[0] hh/hw
    constexpr uint32 ONE = 0x3f800000;
    // unpack $0
    copy_tile(pd1_im.cb_id, i, 2);
    // $1 = floor($2)
    tanto_copy_dst_init();
    copy_dest_values(1, 2);
    tanto_floor_init();
    floor_tile(1);
    // $1 = $2 - $1
    tanto_rsub_dst_init();
    rsub_binary_tile(1, 2);
    // $0 = 1.0 - $1
    tanto_fill_init();
    fill_tile_bitcast(0, ONE);
    tanto_sub_dst_init();
    sub_binary_tile(0, 1);
    // pack $1, $0
    pack_tile(1, ppc_im.cb_id);
    pack_tile(0, ppc_im.cb_id);
    tile_regs_commit();
    tile_regs_release();
  }
  cb_pop_front(pd1_im.cb_id, pd1_im.frame_size);
  cb_push_back(ppc_im.cb_id, ppc_im.frame_size);
}

void interp(Pipe px, Pipe px_im, Pipe pt_im, Pipe pc1_im, Pipe pc2_im,
            uint32 Ct, uint32 Co, uint32 Ci) {
  // pc2_im = tilize(pc1_im)
  cb_reserve_back(pc2_im.cb_id, pc2_im.frame_size);
  cb_wait_front(pc1_im.cb_id, pc1_im.frame_size);
  tanto_unpack_tilize_block_init(pc1_im.cb_id, 4);
  tanto_copy_init();
  tanto_pack_init(pc2_im.cb_id);
  tilize_block(pc1_im.cb_id, 4, pc2_im.cb_id);
  cb_pop_front(pc1_im.cb_id, pc1_im.frame_size);
  cb_push_back(pc2_im.cb_id, pc2_im.frame_size);
  // pc2_im = transpose(pc2_im)
  {
    tanto_unpack_transpose_init(pc2_im.cb_id);
    tanto_transpose_init();
    tanto_pack_init(pc2_im.cb_id);
    tile_regs_acquire();
    tile_regs_wait();
    cb_wait_front(pc2_im.cb_id, pc2_im.frame_size);
    for (uint32 i = 0; i < 4; i++) {
      transpose_wh_tile(pc2_im.cb_id, i, i);
    }
    cb_pop_front(pc2_im.cb_id, pc2_im.frame_size);
    cb_reserve_back(pc2_im.cb_id, pc2_im.frame_size);
    for (uint32 i = 0; i < 4; i++) {
      pack_tile(i, pc2_im.cb_id);
    }
    cb_push_back(pc2_im.cb_id, pc2_im.frame_size);
    tile_regs_commit();
    tile_regs_release();
  }
  cb_wait_front(pc2_im.cb_id, pc2_im.frame_size);
  // px_im = 0
  cb_reserve_back(px_im.cb_id, px_im.frame_size);
  tanto_pack_init(px_im.cb_id);
  for (uint32 c = 0; c < Ct; c++) {
    tile_regs_acquire();
    tile_regs_wait();
    fill_tile_bitcast(0, 0);
    pack_tile(0, px_im.cb_id);
    tile_regs_commit();
    tile_regs_release();
  }
  cb_push_back(px_im.cb_id, px_im.frame_size);
  for (uint32 i = 0; i < 4; i++) {
    // pt_im = tilize(px)
    cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
    cb_wait_front(px.cb_id, px.frame_size);
    tanto_unpack_tilize_block_init(px.cb_id, Ct);
    tanto_copy_init();
    tanto_pack_init(pt_im.cb_id);
    tilize_block(px.cb_id, Ct, pt_im.cb_id);
    cb_pop_front(px.cb_id, px.frame_size);
    cb_push_back(pt_im.cb_id, pt_im.frame_size);
    // c0 = hh * hw
    // c1 = hh * lw
    // c2 = lh * hw
    // c3 = lh * lw
    // [th, tw] in [[0, 2], [0, 3], [1, 2], [1, 3]]
    //     where 0 = hh, 1 = lh, 2 = hw, 3 = lw
    uint32 th = (i >> 1) & 1;
    uint32 tw = (i & 1) + 2;
    // narrow frames
    px_im.frame_size = Ci;
    pt_im.frame_size = Ci;
    // pt_im = pt_im * pc2_im[th]
    tanto_unpack_bcast_cols_init(pt_im.cb_id, pc2_im.cb_id);
    tanto_mul_bcast_cols_init();
    tanto_pack_init(pt_im.cb_id);
    for (uint32 co = 0; co < Co; co++) {
      tile_regs_acquire();
      tile_regs_wait();
      cb_wait_front(pt_im.cb_id, pt_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
            pt_im.cb_id, pc2_im.cb_id, ci, th, ci);
      }
      cb_pop_front(pt_im.cb_id, pt_im.frame_size);
      cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        pack_tile(ci, pt_im.cb_id);
      }
      cb_push_back(pt_im.cb_id, pt_im.frame_size);
      tile_regs_commit();
      tile_regs_release();
    }
    // pt_im = pt_im * pc2_im[tw]
    tanto_unpack_bcast_cols_init(pt_im.cb_id, pc2_im.cb_id);
    tanto_mul_bcast_cols_init();
    tanto_pack_init(pt_im.cb_id);
    for (uint32 co = 0; co < Co; co++) {
      tile_regs_acquire();
      tile_regs_wait();
      cb_wait_front(pt_im.cb_id, pt_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        any_tiles_bcast<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(
            pt_im.cb_id, pc2_im.cb_id, ci, tw, ci);
      }
      cb_pop_front(pt_im.cb_id, pt_im.frame_size);
      cb_reserve_back(pt_im.cb_id, pt_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        pack_tile(ci, pt_im.cb_id);
      }
      cb_push_back(pt_im.cb_id, pt_im.frame_size);
      tile_regs_commit();
      tile_regs_release();
    }
    // px_im = px_im + pt_im
    tanto_unpack_binary_init(px_im.cb_id, pt_im.cb_id);
    tanto_add_init();
    tanto_pack_init(px_im.cb_id);
    for (uint32 co = 0; co < Co; co++) {
      tile_regs_acquire();
      tile_regs_wait();
      cb_wait_front(px_im.cb_id, px_im.frame_size);
      cb_wait_front(pt_im.cb_id, pt_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        add_tiles(px_im.cb_id, pt_im.cb_id, ci, ci, ci);
      }
      cb_pop_front(px_im.cb_id, px_im.frame_size);
      cb_pop_front(pt_im.cb_id, pt_im.frame_size);
      cb_reserve_back(px_im.cb_id, px_im.frame_size);
      for (uint32 ci = 0; ci < Ci; ci++) {
        pack_tile(ci, px_im.cb_id);
      }
      cb_push_back(px_im.cb_id, px_im.frame_size);
      tile_regs_commit();
      tile_regs_release();
    }
    // restore frames
    px_im.frame_size = Ct;
    pt_im.frame_size = Ct;
  }
  cb_pop_front(pc2_im.cb_id, pc2_im.frame_size);
}

void kernel(Pipe px, Pipe pd, Pipe pw, Pipe pb, Pipe py, Pipe px_im, Pipe pt_im,
            Pipe py_im, Pipe pd1_im, Pipe pd2_im, Pipe ppi_im, Pipe ppc_im,
            Pipe pc1_im, Pipe pc2_im, uint32 N, uint32 C, uint32 K, uint32 D,
            uint32 Co, uint32 Ci, uint32 Ko, uint32 Ki, uint32 PQ, uint32 RS) {
  uint32 Ct = C / 32;
  uint32 Kt = K / 32;
  uint32 Dt = D / 32;
  px.frame_size = Ct;
  pd.frame_size = Dt;
  pw.frame_size = Ct;
  pb.frame_size = Kt;
  py.frame_size = Kt;
  px_im.frame_size = Ct;
  pt_im.frame_size = Ct;
  py_im.frame_size = Ki;
  pd1_im.frame_size = Dt;
  pd2_im.frame_size = Dt;
  ppi_im.frame_size = Dt;
  ppc_im.frame_size = (Dt * 2);
  pc1_im.frame_size = 4;
  pc2_im.frame_size = 4;
  cb_wait_front(pb.cb_id, pb.frame_size);
  for (uint32 n = 0; n < N; n++) {
    for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
      prepare_interp(pd, pd1_im, pd2_im, ppi_im, ppc_im, Dt);
      tanto_fill_init();
      for (uint32 rs = 0; rs < RS; rs++) {
        // px_im = interp(px)
        interp(px, px_im, pt_im, pc1_im, pc2_im, Ct, Co, Ci);
        // py_im = matmul(px_im, pw)
        cb_wait_front(px_im.cb_id, px_im.frame_size);
        tanto_pack_init(py_im.cb_id);
        for (uint32 ko = 0; ko < Ko; ko++) {
          tile_regs_acquire();
          tile_regs_wait();
          if (rs != 0) {
            tanto_unpack_unary_init(py_im.cb_id);
            tanto_copy_init();
            cb_wait_front(py_im.cb_id, py_im.frame_size);
            for (uint32 ki = 0; ki < Ki; ki++) {
              copy_tile(py_im.cb_id, ki, ki);
            }
            cb_pop_front(py_im.cb_id, py_im.frame_size);
          }
          tanto_unpack_matmul_init(px_im.cb_id, pw.cb_id, true);
          tanto_matmul_init(true);
          for (uint32 ki = 0; ki < Ki; ki++) {
            cb_wait_front(pw.cb_id, pw.frame_size);
            matmul_slice(px_im, pw, ki, Ct);
            cb_pop_front(pw.cb_id, pw.frame_size);
          }
          cb_reserve_back(py_im.cb_id, py_im.frame_size);
          for (uint32 ki = 0; ki < Ki; ki++) {
            pack_tile(ki, py_im.cb_id);
          }
          cb_push_back(py_im.cb_id, py_im.frame_size);
          tile_regs_commit();
          tile_regs_release();
        } // ko
        cb_pop_front(px_im.cb_id, px_im.frame_size);
      } // rs
      // py_im = py_im + pb
      uint32 kb = 0;
      tanto_unpack_bcast_rows_init(py_im.cb_id, pb.cb_id);
      tanto_add_bcast_rows_init();
      tanto_pack_init(py_im.cb_id);
      for (uint32 ko = 0; ko < Ko; ko++) {
        tile_regs_acquire();
        tile_regs_wait();
        cb_wait_front(py_im.cb_id, py_im.frame_size);
        for (uint32 ki = 0; ki < Ki; ki++) {
          any_tiles_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(
              py_im.cb_id, pb.cb_id, ki, kb, ki);
          kb++;
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
      // py = untilize(py_im)
      py_im.frame_size = Kt;
      cb_reserve_back(py.cb_id, py.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_unpack_untilize_block_init(py_im.cb_id);
      tanto_copy_init();
      tanto_pack_init(py.cb_id);
      untilize_block<1>(py_im.cb_id, Kt, py.cb_id);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(py.cb_id, py.frame_size);
      py_im.frame_size = Ki;
    } // pq_start
  }   // n
  cb_pop_front(pb.cb_id, pb.frame_size);
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe pd;
  pd.cb_id = get_arg_val<uint32>(2);
  pd.frame_size = get_arg_val<uint32>(3);
  Pipe pw;
  pw.cb_id = get_arg_val<uint32>(4);
  pw.frame_size = get_arg_val<uint32>(5);
  Pipe pb;
  pb.cb_id = get_arg_val<uint32>(6);
  pb.frame_size = get_arg_val<uint32>(7);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(8);
  py.frame_size = get_arg_val<uint32>(9);
  Pipe px_im;
  px_im.cb_id = get_arg_val<uint32>(10);
  px_im.frame_size = get_arg_val<uint32>(11);
  Pipe pt_im;
  pt_im.cb_id = get_arg_val<uint32>(12);
  pt_im.frame_size = get_arg_val<uint32>(13);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(14);
  py_im.frame_size = get_arg_val<uint32>(15);
  Pipe pd1_im;
  pd1_im.cb_id = get_arg_val<uint32>(16);
  pd1_im.frame_size = get_arg_val<uint32>(17);
  Pipe pd2_im;
  pd2_im.cb_id = get_arg_val<uint32>(18);
  pd2_im.frame_size = get_arg_val<uint32>(19);
  Pipe ppi_im;
  ppi_im.cb_id = get_arg_val<uint32>(20);
  ppi_im.frame_size = get_arg_val<uint32>(21);
  Pipe ppc_im;
  ppc_im.cb_id = get_arg_val<uint32>(22);
  ppc_im.frame_size = get_arg_val<uint32>(23);
  Pipe pc1_im;
  pc1_im.cb_id = get_arg_val<uint32>(24);
  pc1_im.frame_size = get_arg_val<uint32>(25);
  Pipe pc2_im;
  pc2_im.cb_id = get_arg_val<uint32>(26);
  pc2_im.frame_size = get_arg_val<uint32>(27);
  uint32 N = get_arg_val<uint32>(28);
  uint32 C = get_arg_val<uint32>(29);
  uint32 K = get_arg_val<uint32>(30);
  uint32 D = get_arg_val<uint32>(31);
  uint32 Co = get_arg_val<uint32>(32);
  uint32 Ci = get_arg_val<uint32>(33);
  uint32 Ko = get_arg_val<uint32>(34);
  uint32 Ki = get_arg_val<uint32>(35);
  uint32 PQ = get_arg_val<uint32>(36);
  uint32 RS = get_arg_val<uint32>(37);
  tanto_compute_init();
  kernel(px, pd, pw, pb, py, px_im, pt_im, py_im, pd1_im, pd2_im, ppi_im,
         ppc_im, pc1_im, pc2_im, N, C, K, D, Co, Ci, Ko, Ki, PQ, RS);
}
} // namespace NAMESPACE

