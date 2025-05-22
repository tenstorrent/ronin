// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

static constexpr uint32 REDUCE_OP_MAX = 0, REDUCE_OP_SUM = 1;

static constexpr uint32 reduce_op_code = uint32(0);

void reduce_op(Pipe px, Pipe ps) {

  reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(px.cb_id, ps.cb_id, 0, 0,
                                                    0);
}

void kernel(Pipe px, Pipe ps, Pipe py, Pipe px_im, Pipe py_im, uint32 N,
            uint32 H, uint32 W) {
  uint32 Ht = H / 32;
  uint32 Wt = W / 32;
  px.frame_size = Wt;
  ps.frame_size = 1;
  py.frame_size = 1;
  px_im.frame_size = Wt;
  py_im.frame_size = 1;
  cb_wait_front(ps.cb_id, ps.frame_size);
  for (uint32 n = 0; n < N; n++) {
    for (uint32 h = 0; h < Ht; h++) {
      cb_reserve_back(px_im.cb_id, px_im.frame_size);
      cb_wait_front(px.cb_id, px.frame_size);
      tanto_unpack_tilize_block_init(px.cb_id, Wt);
      tanto_copy_init();
      tanto_pack_init(px_im.cb_id);
      tilize_block(px.cb_id, Wt, px_im.cb_id);
      cb_pop_front(px.cb_id, px.frame_size);
      cb_push_back(px_im.cb_id, px_im.frame_size);
      px_im.frame_size = 1;
      {
        tanto_unpack_reduce_rows_init(px_im.cb_id, ps.cb_id);
        tanto_reduce_max_rows_init();
        tanto_pack_row_init(py_im.cb_id);
        tile_regs_acquire();
        tile_regs_wait();
        for (uint32 w = 0; w < Wt; w++) {
          cb_wait_front(px_im.cb_id, px_im.frame_size);
          reduce_op(px_im, ps);
          cb_pop_front(px_im.cb_id, px_im.frame_size);
        }
        cb_reserve_back(py_im.cb_id, py_im.frame_size);
        pack_tile(0, py_im.cb_id);
        cb_push_back(py_im.cb_id, py_im.frame_size);
        tile_regs_commit();
        tile_regs_release();
      }
      px_im.frame_size = Wt;
      cb_reserve_back(py.cb_id, py.frame_size);
      cb_wait_front(py_im.cb_id, py_im.frame_size);
      tanto_unpack_untilize_block_init(py_im.cb_id);
      tanto_copy_init();
      tanto_pack_init(py.cb_id);
      untilize_block<1>(py_im.cb_id, 1, py.cb_id);
      cb_pop_front(py_im.cb_id, py_im.frame_size);
      cb_push_back(py.cb_id, py.frame_size);
    }
  }
  cb_pop_front(ps.cb_id, ps.frame_size);
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe ps;
  ps.cb_id = get_arg_val<uint32>(2);
  ps.frame_size = get_arg_val<uint32>(3);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(4);
  py.frame_size = get_arg_val<uint32>(5);
  Pipe px_im;
  px_im.cb_id = get_arg_val<uint32>(6);
  px_im.frame_size = get_arg_val<uint32>(7);
  Pipe py_im;
  py_im.cb_id = get_arg_val<uint32>(8);
  py_im.frame_size = get_arg_val<uint32>(9);
  uint32 N = get_arg_val<uint32>(10);
  uint32 H = get_arg_val<uint32>(11);
  uint32 W = get_arg_val<uint32>(12);
  tanto_compute_init();
  kernel(px, ps, py, px_im, py_im, N, H, W);
}
} // namespace NAMESPACE

