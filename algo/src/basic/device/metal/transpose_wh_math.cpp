// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

// Originally "transpose_wh.cpp"

void kernel(Pipe px, Pipe py, uint32 NHtWt) {
  tanto_unpack_transpose_init(px.cb_id);
  tanto_transpose_init();
  tanto_pack_init(py.cb_id);
  // transpose a row-major block
  // assumes the tiles come in in column major order from reader
  for (uint32 n = 0; n < NHtWt; n++) {
    cb_wait_front(px.cb_id, px.frame_size);
    cb_reserve_back(py.cb_id, py.frame_size);
    tile_regs_acquire();
    tile_regs_wait();
    transpose_wh_tile(px.cb_id, 0, 0);
    pack_tile(0, py.cb_id);
    cb_push_back(py.cb_id, py.frame_size);
    cb_pop_front(px.cb_id, px.frame_size);
    tile_regs_commit();
    tile_regs_release();
  }
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  uint32 NHtWt = get_arg_val<uint32>(4);
  tanto_compute_init();
  kernel(px, py, NHtWt);
}
} // namespace NAMESPACE

