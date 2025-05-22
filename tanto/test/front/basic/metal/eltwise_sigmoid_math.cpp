// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/compute.h"

#define T bfloat16

namespace NAMESPACE {

static constexpr uint32 SFPU_OP_ABS = 0, SFPU_OP_ACOS = 1, SFPU_OP_ASIN = 2,
                        SFPU_OP_ATAN = 3, SFPU_OP_COS = 4, SFPU_OP_ELU = 5,
                        SFPU_OP_EQZ = 6, SFPU_OP_ERF = 7, SFPU_OP_ERFC = 8,
                        SFPU_OP_ERFINV = 9, SFPU_OP_EXP = 10, SFPU_OP_EXP2 = 11,
                        SFPU_OP_EXPM1 = 12, SFPU_OP_GELU = 13, SFPU_OP_GEZ = 14,
                        SFPU_OP_GTZ = 15, SFPU_OP_HEAVISIDE = 16,
                        SFPU_OP_I0 = 17, SFPU_OP_ISFINITE = 18,
                        SFPU_OP_ISINF = 19, SFPU_OP_ISNAN = 20,
                        SFPU_OP_ISNEGINF = 21, SFPU_OP_ISPOSINF = 22,
                        SFPU_OP_LEAKY_RELU = 23, SFPU_OP_LEZ = 24,
                        SFPU_OP_LOG = 25, SFPU_OP_LOG_WITH_BASE = 26,
                        SFPU_OP_LOGICAL_NOT = 27, SFPU_OP_LTZ = 28,
                        SFPU_OP_NEZ = 29, SFPU_OP_POWER = 30,
                        SFPU_OP_RECIP = 31, SFPU_OP_RELU = 32,
                        SFPU_OP_RELU_MAX = 33, SFPU_OP_RELU_MIN = 34,
                        SFPU_OP_RSQRT = 35, SFPU_OP_SIGMOID = 36,
                        SFPU_OP_SIGN = 37, SFPU_OP_SIGNBIT = 38,
                        SFPU_OP_SIN = 39, SFPU_OP_SQRT = 40,
                        SFPU_OP_SQUARE = 41, SFPU_OP_TAN = 42,
                        SFPU_OP_TANH = 43;

static constexpr uint32 sfpu_op_code = uint32(36);

void sfpu_op(uint32 iparam, float fparam) { sigmoid_tile(0); }

void kernel(Pipe px, Pipe py, uint32 num_blocks, uint32 block_tiles,
            uint32 iparam, float fparam) {
  tanto_unpack_unary_init(px.cb_id);
  tanto_copy_init();
  tanto_pack_init(py.cb_id);
  tanto_sigmoid_init();
  for (uint32 block = 0; block < num_blocks; block++) {
    cb_reserve_back(py.cb_id, py.frame_size);
    cb_wait_front(px.cb_id, px.frame_size);
    for (uint32 i = 0; i < block_tiles; i++) {
      tile_regs_acquire();
      tile_regs_wait();
      copy_tile(px.cb_id, i, 0);
      sfpu_op(iparam, fparam);
      pack_tile(0, py.cb_id);
      tile_regs_commit();
      tile_regs_release();
    }
    cb_pop_front(px.cb_id, px.frame_size);
    cb_push_back(py.cb_id, py.frame_size);
  }
}

void MAIN {
  Pipe px;
  px.cb_id = get_arg_val<uint32>(0);
  px.frame_size = get_arg_val<uint32>(1);
  Pipe py;
  py.cb_id = get_arg_val<uint32>(2);
  py.frame_size = get_arg_val<uint32>(3);
  uint32 num_blocks = get_arg_val<uint32>(4);
  uint32 block_tiles = get_arg_val<uint32>(5);
  uint32 iparam = get_arg_val<uint32>(6);
  float fparam = get_arg_val<float>(7);
  tanto_compute_init();
  kernel(px, py, num_blocks, block_tiles, iparam, fparam);
}
} // namespace NAMESPACE

