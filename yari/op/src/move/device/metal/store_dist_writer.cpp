// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Local lx, Global gy, uint32 xpos, uint32 ypos, uint32 H, uint32 C,
            uint32 Cb) {
  for (uint32 h = 0; h < H; h++) {
    noc_async_write_global_dram(lx.addr + (xpos << 1), gy.addr,
                                gy.log2_page_size, ypos << 1, Cb << 1);
    xpos += Cb;
    ypos += C;
  }
  noc_async_write_barrier();
}

void kernel_main() {
  Local lx;
  lx.addr = get_arg_val<uint32>(0);
  Global gy;
  gy.addr = get_arg_val<uint32>(1);
  gy.log2_page_size = get_arg_val<uint32>(2);
  uint32 xpos = get_arg_val<uint32>(3);
  uint32 ypos = get_arg_val<uint32>(4);
  uint32 H = get_arg_val<uint32>(5);
  uint32 C = get_arg_val<uint32>(6);
  uint32 Cb = get_arg_val<uint32>(7);
  kernel(lx, gy, xpos, ypos, H, C, Cb);
}

