// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gx, Local ly, uint32 xpos, uint32 ypos, uint32 H, uint32 C,
            uint32 Cb) {
  for (uint32 h = 0; h < H; h++) {
    noc_async_read_global_dram(ly.addr + (ypos << 1), gx.addr,
                               gx.log2_page_size, xpos << 1, Cb << 1);
    xpos += C;
    ypos += Cb;
  }
  noc_async_read_barrier();
}

void kernel_main() {
  Global gx;
  gx.addr = get_arg_val<uint32>(0);
  gx.log2_page_size = get_arg_val<uint32>(1);
  Local ly;
  ly.addr = get_arg_val<uint32>(2);
  uint32 xpos = get_arg_val<uint32>(3);
  uint32 ypos = get_arg_val<uint32>(4);
  uint32 H = get_arg_val<uint32>(5);
  uint32 C = get_arg_val<uint32>(6);
  uint32 Cb = get_arg_val<uint32>(7);
  kernel(gx, ly, xpos, ypos, H, C, Cb);
}

