// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

void kernel(Global gc, Pipe pc, uint32 N, uint32 num_frames, uint32 frame_tiles,
            uint32 start, uint32 stride) {
  pc.frame_size = frame_tiles;
  uint32 frame_items = frame_tiles * 1024;
  for (uint32 n = 0; n < N; n++) {
    uint32 pos = start;
    for (uint32 i = 0; i < num_frames; i++) {
      cb_wait_front(pc.cb_id, pc.frame_size);
      noc_async_write_global_dram(get_read_ptr(pc.cb_id) + (0 << 1), gc.addr,
                                  gc.log2_page_size, pos << 1,
                                  frame_items << 1);
      noc_async_write_barrier();
      cb_pop_front(pc.cb_id, pc.frame_size);
      pos += frame_items;
    }
    start += stride;
  }
}

void kernel_main() {
  Global gc;
  gc.addr = get_arg_val<uint32>(0);
  gc.log2_page_size = get_arg_val<uint32>(1);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(2);
  pc.frame_size = get_arg_val<uint32>(3);
  uint32 N = get_arg_val<uint32>(4);
  uint32 num_frames = get_arg_val<uint32>(5);
  uint32 frame_tiles = get_arg_val<uint32>(6);
  uint32 start = get_arg_val<uint32>(7);
  uint32 stride = get_arg_val<uint32>(8);
  kernel(gc, pc, N, num_frames, frame_tiles, start, stride);
}

