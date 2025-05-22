// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tanto/dataflow.h"

#define T bfloat16

// Originally
// "tt_metal/programming_examples/matmul_common/kernels/dataflow/reader_bmm_8bank.cpp"

void kernel(Global gc, Pipe pc, uint32 gc_pos, uint32 batch, uint32 Mt,
            uint32 Nt) {
  constexpr uint32 onetile = 1024;
  for (uint32 nb = 0; nb < batch; nb++) {
    for (uint32 mt = 0; mt < Mt; mt++) {
      for (uint32 nt = 0; nt < Nt; nt++) {
        cb_wait_front(pc.cb_id, pc.frame_size);
        noc_async_write_global_dram(get_read_ptr(pc.cb_id) + (0 << 1), gc.addr,
                                    gc.log2_page_size, gc_pos << 1,
                                    onetile << 1);
        noc_async_write_barrier();
        cb_pop_front(pc.cb_id, pc.frame_size);
        gc_pos += onetile;
      }
    }
  }
}

void kernel_main() {
  Global gc;
  gc.addr = get_arg_val<uint32>(0);
  gc.log2_page_size = get_arg_val<uint32>(1);
  Pipe pc;
  pc.cb_id = get_arg_val<uint32>(2);
  pc.frame_size = get_arg_val<uint32>(3);
  uint32 gc_pos = get_arg_val<uint32>(4);
  uint32 batch = get_arg_val<uint32>(5);
  uint32 Mt = get_arg_val<uint32>(6);
  uint32 Nt = get_arg_val<uint32>(7);
  kernel(gc, pc, gc_pos, batch, Mt, Nt);
}

