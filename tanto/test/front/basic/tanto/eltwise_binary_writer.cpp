// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gc, 
        pipe<T> pc, 
        uint32 gc_pos,
        uint32 num_blocks,
        uint32 block_tiles) { 
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        pc.wait_front();
        pc.write(0, gc, gc_pos, block_items);
        write_barrier();
        pc.pop_front();
        gc_pos += block_items;
    }
}

