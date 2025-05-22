// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gy, 
        pipe<T> py, 
        uint32 gy_pos,
        uint32 num_blocks,
        uint32 block_tiles) { 
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        py.wait_front();
        py.write(0, gy, gy_pos, block_items);
        write_barrier();
        py.pop_front();
        gy_pos += block_items;
    }
}

