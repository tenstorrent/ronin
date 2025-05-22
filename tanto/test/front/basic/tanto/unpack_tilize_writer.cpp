// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "writer_unary.cpp"

void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 gy_pos,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    py.set_frame(block_tiles);
    for (uint32 b = 0; b < num_blocks; b++) {
        py.wait_front();
        py.write(0, gy, gy_pos, block_items);
        write_barrier();
        py.pop_front();
        gy_pos += block_items;
    }
}

