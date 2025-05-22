// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "reader_unary.cpp"

void kernel(
        global<T> gx,
        pipe<T> px,
        uint32 gx_pos,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    px.set_frame(block_tiles);
    for (uint32 b = 0; b < num_blocks; b++) {
        px.reserve_back();
        px.read(0, gx, gx_pos, block_items);
        read_barrier();
        px.push_back();
        gx_pos += block_items;
    }
}

