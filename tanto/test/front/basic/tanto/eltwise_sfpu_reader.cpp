// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gx,
        pipe<T> px,
        uint32 gx_pos,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        px.reserve_back();
        px.read(0, gx, gx_pos, block_items);
        read_barrier();
        px.push_back();
        gx_pos += block_items;
    }
}

