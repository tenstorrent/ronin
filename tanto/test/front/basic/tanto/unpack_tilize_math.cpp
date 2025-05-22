// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "tilize.cpp"

void kernel(
        pipe<T> px,
        pipe<T> py,
        uint32 num_blocks,
        uint32 block_tiles) {
    px.set_frame(block_tiles);
    py.set_frame(block_tiles);
    for (uint32 b = 0; b < num_blocks; b++) {
        px.wait_front();
        py.reserve_back();
        tilize_block(px, block_tiles, py);
        py.push_back();
        px.pop_front();
    }
}

