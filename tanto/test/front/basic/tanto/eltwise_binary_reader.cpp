// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> ga,
        global<T> gb,
        pipe<T> pa,
        pipe<T> pb,
        uint32 ga_pos,
        uint32 gb_pos,
        uint32 num_blocks,
        uint32 block_tiles) {
    uint32 block_items = block_tiles * 1024;
    for (uint32 i = 0; i < num_blocks; i++) {
        pa.reserve_back();
        pb.reserve_back();
        pa.read(0, ga, ga_pos, block_items);
        pb.read(0, gb, gb_pos, block_items);
        read_barrier();
        pa.push_back();
        pb.push_back();
        ga_pos += block_items;
        gb_pos += block_items;
    }
}

