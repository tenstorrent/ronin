// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// originally "writer_unary_8bank.cpp"

void kernel(
        global<T> gc,
        pipe<T> pc,
        uint32 gc_pos,
        uint32 num_tiles) {
    constexpr uint32 onetile = 1024;
    for (uint32 i = 0; i < num_tiles; i++) {
        pc.wait_front();
        pc.write(0, gc, gc_pos, onetile);
        write_barrier();
        pc.pop_front();
        gc_pos += 1024;
    }
}

