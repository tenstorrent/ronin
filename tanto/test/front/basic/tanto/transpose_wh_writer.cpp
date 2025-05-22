// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "writer_unary_8bank.cpp"

void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 gy_pos,
        uint32 num_tiles) {
    for (uint32 i = 0; i < num_tiles; i++) {
        py.wait_front();
        py.write(0, gy, gy_pos, 1024);
        write_barrier();
        py.pop_front();
        gy_pos += 1024;
    }
}

