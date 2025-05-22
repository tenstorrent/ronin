// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "reader_unary_8bank_reduce.cpp / reader_unary_8bank.cpp"

void kernel(
        global<T> gx,
        global<T> gs,
        pipe<T> px,
        pipe<T> ps,
        uint32 gx_pos,
        uint32 num_tiles) {
    ps.reserve_back();
    ps.read(0, gs, 0, 1024);
    read_barrier();
    ps.push_back();

    for (uint32 i = 0; i < num_tiles; i++) {
        px.reserve_back();
        px.read(0, gx, gx_pos, 1024);
        read_barrier();
        px.push_back();
        gx_pos += 1024;
    }
}

