// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gx,
        local<T> ly,
        uint32 xpos,
        uint32 ypos,
        uint32 H,
        uint32 C,
        uint32 Cb) {
    for (uint32 h = 0; h < H; h++) {
        ly.read(ypos, gx, xpos, Cb);
        xpos += C;
        ypos += Cb;
    }
    read_barrier();
}

