// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        local<T> lx,
        global<T> gy,
        uint32 xpos,
        uint32 ypos,
        uint32 H,
        uint32 C,
        uint32 Cb) {
    for (uint32 h = 0; h < H; h++) {
        lx.write(xpos, gy, ypos, Cb); 
        xpos += Cb;
        ypos += C;
    }
    write_barrier();
}

