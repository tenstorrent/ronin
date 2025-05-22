// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 N,
        uint32 W,
        uint32 y_pos,
        uint32 y_stride) {
    py.set_frame(W / 32);
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        py.wait_front();
        py.write(0, gy, y_start, W * 32);
        write_barrier();
        py.pop_front();
        y_start += y_stride;
    }
}

