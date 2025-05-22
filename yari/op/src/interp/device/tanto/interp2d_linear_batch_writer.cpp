// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gy,
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 y_pos,
        uint32 y_stride) {
    py.set_frame(C / 32);
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            py.wait_front();
            py.write(0, gy, y_start, C * 32);
            write_barrier();
            py.pop_front();
            y_start += C * 32;
        } // pq_start
    } // n
}

