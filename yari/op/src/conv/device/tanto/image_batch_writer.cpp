// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        global<T> gw,
        global<T> gy,
        pipe<T> pw,
        pipe<T> py,
        uint32 N,
        uint32 K,
        uint32 PQ,
        uint32 KRSC_rnd,
        uint32 y_pos,
        uint32 y_stride) {
    pw.set_frame(KRSC_rnd / 1024);
    py.set_frame(K / 32);
    pw.reserve_back();
    pw.read(0, gw, 0, KRSC_rnd);
    read_barrier();
    pw.push_back();
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            py.wait_front();
            py.write(0, gy, y_start, K * 32);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_start
        y_start += y_stride;
    } // n
}

