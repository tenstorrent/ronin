// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

param<uint32> K;
param<uint32> PQ;
param<uint32> RSKC;
param<uint32> y_stride;

void kernel(
        global<T> gw,
        global<T> gz,
        global<T> gy,
        pipe<T> pw,
        pipe<T> pz,
        pipe<T> py,
        uint32 N,
        uint32 y_pos) {
    pw.set_frame(RSKC / 1024);
    pz.set_frame(K / 32);
    py.set_frame(K / 32);
    pw.reserve_back();
    pw.read(0, gw, 0, RSKC);
    read_barrier();
    pw.push_back();
    uint32 y_start = y_pos;
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            pz.reserve_back();
            pz.read(0, gz, y_start, K * 32);
            read_barrier();
            pz.push_back();
            py.wait_front();
            py.write(0, gy, y_start, K * 32);
            write_barrier();
            py.pop_front();
            y_start += K * 32;
        } // pq_start
        y_start += y_stride;
    } // n
}

