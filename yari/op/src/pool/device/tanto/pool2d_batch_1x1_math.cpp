// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        pipe<T> px,
        pipe<T> py,
        uint32 N,
        uint32 C,
        uint32 PQ) {
    uint32 Ct = C / 32;
    px.set_frame(Ct);
    py.set_frame(Ct);
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            py.reserve_back();
            px.wait_front();
            for (uint32 c = 0; c < Ct; c++) {
                math<T> acc;
                acc.copy(px, c, 0);
                acc.pack(0, py);
            } // c
            px.pop_front();
            py.push_back();
        } // pq_start
    } // n
}

