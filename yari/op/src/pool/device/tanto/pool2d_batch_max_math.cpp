// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void kernel(
        pipe<T> px,
        pipe<T> py,
        pipe<T> py_im,
        uint32 N,
        uint32 C,
        uint32 PQ,
        uint32 RS) {
    uint32 Ct = C / 32;
    px.set_frame(Ct);
    py.set_frame(Ct);
    py_im.set_frame(Ct);
    for (uint32 n = 0; n < N; n++) {
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            py_im.reserve_back();
            px.wait_front();
            for (uint32 c = 0; c < Ct; c++) {
                math<T> acc;
                acc.copy(px, c, 0);
                acc.pack(0, py_im);
            } // c
            px.pop_front();
            py_im.push_back();
            for (uint32 rs = 1; rs < RS - 1; rs++) {
                py_im.reserve_back();
                px.wait_front();
                py_im.wait_front();
                for (uint32 c = 0; c < Ct; c++) {
                    math<T> acc;
                    acc.copy(px, c, 0);
                    acc.copy(py_im, c, 1);
                    acc.max(0);
                    acc.pack(0, py_im);
                }
                px.pop_front();
                py_im.pop_front();
                py_im.push_back();
            } // rs
            py.reserve_back();
            px.wait_front();
            py_im.wait_front();
            for (uint32 c = 0; c < Ct; c++) {
                math<T> acc;
                acc.copy(px, c, 0);
                acc.copy(py_im, c, 1);
                acc.max(0);
                acc.pack(0, py);
            }
            px.pop_front();
            py_im.pop_front();
            py.push_back();
        } // pq_start
    } // n
}

