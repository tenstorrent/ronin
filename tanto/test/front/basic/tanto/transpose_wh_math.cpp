// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "transpose_wh.cpp"

void kernel(pipe<T> px, pipe<T> py, uint32 NHtWt) {
    // transpose a row-major block
    // assumes the tiles come in in column major order from reader
    for (uint32 n = 0; n < NHtWt; n++) {
        px.wait_front();
        py.reserve_back();
        math<T> acc;
        acc.transpose(px, 0, 0);
        acc.pack(0, py);
        py.push_back();
        px.pop_front();
    }
}

