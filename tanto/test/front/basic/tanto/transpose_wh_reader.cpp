// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Originally "reader_unary_transpose_wh.cpp"

void kernel(
        global<T> gx,
        pipe<T> px,
        uint32 gx_pos,
        uint32 N,
        uint32 Ht,
        uint32 Wt,
        uint32 HtWt) {
    constexpr uint32 onetile = 1024;
    for (uint32 n = 0; n < N; n++) {
        uint32 gx_idx = gx_pos;
        for (uint32 w = 0; w < Wt; w++) {
            for (uint32 h = 0; h < Ht; h++) {
                px.reserve_back();
                px.read(0, gx, gx_idx, onetile);
                read_barrier();
                px.push_back();
                gx_idx += Wt * onetile;
            }
            gx_idx -= HtWt * onetile;
            gx_idx += onetile;
        }
        gx_pos += HtWt * onetile;
    }
}

