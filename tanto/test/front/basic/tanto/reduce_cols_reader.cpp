// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// originally "reader_unary_transpose_wh_interleaved.cpp"

void kernel(
        global<T> gx,
        global<T> gs,
        pipe<T> px,
        pipe<T> ps,
        uint32 gx_pos,
        uint32 N,
        uint32 Ht,
        uint32 Wt,
        uint32 HtWt) {
    uint32 gx_dh = Wt * 1024;
    uint32 gx_dw = HtWt * 1024;

    ps.reserve_back();
    ps.read(0, gs, 0, 1024);
    read_barrier();
    ps.push_back();

    // read NHW tensor in NWH order
    for (uint32 n = 0; n < N; n++) {
        uint32 gx_idx = gx_pos;
        for (uint32 w = 0; w < Wt; w++) {
            for (uint32 h = 0; h < Ht; h++) {
                px.reserve_back();
                px.read(0, gx, gx_idx, 1024);
                read_barrier();
                px.push_back();
                gx_idx += gx_dh;
            }
            gx_idx -= gx_dw;
            gx_idx += 1024;
        }
        gx_pos += gx_dw;
    }
}

