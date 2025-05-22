// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void read_px(
        local<T> lx, 
        local<uint32> lp, 
        local<T> lzero, 
        pipe<T> px, 
        uint32 C,
        uint32 lp_pos) {
    px.reserve_back();
    px.move_init(C);
    uint32 px_pos = 0;
    for (uint32 i = 0; i < 32; i++) { 
        uint32 lx_pos = lp.get(lp_pos + i);
        if ((lx_pos >> 31) != 0) {
            px.move(px_pos, lzero, 0);
        } else {
            px.move(px_pos, lx, lx_pos);
        }
        px_pos += C;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gw,
        global<uint32> gp,
        global<T> gzero,
        local<T> lx,
        local<uint32> lp,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pw,
        uint32 N,
        uint32 C,
        uint32 HWC,
        uint32 PQ,
        uint32 zero_size,
        uint32 x_pos,
        uint32 x_stride) {
    px.set_frame(C / 32);
    pw.set_frame(4);
    lzero.read(0, gzero, 0, zero_size);
    read_barrier();
    uint32 x_start = x_pos;
    for (uint32 n = 0; n < N; n++) {
        lx.read(0, gx, x_start, HWC);
        read_barrier();
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            pw.reserve_back();
            pw.read(0, gw, pq_start * 4, 128);
            lp.read(0, gp, pq_start * 4, 128);
            read_barrier();
            pw.push_back();
            for (int i = 0; i < 128; i += 32) {
                read_px(lx, lp, lzero, px, C, i);
            }
        } // pq_start
        x_start += x_stride;
    }
}

