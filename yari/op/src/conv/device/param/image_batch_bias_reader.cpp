// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

param<uint32> H;
param<uint32> K;
param<uint32> R;
param<uint32> WC;
param<uint32> PQ;
param<uint32> SC;
param<uint32> RSC_rnd;
param<uint32> before_h;
param<uint32> after_h;
param<uint32> before_wc;
param<uint32> after_wc;
param<uint32> offset_wc;
param<uint32> before_hwc;
param<uint32> delta_p;
param<uint32> delta_q;
param<uint32> delta_r;
param<uint32> end_q;
param<uint32> x_stride;
param<uint32> zero_size;

void clear_lx(local<T> lx, local<T> lzero) {
    uint32 WC_full = before_wc + WC + after_wc;
    uint32 pos = 0;
    for (uint32 i = 0; i < before_h; i++) {
        lx.read(pos, lzero, 0, WC_full);
        pos += WC_full;
    }
    for (uint32 i = 0; i < H; i++) {
        if (before_wc != 0) {
            lx.read(pos, lzero, 0, before_wc);
        }
        pos += before_wc + WC;
        if (after_wc != 0) {
            lx.read(pos, lzero, 0, after_wc);
        }
        pos += after_wc;
    }
    for (uint32 i = 0; i < after_h; i++) {
        lx.read(pos, lzero, 0, WC_full);
        pos += WC_full;
    }
    read_barrier();
}

void load_lx(
        global<T> gx,
        local<T> lx,
        uint32 x_start) {
    uint32 pos = before_hwc + before_wc;
    uint32 stride = WC + after_wc + before_wc;
    for (uint32 i = 0; i < H; i++) {
        lx.read(pos, gx, x_start, WC);
        x_start += WC;
        pos += stride;
    }
    read_barrier();
}

void read_px(
        local <T> lx,
        pipe<T> px,
        uint32 &p_term,
        uint32 &q_term) {
    uint32 dst_start = 0;
    px.reserve_back();
    px.move_init(SC);
    for (uint32 i = 0; i < 32; i++) {
        uint32 dst_pos = dst_start;
        uint32 r_term = 0;
        for (uint32 r = 0; r < R; r++) {
            uint32 src_pos = p_term + q_term + r_term + offset_wc;
            px.move(dst_pos, lx, src_pos);
            dst_pos += SC;
            r_term += delta_r;
        }
        dst_start += RSC_rnd;
        q_term += delta_q;
        if (q_term >= end_q) {
            q_term = 0;
            p_term += delta_p;
        }
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        local<T> lx,
        local<T> lzero,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 x_pos) {
    lzero.read(0, gzero, 0, zero_size);
    // read_barrier is below
    px.set_frame(RSC_rnd / 32);
    pb.set_frame(K / 32);
    pb.reserve_back();
    pb.read(0, gb, 0, K * 32);
    read_barrier();
    pb.push_back();
    clear_lx(lx, lzero);
    uint32 x_start = x_pos;
    uint32 p_term = 0;
    uint32 q_term = 0;
    for (uint32 n = 0; n < N; n++) {
        load_lx(gx, lx, x_start);
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            read_px(lx, px, p_term, q_term);
        } // pq_start
        x_start += x_stride;
    } // n
}

