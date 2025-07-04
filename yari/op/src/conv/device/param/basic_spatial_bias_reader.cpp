// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

param<uint32> C;
param<uint32> K;
param<uint32> R;
param<uint32> S;
param<uint32> PQ;
param<uint32> start_q;
param<uint32> delta_p;
param<uint32> delta_q;
param<uint32> delta_r;
param<uint32> delta_s;
param<uint32> end_q;
param<uint32> zero_size;
param<uint32> mask_size;
param<uint32> x_stride;

void read_px(
        global<T> gx,
        local<T> lzero,
        pipe<T> px,
        uint32 start,
        uint32 &p_term,
        uint32 &q_term,
        uint32 r_term, 
        uint32 s_term,
        uint32 mask) {
    uint32 dst_pos = 0;
    px.reserve_back();
    for (uint32 i = 0; i < 32; i++) {
        if ((mask & 1) == 0) {
            px.read(dst_pos, lzero, 0, C);
        } else {
            uint32 src_pos = start + p_term + q_term + r_term + s_term;
            px.read(dst_pos, gx, src_pos, C);
        }
        q_term += delta_q;
        if (q_term >= end_q) {
            q_term = start_q;
            p_term += delta_p;
        }
        mask >>= 1;
        dst_pos += C;
    }
    read_barrier();
    px.push_back();
}

void kernel(
        global<T> gx,
        global<T> gb,
        global<T> gzero,
        global<uint32> gmask,
        local<T> lzero,
        local<uint32> lmask,
        pipe<T> px,
        pipe<T> pb,
        uint32 N,
        uint32 start_p,
        uint32 x_pos,
        uint32 mask_pos) {
    lzero.read(0, gzero, 0, zero_size);
    lmask.read(0, gmask, 0, mask_size);
    // read_barrier is below
    px.set_frame(C / 32);
    pb.set_frame(K / 32);
    pb.reserve_back();
    pb.read(0, gb, 0, K * 32);
    read_barrier();
    pb.push_back();
    uint32 x_start = x_pos;
    for (uint32 n = 0; n < N; n++) {
        uint32 p_start = start_p;
        uint32 q_start = start_q;
        uint32 p_term = 0;
        uint32 q_term = 0;
        uint32 mask_start = mask_pos;
        for (uint32 pq_start = 0; pq_start < PQ; pq_start += 32) {
            uint32 r_term = 0;
            for (uint32 r = 0; r < R; r++) {
                uint32 s_term = 0;
                for (uint32 s = 0; s < S; s++) {
                    p_term = p_start;
                    q_term = q_start;
                    uint32 mask = lmask.get(mask_start);
                    read_px(
                        gx, 
                        lzero,
                        px,
                        x_start, 
                        p_term, 
                        q_term, 
                        r_term, 
                        s_term, 
                        mask);
                    mask_start++;
                    s_term += delta_s;
                } // s
                r_term += delta_r;
            } // r
            p_start = p_term;
            q_start = q_term;
        } // pq_start
        x_start += x_stride;
    } // n
}

